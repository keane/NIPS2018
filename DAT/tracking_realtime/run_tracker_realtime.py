import numpy as np
import os

import sys
import time
import argparse
import json
from PIL import Image

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable

sys.path.insert(0, '../modules')
sys.path.insert(0, '../tracking')
from sample_generator import *
from data_prov import *
from model import *
from bbreg import *
from options import *
from gen_config import *
from run_tracker import *

np.random.seed(123)
torch.manual_seed(456)
torch.cuda.manual_seed(789)
import guided_backprop

import scipy
import gc

# import cv
import cv2
# import fps statistics
sys.path.insert(0, 'utils')
from app_utils import FPS

# override the para for cuda in run_tracker.py
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

refPt = []
cropping = False
isSelected = False


# imshow mouse callback
def click_and_crop(event, x, y, flags, param):
    global refPt, cropping, isSelected
    if isSelected :
        pass

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cropping = False
        isSelected = True


# fix size for image for network param propose
image_height = 352
image_width = 623

# from cv image to pil image
def fromCVToPIL(init_image): 
    image = cv2.cvtColor(init_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    return image


# real time tracker method
# get image from video source
def run_mdnet_realtime(img_list, cam, init_image, init_bbox, gt=None, savefig_dir='', display=False):
    # Init bbox
    target_bbox = np.array(init_bbox)

    result = target_bbox
    result_bb = target_bbox

    # Init model
    model = MDNet(opts['model_path'])
    if opts['use_gpu']:
        model = model.cuda()
    model.set_learnable_params(opts['ft_layers'])
    GBP = guided_backprop.GuidedBackprop(model, 1)
    # Init criterion and optimizer
    criterion = BinaryLoss()
    init_optimizer = set_optimizer(model, opts['lr_init'])
    update_optimizer = set_optimizer(model, opts['lr_update'])

    tic = time.time()

    # Load first image
    # modify by jerry
    image = fromCVToPIL(init_image)

    # Train bbox regressor
    bbreg_examples = gen_samples(SampleGenerator('uniform', image.size, 0.3, 1.5, 1.1),
                                 target_bbox, opts['n_bbreg'], opts['overlap_bbreg'], opts['scale_bbreg'])
    bbreg_feats = forward_samples(model, image, bbreg_examples)
    bbreg = BBRegressor(image.size)
    bbreg.train(bbreg_feats, bbreg_examples, target_bbox)

    # Draw pos/neg samples
    pos_examples = gen_samples(SampleGenerator('gaussian', image.size, 0.1, 1.2),
                               target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])

    neg_examples = np.concatenate([
        gen_samples(SampleGenerator('uniform', image.size, 1, 2, 1.1),
                    target_bbox, opts['n_neg_init'] // 2, opts['overlap_neg_init']),
        gen_samples(SampleGenerator('whole', image.size, 0, 1.2, 1.1),
                    target_bbox, opts['n_neg_init'] // 2, opts['overlap_neg_init'])])
    neg_examples = np.random.permutation(neg_examples)

    # Extract pos/neg features
    pos_feats = forward_samples(model, image, pos_examples)
    neg_feats = forward_samples(model, image, neg_examples)

    pos_imgids=np.array([[0]]*pos_feats.size(0))
    neg_imgids=np.array([[0]]*neg_feats.size(0))

    feat_dim = pos_feats.size(-1)

    # Initial training
    train(model, criterion, init_optimizer, pos_feats, neg_feats, opts['maxiter_init'],
            pos_imgids, pos_examples, neg_imgids, neg_examples, img_list, GBP)

    # Init sample generators
    sample_generator = SampleGenerator('gaussian', image.size, opts['trans_f'], opts['scale_f'], valid=True)
    pos_generator = SampleGenerator('gaussian', image.size, 0.1, 1.2)
    neg_generator = SampleGenerator('uniform', image.size, 1.5, 1.2)

    # Init pos/neg features for update
    pos_feats_all = [pos_feats[:opts['n_pos_update']]]
    neg_feats_all = [neg_feats[:opts['n_neg_update']]]

    pos_examples_all=[pos_examples[:opts['n_pos_update']]]
    neg_examples_all=[neg_examples[:opts['n_neg_update']]]

    pos_imgids_all = [pos_imgids[:opts['n_pos_update']]]
    neg_imgids_all = [neg_imgids[:opts['n_neg_update']]]



    spf_total = time.time() - tic

    # Display
    savefig = savefig_dir != ''

    # modify by jerry
    i = -1
    # start fps
    fps = FPS().start()
    # end modify

    while True:
        # modify by jerry
        # get image from vidoe source and convert to PIL image
        ret, image_cv = cam.read()
        if image_cv is None:
            continue
        croped = np.zeros((image_height, image_width))
        image_cv = np.array(image_cv)[:image_height, :image_width, :]
        image = fromCVToPIL(image_cv)
        # end modify

        i += 1
        tic = time.time()
        # Estimate target bbox
        samples = gen_samples(sample_generator, target_bbox, opts['n_samples'])
        sample_scores = forward_samples(model, image, samples, out_layer='fc6')
        top_scores, top_idx = sample_scores[:, 1].topk(5)
        top_idx = top_idx.cpu().numpy()
        target_score = top_scores.mean()
        target_bbox = samples[top_idx].mean(axis=0)

        success = target_score > opts['success_thr']

        # Expand search area at failure
        if success:
            sample_generator.set_trans_f(opts['trans_f'])
        else:
            sample_generator.set_trans_f(opts['trans_f_expand'])

        # Bbox regression
        if success:
            bbreg_samples = samples[top_idx]
            bbreg_feats = forward_samples(model, image, bbreg_samples)
            bbreg_samples = bbreg.predict(bbreg_feats, bbreg_samples)
            bbreg_bbox = bbreg_samples.mean(axis=0)
        else:
            bbreg_bbox = target_bbox

        # Copy previous result at failure
        if not success:
            target_bbox = result
            bbreg_bbox = result_bb

        # Save result
        result = target_bbox
        result_bb = bbreg_bbox

        # Data collect
        if success:
            # Draw pos/neg samples
            pos_examples = gen_samples(pos_generator, target_bbox,
                                       opts['n_pos_update'],
                                       opts['overlap_pos_update'])
            neg_examples = gen_samples(neg_generator, target_bbox,
                                       opts['n_neg_update'],
                                       opts['overlap_neg_update'])

            # Extract pos/neg features
            pos_feats = forward_samples(model, image, pos_examples)
            neg_feats = forward_samples(model, image, neg_examples)
            pos_feats_all.append(pos_feats)
            neg_feats_all.append(neg_feats)

            pos_examples_all.append(pos_examples)
            neg_examples_all.append(neg_examples)

            pos_imgids_all.append(np.array([[i]]*pos_feats.size(0)))
            neg_imgids_all.append(np.array([[i]]*neg_feats.size(0)))



            if len(pos_feats_all) > opts['n_frames_long']:
                del pos_feats_all[0]
                del pos_examples_all[0]
                del pos_imgids_all[0]
            if len(neg_feats_all) > opts['n_frames_short']:
                del neg_feats_all[0]
                del neg_examples_all[0]
                del neg_imgids_all[0]



        # Short term update
        if not success:
            nframes = min(opts['n_frames_short'], len(pos_feats_all))
            pos_data = torch.stack(pos_feats_all[-nframes:], 0).view(-1, feat_dim)
            neg_data = torch.stack(neg_feats_all, 0).view(-1, feat_dim)

            pos_examples_data=torch.from_numpy(np.stack(pos_examples_all[-nframes:],0)).view(-1, 4).numpy()
            neg_examples_data=torch.from_numpy(np.stack(neg_examples_all,0)).view(-1, 4).numpy()

            pos_imgids_data=torch.from_numpy(np.stack(pos_imgids_all[-nframes:],0)).view(-1, 1).numpy()
            neg_imgids_data=torch.from_numpy(np.stack(neg_imgids_all,0)).view(-1, 1).numpy()


            train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'],
                  pos_imgids_data, pos_examples_data, neg_imgids_data, neg_examples_data,img_list,GBP)

        # Long term update
        elif i % opts['long_interval'] == 0:
            pos_data = torch.stack(pos_feats_all, 0).view(-1, feat_dim)
            neg_data = torch.stack(neg_feats_all, 0).view(-1, feat_dim)

            pos_examples_data = torch.from_numpy(np.stack(pos_examples_all, 0)).view(-1, 4).numpy()
            neg_examples_data = torch.from_numpy(np.stack(neg_examples_all, 0)).view(-1, 4).numpy()

            pos_imgids_data = torch.from_numpy(np.stack(pos_imgids_all, 0)).view(-1, 1).numpy()
            neg_imgids_data = torch.from_numpy(np.stack(neg_imgids_all, 0)).view(-1, 1).numpy()

            train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'],
                  pos_imgids_data, pos_examples_data, neg_imgids_data, neg_examples_data, img_list,GBP)

        spf = time.time() - tic
        spf_total += spf

        # modify by jerry
        fps.update()

        # get bbox and draw image
        x = result_bb[0]
        y = result_bb[1]
        w = result_bb[2]
        h = result_bb[3]

        cv2.rectangle(image_cv, (x,y), (x+w,y+h), (0,255,0), 2)
        text = 'approx. FPS: {:.2f}'.format(fps.fps())
        cv2.putText(image_cv, text, (0, 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 3)
        cv2.imshow('tracking', image_cv)
        cv2.waitKey(1)
        # end modify

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seq', default='', help='input seq')
    parser.add_argument('-j', '--json', default='', help='input json')
    parser.add_argument('-f', '--savefig', action='store_true')
    parser.add_argument('-d', '--display', action='store_true')

    args = parser.parse_args()
    args.display = False
    args.savefig = True
    assert (args.seq != '' or args.json != '')

    # Generate sequence config
    img_list, init_bbox, gt, savefig_dir, display, result_path = gen_config(args)

    vidoe_source = 0

    cam = cv2.VideoCapture(vidoe_source)
    cam.set(3, 640)
    cam.set(4, 480)

    init_image = None
    stopFlag = False
    box_string = ''
    cv2.namedWindow('tracking')
    cv2.setMouseCallback('tracking', click_and_crop)

    while init_image is None and not stopFlag:
        ret, current_image = cam.read()
        croped = np.zeros((image_height, image_width))
        croped = np.array(current_image)[:image_height, :image_width, :]
        if isSelected :
            init_image = croped.copy()
            print refPt
            cv2.rectangle(croped, (refPt[0][0],refPt[0][1]), (refPt[1][0],refPt[1][1]), (0,255,0), 2)
            break

        cv2.imshow('tracking', croped)
        key = cv2.waitKey(1)
        # esc key
        if key == 27:
            stopflag = True

    init_bbox = [refPt[0][0], refPt[0][1], refPt[1][0] - refPt[0][0], refPt[1][1]-refPt[0][1]]
    print(init_bbox)

    run_mdnet_realtime(None, cam, init_image, init_bbox, gt=gt, savefig_dir=savefig_dir, display=display)

    cam.release()

