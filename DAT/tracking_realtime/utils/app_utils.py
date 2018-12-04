#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# From http://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/

import struct
import six
import collections
import cv2
import datetime
from threading import Thread
from matplotlib import colors
import numpy as np
import serial
import serial.tools.list_ports


class OBJECT_DETECT:
    def __init__(self,_Xoffset,_Yoffset,_depth,_numBox,_numSuccessiveDetect,_arrAreaFilter,_flagPrint):
        self._Xoffset = _Xoffset
        self._Yoffset = _Yoffset
        self._depth = _depth
        self._numBox = _numBox
        self._numSuccessiveDetect = _numSuccessiveDetect
        self._arrAreaFilter = _arrAreaFilter
        self._flagPrint = _flagPrint

    # def arrAreaFilter(self,_arrAreaFilter):
        # self._arrAreaFilter = _arrAreaFilter
        # return self

    # def numSuccessiveDetect(self,_numBox):
        # self._numSuccessiveDetect = _numBox

    # def update(self):
        # self._numFrames += 1

    # def elapsed(self):
        # return (self._end - self._start).total_seconds()

    # def fps(self):
        # return self._numFrames / self.elapsed()


class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1
        self._end = datetime.datetime.now()

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()


class WebcamVideoStream:
    def __init__(self, src, width, height):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


def standard_colors():
    colors = [
        'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
        'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
        'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
        'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
        'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
        'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
        'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
        'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
        'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
        'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
        'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
        'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
        'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
        'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
        'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
        'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
        'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
        'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
        'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
        'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
        'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
        'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
        'WhiteSmoke', 'Yellow', 'YellowGreen'
    ]
    return colors

def depth_LUT():
    depth = [
        1428, 1360, 1239, 1214, 1189, 1118, 1089, 1063, 1043,
        989, 937, 921, 906, 844, 836, 790, 771, 759, 748,
        720, 711, 701, 692, 663, 638, 620, 603, 586, 572,
        565, 558, 551, 544, 538, 524, 498, 483, 477, 471,
        465, 459, 452, 446, 440, 431, 423, 415, 406, 401,
        396, 390, 385, 380, 375, 369, 364, 359, 354, 349,
        343, 338, 333, 328, 323, 319, 315, 312, 309, 305,
        302, 298, 295, 291, 288, 285, 283, 280, 278, 275,
        272, 270, 267, 265, 262, 260, 257, 254, 251, 248,
        246, 243, 240, 237, 234, 231, 228, 225, 223, 220,
        218, 217, 215, 213, 212, 210, 208, 207, 205, 204,
        202, 200, 199, 197, 195, 194, 192, 190, 189, 187,
        185, 183, 181, 179
    ]
    return depth

def color_name_to_rgb():
    colors_rgb = []
    for key, value in colors.cnames.items():
        colors_rgb.append((key, struct.unpack('BBB', bytes.fromhex(value.replace('#', '')))))
    return dict(colors_rgb)


def draw_boxes_and_labels(
        boxes,
        classes,
        scores,
        category_index,
        instance_masks=None,
        keypoints=None,
        max_boxes_to_draw=20,
        min_score_thresh=.5,
        agnostic_mode=False):
    """Returns boxes coordinates, class names and colors

    Args:
      boxes: a numpy array of shape [N, 4]
      classes: a numpy array of shape [N]
      scores: a numpy array of shape [N] or None.  If scores=None, then
        this function assumes that the boxes to be plotted are groundtruth
        boxes and plot all boxes as black with no classes or scores.
      category_index: a dict containing category dictionaries (each holding
        category index `id` and category name `name`) keyed by category indices.
      instance_masks: a numpy array of shape [N, image_height, image_width], can
        be None
      keypoints: a numpy array of shape [N, num_keypoints, 2], can
        be None
      max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
        all boxes.
      min_score_thresh: minimum score threshold for a box to be visualized
      agnostic_mode: boolean (default: False) controlling whether to evaluate in
        class-agnostic mode or not.  This mode will display scores but ignore
        classes.
    """
    # Create a display string (and color) for every box location, group any boxes
    # that correspond to the same location.
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
    box_to_class_value = collections.defaultdict(str)
    box_to_instance_masks_map = {}
    box_to_keypoints_map = collections.defaultdict(list)
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            # print('-------box :',box)
            if instance_masks is not None:
                box_to_instance_masks_map[box] = instance_masks[i]
            if keypoints is not None:
                box_to_keypoints_map[box].extend(keypoints[i])
            if scores is None:
                box_to_color_map[box] = 'black'
            else:
                if not agnostic_mode:
                    if classes[i] in category_index.keys():
                        # print('---------classes[i]：',classes[i])
                        class_name = category_index[classes[i]]['name']
                    else:
                        class_name = 'N/A'
                    display_str = '{}: {}%'.format(
                        class_name,
                        int(100 * scores[i]))
                else:
                    display_str = 'score: {}%'.format(int(100 * scores[i]))
                box_to_display_str_map[box].append(display_str)
                box_to_class_value[box]=classes[i]
                if agnostic_mode:
                    box_to_color_map[box] = 'DarkOrange'
                else:
                    box_to_color_map[box] = standard_colors()[
                        classes[i] % len(standard_colors())]

    # print('-----------box:',box)
    # Store all the coordinates of the boxes, class names and colors
    color_rgb = color_name_to_rgb()
    rect_points = []
    class_names = []
    class_value = []
    class_colors = []
    # print('-----------box_to_color_map:',box_to_color_map)
    # print('-----------box_to_display_str_map:',box_to_display_str_map)
    # print('-----------box_to_class_value:',box_to_class_value)
    for box, color in six.iteritems(box_to_color_map):
        ymin, xmin, ymax, xmax = box
        rect_points.append(dict(ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax))
        class_names.append(box_to_display_str_map[box])
        class_value.append(box_to_class_value[box])
        class_colors.append(color_rgb[color.lower()])
    return rect_points, class_names, class_colors, class_value



def k_LUT():
    k = [
        0.904, 0.974, 1.042, 1.111, 1.179, 1.247, 1.315, 1.382, 1.449, 1.516,
        1.582, 1.649, 1.724, 1.804, 1.888, 1.972, 2.053, 2.129, 2.196, 2.252,
        2.293, 2.317, 2.324, 2.327, 2.330, 2.332, 2.333, 2.334, 2.336, 2.337,
        2.338, 2.340, 2.342, 2.345, 2.349, 2.375, 2.440, 2.539, 2.666, 2.816,
        2.983, 3.162, 3.347, 3.533, 3.714, 3.905, 4.122, 4.358, 4.611, 4.875,
        5.145, 5.416, 5.685, 5.945, 6.201, 6.459, 6.718, 6.980, 7.243, 7.508,
        7.774, 8.042, 8.312
    ]
    return k


def b_LUT():
    b = [
        3.291, 3.419, 3.420, 3.303, 3.081, 2.763, 2.360, 1.884, 1.346, 0.756,
        0.126, -0.639, -2.090, -4.168, -6.701, -9.517, -12.444, -15.312, -17.947, -20.180,
        -21.837, -22.748, -22.749, -21.845, -20.187, -17.935, -15.247, -12.282, -9.197, -6.153,
        -3.306, -0.817, 1.158, 2.459, 2.928, 2.912, 2.868, 2.804, 2.726, 2.641,
        2.557, 2.479, 2.415, 2.371, 2.355, 2.416, 2.580, 2.818, 3.099, 3.395,
        3.676, 3.914, 4.078, 4.139, 3.984, 3.546, 2.864, 1.977, 0.925, -0.253,
        -1.518, -2.831, -4.151
    ]
    return b

def cal_area_center(
        num,
        point,
        width,
        height,
        filter_arr,
        numSuccessiveDetect,
        # instance_masks=None,
        # keypoints=None,
        # max_boxes_to_draw=20,
        OFFSET_INVALID=127,
        print_mode=True
        ):

    num = num+1
    # if print_mode :
        # print('------xmax:',round(point['xmax'] * width))
        # print('------xmin:',round(point['xmin'] * width))
        # print('------ymax:',round(point['ymax'] * height))
        # print('------ymin:',round(point['ymin'] * height))
    area= round( width*height*(point['xmax']-point['xmin'])*(point['ymax']-point['ymin']) )
    if print_mode :
        print('------area:',area)

    filter_arr[0,0:2+1] = filter_arr[0,1:3+1]
    filter_arr[0,3]   = area
    area = np.mean(filter_arr[0,0:3+1])
    if print_mode :
        print('------mean area:',area)

    if 576 < area < 21316 :
        area = int(round(area**0.5))
        if print_mode :
            print('------lut index:',area)

        depth = depth_LUT()[
                (area-24) % len(depth_LUT())]
        # depth_low  = depth & 0xFF # 强制截断
        # depth_high = (depth & 0xFF00)/256 # 强制截断
        if print_mode :
            print('------ depth lut value:',depth)
            # print('------ depth_high:',depth_high)
            # print('------ depth_low :',depth_low )
    else :
        depth = 2550

    # center = round(0-((point['xmax']+point['xmin'])/2*640-320 )/2)
    center = round(((point['xmax']+point['xmin'])/2*640-320 )/2)
    filter_arr[1,0:2+1] = filter_arr[1,1:3+1]
    filter_arr[1,3]   = center
    center = np.mean(filter_arr[1,2:3+1])

    if print_mode :
        print('------ center:',center)

    if 150 < depth < 1400 :
        k = k_LUT()[
                (round((depth-150)/20)) % len(k_LUT())]
        if print_mode :
            print('------ k lut value :',k)

        b = b_LUT()[
                (round((depth-150)/20)) % len(b_LUT())]
        if print_mode :
            print('------ b lut value :',b)

        offset = round(k*abs(center) + b)
        if center < 0 :
            offset = 0-offset
        if print_mode :
            print('------ offset :',offset)

        if abs(offset) < OFFSET_INVALID*2-3 :
            offset = round(offset/2)
        else :
            offset = OFFSET_INVALID
            print('------ offset overflow')
    else :
        print('------ depth overflow')
        offset = OFFSET_INVALID

    depth = round(depth/10)
    print('------ depth/10 :',depth)
    if offset < 0 :
        offset = offset + 256


    # if offset==OFFSET_INVALID and depth==255:
        # flag = False
    # else :
        # flag = True


    # if flag==True :
        # if numSuccessiveDetect<6 :
            # numSuccessiveDetect +=1;
    # else :
        # if numSuccessiveDetect>0 :
            # numSuccessiveDetect -=1;

    if numSuccessiveDetect<5 :
        numSuccessiveDetect +=1;
    if print_mode :
        print('------ numSuccessiveDetect',numSuccessiveDetect)

    return num,filter_arr,int(depth),int(offset),numSuccessiveDetect

def cal_center(
        num,
        point,
        width,
        height,
        filter_arr,
        numSuccessiveDetect,
        # instance_masks=None,
        # keypoints=None,
        # max_boxes_to_draw=20,
        OFFSET_INVALID=127,
        print_mode=True
        ):

    num = num+1
    # if print_mode :
        # print('------xmax:',round(point['xmax'] * width))
        # print('------xmin:',round(point['xmin'] * width))
        # print('------ymax:',round(point['ymax'] * height))
        # print('------ymin:',round(point['ymin'] * height))

    # center = round(0-((point['xmax']+point['xmin'])/2*640-320 )/2)
    center = round(((point['xmax']+point['xmin'])/2*640-320 )/2)
    filter_arr[1,0:2+1] = filter_arr[1,1:3+1]
    filter_arr[1,3]   = center
    center = np.mean(filter_arr[1,2:3+1])
    if print_mode :
        print('------ center:',center)

    Xoffset = int(round(center/4))
    if Xoffset < 0 :
        Xoffset = Xoffset + 256
    if print_mode :
        print('------ Xoffset:',Xoffset)



    if numSuccessiveDetect<5 :
        numSuccessiveDetect +=1;
    if print_mode :
        print('------ numSuccessiveDetect',numSuccessiveDetect)

    return num,filter_arr,Xoffset,numSuccessiveDetect

# class MSerialPort:
	# message='default'
	# def __init__(self,name,buand):
        # self.port = serial.Serial(port=name, baudrate=buand, timeout=2)

	# def port_close(self):
		# self.port.close()
	# def send_data(self,data):
		# number=self.port.write(data)
		# return number
    # def read_data(self):
        # while True:
            # data=self.port.readline()
            # self.message+=data
