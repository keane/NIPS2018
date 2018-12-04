import os, sys
import scipy.io as sio
seqs = sio.loadmat('../tracking/OTB50.mat')
#seqs = sio.loadmat('OTB51-100.mat')
for i in range(len(seqs['seqs'][0])):
    name = str(seqs['seqs'][0][i][0][0][0][0])
    if name != "football":

        continue

    os.system("/home/kiktech/.conda/envs/lane_segmentation/bin/python run_tracker_realtime.py -s "+name)

