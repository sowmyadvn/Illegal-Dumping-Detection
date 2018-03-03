# Stupid python path shit.
# Instead just add darknet.py to somewhere in your python path
# OK actually that might not be a great idea, idk, work in progress
# Use at your own risk. or don't, i don't care

import os, sys
import numpy as np
import glob
from PIL import Image
from scipy.misc import imread
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
sys.path.append(os.path.join(os.getcwd(),'python/'))

import darknet as dn
import pdb
from datetime import datetime

dn.set_gpu(0)
net = dn.load_net(b"cfg/yolo-obj.cfg", b"yolo-obj_40000.weights", 0)
meta = dn.load_meta(b"cfg/obj.data")
folder_raw = "/Users/sowmiteja/Desktop/Raw_Images"
files = os.listdir(folder_raw)
folder_classified = "/Users/sowmiteja/Desktop/Classified_Images/"
count = 0
file1 = open(os.path.join(folder_classified,datetime.now().strftime("%Y%m%d-%H%M%S")+".txt"),"w")
images_list = glob.glob(folder_raw)

def array_to_image(arr):
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = (arr/255.0).flatten()
    data = dn.c_array(dn.c_float, arr)
    im = dn.IMAGE(w,h,c,data)
    return im

def detect2(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    boxes = dn.make_boxes(net)
    probs = dn.make_probs(net)
    num =   dn.num_boxes(net)
    dn.network_detect(net, image, thresh, hier_thresh, nms, boxes, probs)
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if probs[j][i] > 0:
                res.append((meta.names[i], probs[j][i], (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))
    res = sorted(res, key=lambda x: -x[1])
    dn.free_ptrs(dn.cast(probs, dn.POINTER(dn.c_void_p)), num)
    return res

for f in files:
   if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png"):
        print (f)
        path = os.path.join(folder_raw, f)
        #path = bytes(os.path.join(folder_raw, f).encode("utf-8"))
        arr = cv2.imread(path)
        im = array_to_image(arr)
        dn.rgbgr_image(im)
        r = detect2(net, meta, im)
        print (r)
        cnt = 0
        if r is not None:
             while cnt < len(r):
                  name = r[cnt][0]
                  predict = r[cnt][1]
                  print (name+":"+str(predict))
                  x = r[cnt][2][0]
                  y = r[cnt][2][1]
                  w = r[cnt][2][2]
                  z = r[cnt][2][3]

                  x_max = (2*x+w)/2
                  x_min = (2*x-w)/2
                  y_min = (2*y-z)/2
                  y_max = (2*y+z)/2
                  print (x_min, y_min, x_max, y_max)
                  cnt+=1
             image = Image.open(path)
             #cropped = image.crop((x_min, y_min+20, x_max, y_max))
             count += 1
             saving_path = folder_classified+ name +"_"+ str(count) + ".jpg"
             file1.write(name+"\n")
             save_file = open(saving_path, 'w')
             image.save(saving_path)
             save_file.close()
file1.close()
