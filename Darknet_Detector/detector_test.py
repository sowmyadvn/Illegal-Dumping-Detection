# Stupid python path shit.
# Instead just add darknet.py to somewhere in your python path
# OK actually that might not be a great idea, idk, work in progress
# Use at your own risk. or don't, i don't care

import os, sys
import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

for f in files:
   if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png"):
        print (f)
        path = bytes(os.path.join(folder_raw, f).encode("utf-8"))
        r = dn.detect(net, meta, path)
        cnt = 0
        while cnt < len(r):
             print (r)
             name = r[cnt][0]
             predict = r[cnt][1]
        
             x = r[cnt][2][0]
             y = r[cnt][2][1]
             w = r[cnt][2][2]
             z = r[cnt][2][3]

             x_max = (2*x+w)/2
             x_min = (2*x-w)/2
             y_min = (2*y-z)/2
             y_max = (2*y+z)/2
             print (x_min, y_min, x_max, y_max)
             image = Image.open(path)
             cropped = image.crop((x_min, y_min+20, x_max, y_max))
             count += 1
             saving_path = folder_classified+ name +"_"+ str(count) + ".jpg"
             file1.write(name+"\n")
             save_file = open(saving_path, 'w')
             cropped.save(saving_path)
             save_file.close()
             cnt += 1
file1.close()
