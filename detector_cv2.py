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

#Load config files
dn.set_gpu(0)
net = dn.load_net(b"cfg/yolo-obj.cfg", b"yolo-obj_40000.weights", 0)
meta = dn.load_meta(b"cfg/obj.data")

#Raw folder path
folder_raw = "/Users/sowmiteja/Desktop/Raw_Images"
files = os.listdir(folder_raw)

#Classified folder path
folder_classified = "/Users/sowmiteja/Desktop/Classified_Images/"
count = 0
file1 = open(os.path.join(folder_classified,datetime.now().strftime("%Y%m%d-%H%M%S")+".txt"),"w")

labels_list = ['cart','electronics','furniture','mattress','sofa','trash','trash_bags']
font = cv2.FONT_HERSHEY_SIMPLEX
'''
chart_colors = ['#3366CC', # 0.  light blue
                '#DC3912', # 1.  orange
                '#FF9900', # 2.  yellow
                '#109618', # 3.  bright green
                '#990099', # 4.  bright purple
                '#3B3EAC', # 5.  mid blue
                '#0099C6', # 6.  turquoise
               ]
'''
chart_colors = [(204,102,51),(18,557,220),(0,153,255),(24,150,16),(175,175,246),(172,62,59),(198,153,0)]
#Perform detection for every image in the files list
for f in files:
   if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png"):
        print (f)
        image_cv2 = cv2.imread(os.path.join(folder_raw,f),cv2.IMREAD_COLOR)
        image_path = bytes(os.path.join(folder_raw, f).encode("utf-8"))
        r = dn.detect(net, meta, image_path)
        print (r)
        cnt = 0
        if r is not None:
             while cnt < len(r):
                  name = r[cnt][0]
                  if name in labels_list:
                        i = labels_list.index(name)
                  predict = r[cnt][1]
                  print (name+":"+str(predict))
                  x = r[cnt][2][0]
                  y = r[cnt][2][1]
                  w = r[cnt][2][2]
                  z = r[cnt][2][3]
                  #print (x, y, w, z)

                  x_max = int(round((2*x+w)/2))
                  x_min = int(round((2*x-w)/2))
                  y_min = int(round((2*y-z)/2))
                  y_max = int(round((2*y+z)/2))
                  print (x_min, y_min, x_max, y_max)
                  pixel_list = [ x_min, y_min, x_max, y_max]
                  neg_index = [pixel_list.index(val) for val in pixel_list if val < 0]
                  cv2.rectangle(image_cv2,(x_min,y_min),(x_max,y_max),(chart_colors[i]), 2)
                  if neg_index == []:
                          cv2.rectangle(image_cv2,(x_min,y_min-24), (x_min+10*len(name),y_min),chart_colors[i],-1)
                          cv2.putText(image_cv2,name,(x_min,y_min-12), font, 0.5,(0,0,0),1,cv2.LINE_AA)
                  else:
                          if (y_min < 0 and x_min > 0):
                                  cv2.rectangle(image_cv2,(x_min,0), (x_min+10*len(name),24),chart_colors[i],-1)
                                  cv2.putText(image_cv2,name,(x_min,12), font, 0.5,(0,0,0),1,cv2.LINE_AA)
                          elif (x_min < 0 and y_min > 0):
                                  cv2.rectangle(image_cv2,(0,y_min-24), (10*len(name),y_min),chart_colors[i],-1)
                                  cv2.putText(image_cv2,name,(0,y_min-12), font, 0.5,(0,0,0),1,cv2.LINE_AA)
                          elif (x_min < 0 and y_min < 0):
                                  cv2.rectangle(image_cv2,(0,0), (10*len(name),24),chart_colors[i],-1)
                                  cv2.putText(image_cv2,name,(0,12), font, 0.5,(0,0,0),1,cv2.LINE_AA)
                  #cv2.imshow('image',image_cv2)
                  #cropped = image.crop((x_min, y_min+20, x_max, y_max))
                  cnt+=1
             
             count += 1
             saving_path = folder_classified+ name +"_"+ str(count) + ".jpg"
             file1.write(name+"\n")
             cv2.imwrite(saving_path,image_cv2)
             cv2.destroyAllWindows()
file1.close()
