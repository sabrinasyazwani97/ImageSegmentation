# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 16:34:43 2020

DO NOT MODIFY ANY CODES IN THIS FILE EXCEPT THE DEFAULT PARAMETERS
OTHERWISE YOUR RESULTS MAY BE INCORRECTLY EVALUATED! 

@author: LOH

For questions or bug reporting, please send an email to yploh@mmu.edu.my
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import importlib
import sys, getopt
from prettytable import PrettyTable
import imageSegment as seg
from os import listdir
from os.path import isfile, join, splitext

# Default parameters (the only code you can change)
verbose = False
#input_dir = 'Documents/1171302745_Assignment/dataset/test'
#output_dir = 'Documents/1171302745_Assignment/output'
#groundtruth_dir = 'Documents/1171302745_Assignment/dataset/groundtruth'
input_dir = 'Documents/1171302745_Assignment/add_dataset/test'
output_dir = 'Documents/1171302745_Assignment/output_add'
groundtruth_dir = 'Documents/1171302745_Assignment/add_dataset/groundtruth'
#numImages = 55
numImages = 150


onlyfiles = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
files = onlyfiles[0:numImages]

## Read command linehargs
myopts, args = getopt.getopt(sys.argv[1:],"i:vh")

# Reload module
importlib.reload(seg)

################################################
# o == option    a == argument passed to the o #
################################################

# parsing command line args
for o, a in myopts:
    #print(o)
    #print(a)
    if o == '-v':
        verbose = True
    elif o == '-h':
        print("\nUsage: %s -v               for extra verbosity" % sys.argv[0])
        sys.exit()
    else:
        print(' ')

error = np.zeros(numImages,)
precision = np.zeros(numImages,)
recall = np.zeros(numImages,)
iou = np.zeros(numImages,)

# Evaluate each image and compare with ground-truth
for i,name in enumerate(files):
    input = cv2.imread(input_dir + '/' + name)
    output = seg.segmentImage(input).astype('float32')
    imgName = splitext(name)
    plt.imsave(output_dir + '/' + imgName[0] + '.png',output,cmap=cm.gray)
    gt = cv2.imread(groundtruth_dir + '/' + imgName[0] + '.png',0)
    gt = np.clip(gt,0,1).astype('float32')
    
    precision[i] = sum(sum(gt*output))/sum(sum(output))
    recall[i] = sum(sum(gt*output))/sum(sum(gt))
    error[i] = 1 - ((2*precision[i]*recall[i])/(precision[i]+recall[i]))
    iou[i] = sum(sum(gt*output))/sum(sum(np.clip(gt+output,0,1)))    

# Print performance scores        
if verbose:
    print('####  DETAILED RESULTS  ####')
    t = PrettyTable(['Image', 'Error','Precision','Recall','IoU'])#,'Splits','Merges'])
    for i in range(numImages):
        t.add_row([i+1, str(round(error[i],4)),str(round(precision[i],4)),\
                   str(round(recall[i],4)),str(round(iou[i],4))]) 
                   
    t.add_row([' ',' ',' ',' ',' '])
    t.add_row(['All', str(round(np.sum(error)/numImages,4)),str(round(np.sum(precision)/numImages,4)),\
               str(round(np.sum(recall)/numImages,4)),str(round(np.sum(iou)/numImages,4))])
    print(t)
else:
    print('Adapted Rand Error: %d%%' % (np.sum(error)/numImages*100))
    print('Precision: %d%%' % (np.sum(precision)/numImages*100))
    print('Recall: %d%%' % (np.sum(recall)/numImages*100))
    print('IoU: %d%%' % (np.sum(iou)/numImages*100))
        
        
# END OF EVALUATION CODE####################################################
