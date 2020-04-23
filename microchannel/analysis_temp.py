# -*- coding: utf-8 -*-
"""
@author: nune558
"""

# Imports
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import module as mod

path = '../InputImages/new/'
fmat = '.tif'
fname = 'C_10_3_S_ch2_t5h'

#files = [x.replace(fmat, '') for x in os.listdir(path) if fmat in x]
#for fname in files:

if True:

    # Open image
    og_img = cv2.imread(os.path.join(path, fname + fmat))
    img = cv2.cvtColor(np.copy(og_img), cv2.COLOR_BGR2GRAY)
    
    # Set threshold
    #if path.replace('repA/', '') == 'D:/Data/SoilSFA/chitin_drying experiments_10182018/repA/':
    val = mod.find_thresh(img)

## Chitin
#    if fname in ['B10_1_P_ch1_t25h']:
#        val = 32
#    elif fname in ['A10_4_S_ch2_t50h']:
#        val = 37
#    elif fname in ['A10_4_S_ch1_t15h', 'A10_4_S_ch1_t40h', 'B10_1_P_ch2_t20h',
#                   'B10_1_S_ch3_t10h', 'B10_1_S_ch2_t10h']:
#        val = 38
#    elif fname in ['B10_1_P_ch3_t20h']:
#        val = 39
#    elif fname in ['A10_4_S_ch1_t50h']:
#        val = 43
#    elif fname in ['C10_4_S_ch1_t5']:
#        val = 50
#    elif fname in ['A10_3_P_ch1_t10h']:
#        val = 36
#    elif fname in ['A10_1_S_ch3_t5', 'A10_1_S_ch1_t5', 'A10_2_S_ch1_t5',
#                   'A10_4_S_ch3_t5', 'A10_2_S_ch2_t5', 'A10_1_S_ch2_t5',
#                   'A10_4_P_ch3_t5']:
#        val = 63
    
## NAG
    if fname in ['A10_1_P_ch2_t30h', 'A10_2_S_ch3_t20h']:
        val = 50
    elif fname in ['A10_2_S_ch1_t5h']:
        val = 52
    elif fname in ['A10_2_S_ch1_t20h', 'B_10_2_S_ch2_t5h']:
        val = 47
    elif fname in ['A10_4_S_ch2_t5']:
        val = 63
    elif fname in ['A10_1_P_ch1_t30h', 'A10_1_P_ch1_t40h']:
        val = 39

    lines = None
    
    # Find lines for bounding box
    hd_path = os.path.join(path, 'handdrawn', fname + fmat)
    if os.path.exists(hd_path):
        hd_img = cv2.imread(hd_path)
        hd_img = cv2.cvtColor(np.copy(hd_img), cv2.COLOR_BGR2GRAY)
        hd_img = mod.threshold_img(hd_img, 255)
        lines = mod.find_lines_handdrawn(hd_img)
    
    if lines is None:  # Handdrawn not being used. Find box with IA tools
    
        # Threshold
        print(val)
        img = mod.threshold_img(img, val)
        
        # Remove noise
        kernel = np.ones((5, 5), np.uint8)
        eroded_img = cv2.erode(img, kernel, iterations=2)
        eroded_img = cv2.dilate(eroded_img, kernel, iterations=2)
        
        # Find lines
        lines = mod.find_lines(eroded_img)
    
    if lines is None:
        print('Second')
        # Remove noise
        kernel = np.ones((20, 20), np.uint8)
        eroded_img = cv2.erode(img, kernel, iterations=2)
        eroded_img = cv2.dilate(eroded_img, kernel, iterations=2)
        
        # Find lines
        lines = mod.find_lines(eroded_img)
    
    # This part is new. Add to analysis.py when working
    if lines is None:  # Try Hough Transform
        print('Hough')
        edges = cv2.Canny(np.copy(img), 0, 1)
        lines = mod.hough(edges)
    
    if lines is None:
        print('Second line find attempt')
        lines = mod.find_lines(eroded_img, start=1, stop=75, step=5)
    
    # Exit if lines not found
    if lines is None:
        print('Fname:', fname)
    
    else:
        # Remove empty space outside bounding box
        img = mod.get_channel(og_img, lines)
        
        # Split wet and dry regions
#        res = [fname].extend('\t'.join([str(x) for x in mod.stats(img, val)]))
        print(mod.stats(img, val))
        
        # Show images
        mod.plot_all_results(path, fname, og_img, lines, img, val)
        
