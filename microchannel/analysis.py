# -*- coding: utf-8 -*-
"""
Example use of image analysis module

@author: Jamie R. Nunez
(C) 2019 - Pacific Northwest National Laboratory
"""

# Imports
import argparse
import cv2
import numpy as np
import os

import module as mod


def main(path, fmat, delim, save=True):
    
    # Create output folder if needed
    save_path = os.path.join(path, 'output/')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    f = open(os.path.join(save_path, '_results.txt'), 'w')
    f.write(delim.join(['Filename', 'Dried Area', 'Wet Area', 'Perc. Dry\n']))
    
    files = [x.replace(fmat, '') for x in os.listdir(path) if fmat in x]
    
    for fname in files:
        res = process_image(path, fname, save_path, fmat, save=save)
        f.write('%s%s' % (res, '\n'))
    f.close()


def process_image(path, fname, save_path, fmat, save=True):

    # Open image
    print('\n')
    print(fname)
    og_img = cv2.imread(os.path.join(path, fname + fmat))
    img = cv2.cvtColor(np.copy(og_img), cv2.COLOR_BGR2GRAY)
    
    # Set threshold
    val = mod.find_thresh(img)
    
    # Set manually for some images
    if 'chitin_drying experiments_10182018' in path:
        if fname in ['B10_1_P_ch1_t25h']:
            val = 32
        elif fname in ['A10_4_S_ch2_t50h']:
            val = 37
        elif fname in ['A10_4_S_ch1_t15h', 'A10_4_S_ch1_t40h', 'B10_1_P_ch2_t20h',
                       'B10_1_S_ch3_t10h', 'B10_1_S_ch2_t10h']:
            val = 38
        elif fname in ['B10_1_P_ch3_t20h']:
            val = 39
        elif fname in ['A10_4_S_ch1_t50h']:
            val = 43
        elif fname in ['C10_4_S_ch1_t5']:
            val = 50
        elif fname in ['A10_3_P_ch1_t10h']:
            val = 36
        elif fname in ['A10_1_S_ch3_t5', 'A10_1_S_ch1_t5', 'A10_2_S_ch1_t5',
                       'A10_4_S_ch3_t5', 'A10_2_S_ch2_t5', 'A10_1_S_ch2_t5',
                       'A10_4_P_ch3_t5']:
            val = 63
    else:
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
        
        print(val)
        img = mod.threshold_img(img, val)
        
        # Remove small areas of noise
        kernel = np.ones((5, 5), np.uint8)
        eroded_img = cv2.erode(img, kernel, iterations=2)
        eroded_img = cv2.dilate(eroded_img, kernel, iterations=2)
        
        # Find lines
        lines = mod.find_lines(eroded_img)
        
    if lines is None:
        print('Second line find attempt')
        
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
        print('Third line find attempt')
        lines = mod.find_lines(eroded_img, start=1, stop=75, step=5)

    # Exit if lines not found
    if lines is None:
        print('Failed')
        return fname
        
    # Remove empty space outside bounding box
    img = mod.get_channel(og_img, lines)

    # Split wet and dry regions
    res = fname + '\t'
    res = res + '\t'.join([str(x) for x in mod.stats(img, val)])

    # Show images and save
    mod.plot_all_results(save_path, fname, og_img,
                         lines, img, val)
        
    # Report stats
    return res


if __name__ == '__main__':
    
    # Parse input
#    parser = argparse.ArgumentParser(description='Property calculation using cxcalc')
#    parser.add_argument('path', type=str, help='path to folder root')
#    parser.add_argument('-f','--format', help='Format of images',
#                        required=False, default='.tif')
#    parser.add_argument('-d', '--delim', required=False,
#                        help='Delimiter for results file', default='\t')
#    parser.add_argument('-s', action='store_false', default=True,
#                    dest='save', help='Don\'t save images')
#    parser.add_argument('-a', action='store_true', default=False,
#                    dest='handdrawn', help='Use handdrawn lines')
#    args = parser.parse_args()
#
#    # Run
#    main(args.path, args.format, args.delim, save=args.save,
#         handdrawn=args.handdrawn)
    
    path = r'D:/Data/SoilSFA/NAG_drying experiments_10112018/devices without water/'
    main(path, '.tif', '\t', True)