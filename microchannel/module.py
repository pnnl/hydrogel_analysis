# -*- coding: utf-8 -*-
"""
Image analysis module to find microchannel in images.

Assumes microchannel has parallel sides and the tilt of the image is less than 60 deg.

@author: Jamie R. Nunez
(C) 2019 - Pacific Northwest National Laboratory
"""

import cv2
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from skimage import filters

resolution = 607  # pixel/mm

# Register cividis with matplotlib
rgb_cividis = np.loadtxt('cividis.txt').T
cmap = colors.ListedColormap(rgb_cividis.T, name='cividis')
cm.register_cmap(name='cividis', cmap=cmap)


#%% Image analysis and processing functions

def find_thresh(img):
    return filters.threshold_otsu(img)


def threshold_img(img, val):
    img[img < val] = 0
    img[img > 0] = 1
    return img


def _get_points(edges, x):
    top_y = []
    bot_y = []
    for col in x:
        points = np.where(edges[:, col] > 0)
        if len(points[0]) > 0:
            top_y.append(points[0][0])
            bot_y.append(points[0][-1])
        else:
            top_y.append(np.nan)
            bot_y.append(np.nan)
    return np.array(top_y), np.array(bot_y)


def get_points(img, left=False, right=False, start=None, stop=None,
               step=None):
    if start is None:
        start = 20
    if stop is None:
        stop = 200
    if step is None:
        step = 10

    edges = cv2.Canny(np.copy(img), 0, 1)
    w = img.shape[1]
    x = list(np.arange(start, stop, step))
    for temp in list(reversed(x)):
        x.append(w - temp)
    if left:
        x = x[:len(x) / 2]
    elif right:
        x = x[len(x) / 2:]
    top_y, bot_y = _get_points(edges, x)
    return np.array(x), np.array(top_y), np.array(bot_y)


def _find_lines(img, left=False, right=False, start=None, stop=None,
                step=None):

    x, top_y, bot_y = get_points(img, left=left, right=right,
                                 start=start, stop=stop,
                                 step=step)
    ind = np.array(np.isfinite(top_y), ndmin=1)

    if len(x[ind]) == 0:
        return None, 5000

    (m1, b1), r1, _, _, _ = np.polyfit(x[ind], top_y[ind], 1, full=True)
    (m2, b2), r2, _, _, _ = np.polyfit(x[ind], bot_y[ind], 1, full=True)

    # Hack to make this fail if lines are too close to be correct
    if (b2 - b1) < 400 or (b2 - b1) > 625 or abs(m1 - m2) > 0.02:
        r1 = 5000

    if r1 > r2:
        m1 = m2
    else:
        m2 = m1
    print(r1 + r2)
    return [[m1, b1], [m2, b2]], r1 + r2


def find_lines(img, start=None, stop=None, step=None):
    img = np.copy(img)
    lines, r = _find_lines(img, start=start, stop=stop,
                               step=step)
    if r < 2000:
        return lines

    else:
        lines, r = _find_lines(img, right=True, start=start, stop=stop,
                               step=step)
        if r < 2000:
            return lines

        else:
            lines, r = _find_lines(img, left=True, start=start, stop=stop,
                               step=step)
            if r < 2000:
                return lines

    return None


def remove_pixels(img, line, above=True):
    m, b = line
    x = np.arange(img.shape[1])
    y = m * x + b
    for i in range(len(x)):
        if above:
            img[:y[i], x[i]] = np.nan
        else:
            img[y[i]:, x[i]] = np.nan
    return img


# Extract the actual channel from the image
def get_channel(og_img, lines):
    img = cv2.cvtColor(np.copy(og_img), cv2.COLOR_BGR2GRAY)
    img = np.array(img, dtype=np.float)
    img = remove_pixels(img, lines[0])
    img = remove_pixels(img, lines[1], above=False)
    return img


def draw_lines(img, lines):
    img = np.copy(img)
    w = img.shape[1]
    for line in lines:
        m, b = line
        x1 = 0; y1 = int(b); x2 = w; y2 = int(m * w + b)
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 20)
    return img


def get_cartesian_points(line, w):
    rho, theta = line
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + w * (-b))
    y1 = int(y0 + w * (a))
    x2 = int(x0 - w * (-b))
    y2 = int(y0 - w * (a))
    return x1, y1, x2, y2


def get_equation(line, w):
    rho, theta = line
    points = get_cartesian_points(line, w)
    x1, y1, x2, y2 = [float(x) for x in points]
    m = (y2 - y1) / (x2 - x1)
    b = y1 - x1 * m
    return m, b


def hough(edges):
    lines = cv2.HoughLines(edges, 7, np.pi / 180, 400)
    if lines is None:
        return None

    lines = [x for x in lines if x[0][1] > 1 and x[0][1] < 2]

    if len(lines) < 2:
        return None

    # Pick two lines
    temp1 = [x for x in lines[1:] if abs(x[0][0] - lines[0][0][0]) > 450]
    if len(temp1) == 0:
        temp1 = [x for x in lines[1:] if abs(x[0][0] - lines[0][0][0]) > 400]
    temp2 = [x for x in temp1 if abs(x[0][1] - lines[0][0][1]) < 0.01]
    if len(temp2) == 0:
        temp2 = [x for x in temp1 if abs(x[0][1] - lines[0][0][1]) < 0.02]
    lines = [lines[0]]
    lines.extend(temp2)

    if len(lines) < 2:
        return None

    # Convert polar function to cartesian
    m1, b1 = get_equation(lines[0][0], edges.shape[1])
    m2, b2 = get_equation(lines[1][0], edges.shape[1])

    # Ensure first line is the bottom line
    if b1 > b2:
        temp1, temp2 = m1, b1
        m1, b1 = m2, b2
        m2, b2 = temp1, temp2
    lines = [[m1, b1], [m2, b2]]

    return lines


def find_lines_handdrawn(img):
    edges = cv2.Canny(np.copy(img), 0, 1)
    return hough(edges)


#%% Reporting functions

def stats(img, val):
    img = img[~np.isnan(img)]
    dried_pixels = np.sum(img >= val)
    wet_pixels = np.sum(img < val)
    dried_area = float(dried_pixels) * (1. / (resolution ** 2))
    wet_area = float(wet_pixels) * (1. / (resolution ** 2))
    if dried_area + wet_area > 0:
        perc = dried_area / (dried_area + wet_area) * 100
    else:
        perc = 'NA'
    return dried_area, wet_area, perc


def print_stats(img, val):
    dried_area, wet_area, perc = stats(img, val)
    print('%.2f mm2, %.2f mm2, %i perc' % (dried_area, wet_area, perc))


#%% Plotting functions

def show(img, name=None, vmax=None, vmin=None, cmap='cividis', axis='off', origin='upper', norm=None):
    plt.imshow(img, cmap=cmap, vmax=vmax, vmin=vmin, interpolation='none', origin=origin, norm=norm)
    plt.axis(axis)
    if name is not None:
        plt.savefig(name, dpi=500, bbox_inches='tight')
        plt.pause(0.005)
        plt.ion()
        plt.show()
        plt.close()


def show_wet_dry(img, val, name=None):    
    temp_img = np.copy(img)
    img[np.isnan(img)] = -1  # Avoids runtime warnings
    ind = np.where(np.logical_and(img >= 0, img < val))
    temp_img[ind] = 1
    ind = np.where(img >= val)
    temp_img[ind] = 2
    show(temp_img, name=name)


def plot_all_results(save_path, fname, og_img, lines, img, val):
    if save_path is not None and fname is not None:
        save_name = '%s%s' % (save_path, fname)
    else:
        save_name = None
    
    plt.figure(figsize=(5, 5))
    plt.subplots_adjust(hspace=0.1)
    
    plt.subplot(4, 1, 1)
    show(og_img)
    
    plt.subplot(4, 1, 2)
    show(draw_lines(og_img, lines))

    plt.subplot(4, 1, 3)
    show(img)
    
    plt.subplot(4, 1, 4)
    show_wet_dry(img, val)
    
    if save_name is not None:
        plt.savefig(save_name, dpi=500, bbox_inches='tight')
        plt.pause(0.005)
        plt.ion()
        plt.show()
        plt.close()
    else:
        plt.show()
