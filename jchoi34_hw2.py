import os
import numpy as np
import cv2
from tkinter import filedialog
from matplotlib import pyplot as plt


# calculates and displays pixel values, window around cursor, pixel intensity, mean and std devs, etc.
# on cursor movements
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        img_copy = img_padded.copy()
        cv2.rectangle(img_copy, (x - 5, y + 5), (x + 5, y - 5), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        cv2.putText(img_copy, 'x: {0}, y: {1}'.format(x, y), (5, 280), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        color = img_padded[y, x]
        cv2.putText(img_copy, '(R: {0}, G: {1}, B: {2})'.format(color[2], color[1], color[0]), (5, 250), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        intensity = (int(color[2]) + int(color[1]) + int(color[0])) / 3
        mean, std_dev = cv2.meanStdDev(img_padded[y - 5: y + 6, x - 5: x + 6])
        cv2.putText(img_copy, '(Pixel intensity: {0})'.format(intensity), (5, 160), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(img_copy, 'Mean: (R: {0}, G: {1}, B: {2})'.format(mean[2], mean[1], mean[0]), (5, 220), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(img_copy, 'SD (R: {0}, G: {1}, B: {2})'.format(std_dev[2], std_dev[1], std_dev[0]), (5,190), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('img', img_copy)


# load image, register cursor move handler, and display RGB histograms
def part_1():
    cv2.setMouseCallback('img', on_mouse)
    fig, axis = plt.subplots(3)
    blue_hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    axis[0].set_title('Blue channel')
    axis[0].plot(blue_hist, color = ('b'))
    red_hist = cv2.calcHist([img], [2], None, [256], [0, 256])
    axis[1].set_title('Red channel')
    axis[1].plot(red_hist, color = ('r'))
    green_hist = cv2.calcHist([img], [1], None, [256], [0, 256])
    axis[2].set_title('Green channel')
    axis[2].plot(green_hist, color = ('g'))
    plt.xlim([0,256])
    plt.show()

# 1.1.4
'''
    Homogeneous areas: Where there low variances and standard deviations across all RGB channels
    (mostly the same color throughout the window).
    No edges with sharp contrast within the window and mostly similar color pixels within the window.
    E.G.: Windows where all the pixels are nearly the same color and there are no edges or sharp contrasts between
    pixel values (window on a white wall or a black screen).
    
    Inhomogeneous areas: Where there are higher variances and standard deviations across all RGB channels
    (presence of different colors in the window).
    Presence of edges in the window where there are sharp contrasts in color in some neighboring pixels.
    E.G: windows where there are different colors in it or where there is an edge with sharp contrast between its
    adjacent pixel values (window capturing parts of white computer monitor and its black screen).
'''

'''
    Pixel color values are nearly the same everywhere with the jpeg images according to the RGB histograms.
'''


# Calculates the histograms for all images in a folder.
# Using [(r/32) ∗ 64 + (g/32) ∗ 8 + b/32] we normalize the value and insert it into one of 512 bins for the
# histogram. Using numpy for the calculations whenever possible to help speed up the process.
def get_histograms():
    histograms = np.empty(512)
    bins = np.array(range(512))
    inds = np.digitize((np.array(range(581)) * 0.882), bins) - 1
    for filename in os.listdir('./images/ST2MainHall4'):
        print(filename)
        img = cv2.imread('./images/ST2MainHall4/' + filename, 1).astype(int)
        # apply formula [(r/32) ∗ 64 + (g/32) ∗ 8 + b/32]. Max index is 580
        indices = ((img[:, :, 2] << 1) + (img[:, :, 1] >> 2) + (img[:, :, 0] >> 5)).ravel()
        histogram = np.zeros(512)
        counts = np.bincount(indices)
        # map indices from 0 to 580 into the 512 bins
        for i in range(np.shape(counts)[0]):
            histogram[inds[i]] += counts[i]
        histograms = np.vstack([histograms, histogram])
    return histograms[1:]


# Calculate amount of intersection between two histograms
def histogram_intersection(h1, h2):
    min_sum, max_sum = (0, 0)
    for i in range(512):
        val1 = h1[i]
        val2 = h2[i]
        if val1 < val2:
            min_sum += val1
            max_sum += val2
        else:
            min_sum += val2
            max_sum += val1
    return min_sum / max_sum


# Calculate chi square measure between two histograms
def chi_squared(h1, h2):
    result = 0
    for i in range(512):
        val1 = h1[i]
        val2 = h2[i]
        result += np.square(val1 - val2) / (val1 + val2)
    return result


# Calculates histograms for all images, computes intersections and chi square measures
# for all image pairs, and then displays them
def part_2():
    histograms = get_histograms()
    intersections = np.empty((99, 99))
    chi_squares = np.empty((99, 99))
    for i in range(np.shape(histograms)[0]):
        for j in range(np.shape(histograms)[0]):
            print('i:', i, ' j:', j)
            intersections[i][j] = histogram_intersection(histograms[i], histograms[j])
            chi_squares[i][j] = chi_squared(histograms[i], histograms[j])

    intersections *= 255 / np.max(intersections)
    chi_squares *= 255 / np.max(chi_squares)
    intersections = intersections.astype(np.uint8)
    chi_squares = chi_squares.astype(np.uint8)
    cv2.imshow('intersections', intersections)
    cv2.imshow('chi_squares', chi_squares)
    cv2.imshow('intersections scaled up', cv2.resize(intersections, (0,0), fx=4, fy=4))
    cv2.imshow('chi_squares scaled up', cv2.resize(chi_squares, (0,0), fx=4, fy=4))



img_path = filedialog.askopenfilename()
img = cv2.imread(img_path, 1)
img_padded = cv2.copyMakeBorder(cv2.imread(img_path, 1), 300, 0, 0, 300, cv2.BORDER_CONSTANT)
cv2.imshow('img', img_padded)
part_1()
part_2()
cv2.waitKey(0)
cv2.destroyAllWindows()
