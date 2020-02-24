import numpy as np
import cv2


def get_histograms():
    histograms = np.empty(512)
    bins = np.array(range(512))
    inds = np.digitize((np.array(range(581)) * 0.882), bins) - 1
    for filename in os.listdir('../images/ST2MainHall4'):
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


gray_img = cv2.imread('../images/ST2MainHall4/ST2MainHall4001.jpg', 0)
img = cv2.imread('../images/ST2MainHall4/ST2MainHall4001.jpg', 1)
blues = img[:, :, 0]
greens = img[:, :, 1]
reds = img[:, :, 2]
gray_edges = cv2.Canny(gray_img, 100, 200).astype(bool)
blue_edges = cv2.Canny(blues, 100, 200)
red_edges = cv2.Canny(greens, 100, 200)
green_edges = cv2.Canny(reds, 100, 200)
sobelX = np.ma.array(cv2.Sobel(gray_img,cv2.CV_64F,1,0,ksize=5), mask=gray_edges).filled(fill_value=0)
sobelY = np.ma.array(cv2.Sobel(gray_img,cv2.CV_64F,0,1,ksize=5), mask=gray_edges).filled(fill_value=0)
# sobelX = np.array([sobelX[i] if gray_edges[i] != 0 else 0 for i in range(np.shape(gray_edges)[0])])
# sobelY = np.array([sobelY[i] if gray_edges[i] != 0 else 0 for i in range(np.shape(gray_edges)[0])])
print(cv2.Sobel(gray_img,cv2.CV_64F,1,0,ksize=5))
print()
print(sobelX)
# cv2.imshow('img', sobelX)
# cv2.imshow('edges', sobelY)
# cv2.waitKey(0)
# cv2.destroyAllWindows()