import os
import cv2
import numpy as np
from scipy.spatial import distance as dist
from matplotlib import pyplot as plt

class DoesNotMatchCodeException(Exception):
    pass

class InvertedImageException(Exception):
    pass

class InsuficientPointsError(Exception):
    pass

def read_img(path, as_gray=False):
    """
    From a path return a gray image
    """
    
    if as_gray:
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        return cv2.imread(path)
        

def display(imgs, n_cols=1, titles=None, figsize=(15, 10), from_opencv=True):
    if isinstance(imgs, dict):
        imgs = list(imgs.values())
        titles = list(imgs.keys())
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
    if titles is None:
        titles = len(imgs)*[None]
    assert len(titles) == len(imgs)
        
    n_rows = np.ceil(len(imgs)/n_cols).astype(int)
    plt.figure(figsize=figsize)
    for i, (img, title) in enumerate(zip(imgs, titles)):
        if from_opencv:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(n_cols, n_rows, i+1)
        plt.imshow(img)
        plt.title(title)
        plt.grid(False)
        plt.axis(False)

def find_bounding_rect(image):
    min_x, min_y, max_x, max_y = np.inf, np.inf, 0, 0
    for i in np.ndindex(image.shape):
        if image[i] == 0: # if is a background pixel
            if min_y>i[0]:
                min_y = i[0]
            elif max_y<i[0]:
                max_y = i[0]
            if min_x>i[1]:
                min_x = i[1]
            elif max_x<i[1]:
                max_x = i[1]
    return (min_x,min_y, max_x,max_y)

def drawKeyPts(image,keypoints, color = (0,0,255)):
    for curKey in keypoints:
        x=np.int(curKey.pt[0])
        y=np.int(curKey.pt[1])
        size = 30#np.int(curKey.size/2)
        cv2.circle(image,(x,y),size, color,thickness=-1)#, lineType=8, shift=0) 
    return image

def sort_kpts(keypoints):
    """
    copy from: 
    https://www.pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
    """
	
    xSorted = keypoints[np.argsort(keypoints[:, 0]), :]

    if len(xSorted)==2:
        return xSorted

    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    if len(xSorted)==3:

        return np.array([tl, bl, rightMost[0]], dtype="float32")

    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]

    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    return np.array([tl, tr, br, bl], dtype="float32")


def non_max_suppression(boxes, overlapThresh):
    """
    Non-Maximum suppression code.

    This is a copy from: https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes	
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")