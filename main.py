import argparse
import math
import cv2
import numpy as np
from utils import (read_img,
                   InvertedImageException,
                   DoesNotMatchCodeException,
                   InsuficientPointsError,
                   sort_kpts)


def preprocess(image):
    """
    Apply a pre-processing in the image. The flowing operations are applied:

    - image to gray
    - image normalization: histogram equalization
    - segmentation: adaptative threshold
    - remove noisy with closing mophological operation
    - find the contour of the ROI in the image
    - crop this ROI of the image

    """

    cv2.resize(image, None, fx=0.5, fy=0.5)
    original_color = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 65, 4)

    kernel = np.array([[0,0,1,0,0],
                       [0,1,1,1,0],
                       [1,1,1,1,1],
                       [0,1,1,1,0],
                       [0,0,1,0,0]], dtype=np.uint8)
    
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    ## Get largest contour from contours
    contours, hierarchy = cv2.findContours(255-opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    ## Get minimum area rectangle and corner points
    rect = cv2.minAreaRect(max(contours, key = cv2.contourArea))
    
    # rotate img
    angle = rect[2]
    if angle != 90.0:
        rows,cols = image.shape[0], image.shape[1]
        rot_matrix = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        image = cv2.warpAffine(original_color,rot_matrix,(cols,rows))
        opening = cv2.warpAffine(opening,rot_matrix,(cols,rows))

        # rotate bounding box
        rect0 = (rect[0], rect[1], 0.0)
        box = cv2.boxPoints(rect0)
        pts = np.int0(box)
        pts[pts < 0] = 0
    else:

        pts = cv2.boxPoints(rect)
        pts = np.int0(pts)
        image = original_color

    # crop
    image = image[pts[:,1].min():pts[:,1].max(), 
                  pts[:,0].min():pts[:,0].max()]
    opening = opening[pts[:,1].min():pts[:,1].max(), 
                      pts[:,0].min():pts[:,0].max()]

    return image, opening

def detect_by_blob(image):
    """
    process the image to find the reference circle in the tests

    Basic method: Here are used a simple blob detector to detect the circles in the image

    Return:
        keypoints do opencv.
    """

    croped, preprocessed = preprocess(image)

    # estimate sizes
    rx_max = 30/1403.0
    rx_min = 10/1403.0
    h,w = preprocessed.shape

    params = cv2.SimpleBlobDetector_Params()
    
    params.filterByArea = True
    params.minArea = int(np.pi*((rx_min*w/2.0)**2)) # área do círculo inscrito em um quadrado
    params.maxArea = int(np.pi*((rx_max*w/2.0)**2))

    params.filterByColor = True
    params.blobColor=255

    params.filterByCircularity = True
    params.minCircularity = 0.01

    detector = cv2.SimpleBlobDetector_create(params)
            
    keypoints = np.array(detector.detect(preprocessed))

    # drop keypoints centered in the image. All elements in the ellipse are dropped
    y_c, x_c = (np.array(preprocessed.shape)/2).astype(int) # center point

    ellipse_eq = lambda x,y: (((x-x_c)**2)/(x_c-1)**2)+(((y-y_c)**2)/(y_c-1)**2)
    ellipse_coef = np.array([ellipse_eq(k.pt[0], k.pt[1]) for k in keypoints])

    keypoints = keypoints[ellipse_coef>=1]

    # check the keypoints sanity, that is, if positions makes sense
    if len(keypoints)<2:
        raise InsuficientPointsError('insuficient number of keypoints, image probably is bad')

    # pass only if the points make a angle near 45º
    elif len(keypoints)==2:
        p1 = keypoints[0].pt
        p2 = keypoints[1].pt

        hip = np.sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)
        cos = abs(p2[0]-p1[0])/hip

        if not math.isclose(cos, (math.sqrt(2)/2), rel_tol=0.12, abs_tol=0.1):
            raise InsuficientPointsError('insuficient number of keypoints to continue the process')

    elif len(keypoints)>4:
        x_p = np.array([kp.pt[0] for kp in keypoints], dtype=float)
        y_p = np.array([kp.pt[1] for kp in keypoints], dtype=float)
        keypoints = np.array([ cv2.KeyPoint(x = x_p.min(), y =  y_p.min(), size=50),
                               cv2.KeyPoint(x = x_p.min(), y =  y_p.max(), size=50),
                               cv2.KeyPoint(x = x_p.max(), y =  y_p.min(), size=50),
                               cv2.KeyPoint(x = x_p.max(), y =  y_p.max(), size=50),
                             ])

    return keypoints, croped, preprocessed

def detect_by_label(image):
    """
    process the image to find the reference circle in the tests

    Basic method: Here are used a simple label the image and filter the circles in the image

    Return:
        keypoints do opencv.
    """

    croped, preprocessed = preprocess(image)

    # estimate sizes
    rx_max = 30/1403.0
    rx_min = 10/1403.0
    ry_max = 25/992.0
    ry_min = 10/992.0
    h,w = preprocessed.shape

    output = cv2.connectedComponentsWithStats(preprocessed, 8, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output

    # drop keypoints centered in the image. All elements in the ellipse are dropped
    y_c, x_c = (np.array(preprocessed.shape)/2).astype(int)

    ellipse_eq = lambda x,y: (((x-x_c)**2)/(x_c-1)**2)+(((y-y_c)**2)/(y_c-1)**2)
    ellipse_coef = np.array([ellipse_eq(k[0], k[1]) for k in centroids])

    centroids = centroids[ellipse_coef>=1.2]
    stats = stats[ellipse_coef>=1.2]


    # filter by area
    max_area = int(np.pi*((rx_max*w/2.0)**2))
    min_area = int(np.pi*((rx_min*w/2.0)**2))
    area = stats[:, cv2.CC_STAT_AREA]

    area_filter = np.bitwise_and(area>min_area, area<max_area)
    centroids = centroids[area_filter]
    stats = stats[area_filter]

    #filter by size
    w_obj = stats[:, cv2.CC_STAT_WIDTH]
    h_obj = stats[:, cv2.CC_STAT_HEIGHT]

    max_w = rx_max*w
    min_w = rx_min*w
    max_h = ry_max*w
    min_h = ry_min*w

    w_filter = np.bitwise_and(w_obj>min_w,w_obj<max_w)
    h_filter = np.bitwise_and(h_obj>min_h,h_obj<max_h)
    centroids = centroids[np.bitwise_and(w_filter, h_filter)]
    stats = stats[np.bitwise_and(w_filter, h_filter)]

    #filter ratio between circle area and rect
    w_obj = stats[:, cv2.CC_STAT_WIDTH]
    h_obj = stats[:, cv2.CC_STAT_HEIGHT]
    area = stats[:, cv2.CC_STAT_AREA]

    #in our case the ratio is equivallent to np.pi/4, that comes from te formula of the circle and rect
    r = stats[:,4]/(w_obj*h_obj)
    r_filter = np.absolute(r-np.pi/4)<0.12
    centroids = centroids[r_filter]

    # check the keypoints sanity, that is, if positions makes sense
    if len(centroids)<2:
        raise InsuficientPointsError('insuficient number of keypoints, image probably is bad')

    # pass only if the points make a angle near 45º
    elif len(centroids)==2:
        p1 = centroids[0]
        p2 = centroids[1]

        hip = np.sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)
        cos = abs(p2[0]-p1[0])/hip

        if not math.isclose(cos, (math.sqrt(2)/2), rel_tol=0.12, abs_tol=0.1):
            raise InsuficientPointsError('insuficient number of keypoints to continue the process')

    elif len(centroids)>4:
        x_p = np.array([kp[0] for kp in centroids], dtype=float)
        y_p = np.array([kp[1] for kp in centroids], dtype=float)
        centroids = np.array([ cv2.KeyPoint(x = x_p.min(), y =  y_p.min(), size=50),
                               cv2.KeyPoint(x = x_p.min(), y =  y_p.max(), size=50),
                               cv2.KeyPoint(x = x_p.max(), y =  y_p.min(), size=50),
                               cv2.KeyPoint(x = x_p.max(), y =  y_p.max(), size=50),
                             ])

    if not isinstance(centroids[0], cv2.KeyPoint):
        centroids = np.array([cv2.KeyPoint(x = k[0], y = k[1], size=50) for k in centroids])

    return centroids, croped, preprocessed


def compute_rotation(keypoints, image):
    """
    Apply the rotation in the image and points and return these variables
    """

    points = np.array([k.pt for k in keypoints])
    points = points[points[:,0].argsort()]
    
    flag=0
    if isinstance(image, list):
        assert(len(image)==2)
        image, preprocessed = image
        flag=1
    
    h,w = image.shape[:2]
    centers = (w//2,h//2)

    idx = ((((points[1:,0]-points[0,0])**2)+((points[1:,1]-points[0,1])**2))**0.5).argmin()+1
    x = abs(points[idx, 1] - points[0,1])
    y = abs(points[idx, 0] - points[0,0])

    angle = math.atan2(x,y)
    angle = int(math.degrees(angle))
    angle = 0 if abs(angle-90)<10 else angle
    tx = 0
    ty = 0
    new_shape = (image.shape[1],image.shape[0])
    if centers[0]>centers[1]:
        angle = 90-angle
        tx = centers[1]-centers[0]
        ty = centers[0]-centers[1]
        new_shape = (new_shape[1], new_shape[0])

    rot_matrix = cv2.getRotationMatrix2D(centers,angle,1.0)
    rot_matrix[0,2]+= tx
    rot_matrix[1,2]+= ty

    image = cv2.warpAffine(image,rot_matrix, new_shape)

    if flag:
        preprocessed = cv2.warpAffine(preprocessed,rot_matrix, new_shape)
        return image, preprocessed, points

    points = cv2.transform(np.array([points]), rot_matrix).squeeze()

    return image, sort_kpts(points)

def find_code(keypoints, image):
    if isinstance(image, list):
        image, transformed = image

    h,w, _ = image.shape
    # top-left circle
    top_left = (int(keypoints[0,0]), int(keypoints[0,1]))
    idx = keypoints[:,0].argmax()
    top_right = (int(keypoints[idx,0]), int(keypoints[idx,1]))

    # if the true top-left point is missing
    if len(keypoints) <=3 and (top_left[0]>w/2 or top_left[1]>h/2):
        x = keypoints[:,0].min()
        y = keypoints[:,1].min()
        keypoints = np.append(keypoints,[[x, y]], axis=0)
        top_left = (int(x), int(y))
        top_right = (int(keypoints[0,0]), int(keypoints[0,1]))
        keypoints = sort_kpts(keypoints)

    # if the true top-right point is missing
    if len(keypoints) <=3 and (top_right[0]<w/2 or top_right[1]>h/2):
        x = keypoints[:,0].max()
        y = keypoints[:,1].min()
        keypoints = np.append(keypoints,[[x, y]], axis=0)
        top_right = (int(x), int(y))
        keypoints = sort_kpts(keypoints)

    W = abs(top_right[0] - top_left[0])
    
    tl_code = (int(W*0.259)+top_left[0], int(W*(-0.0435))+top_left[1])
    br_code = (int(W*0.474)+top_left[0], int(W*(-0.0032))+top_left[1])
    blur = np.mean(image,axis=-1, dtype=int)
    blur = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    blur = cv2.GaussianBlur(blur, (5, 5), 0)

    _, binary = cv2.threshold(blur,
                              0,
                              255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    code = binary[tl_code[1]:br_code[1], tl_code[0]:br_code[0]]

    canny = cv2.Canny(code, 100,250)

    borders = (canny/255.0).sum()

    if borders<canny.size*0.03 or borders>canny.size*0.1:
        raise InvertedImageException("Can't find the code")

    return code
    
def binary_decoder(img):
    """
    For the given problem decode the binary code given a sliced binary image

    return a int relative to binary code read
    """
    h,w = img.shape

    upper = np.empty((12,), dtype=str)
    bottom = np.empty((12,), dtype=str)

    for i, im in enumerate(np.array_split(img, indices_or_sections=12, axis=1)):

        upper[i] = (((im[:h//2,:])/255).sum()<im[:h//2,:].size*0.5).astype(int)
        bottom[i] = (((im[h//2:,:])/255).sum()<im[h//2:,:].size*0.5).astype(int)

    test_number = int(''.join(upper),2)
    pag = int(''.join(bottom[:6]),2)
    nv = int(''.join(bottom[6:]),2)

    if not (nv==(60-((test_number-1)*4+(pag-1)))%60):
        raise DoesNotMatchCodeException('The number of verification does not have passed in the test')

    return test_number, pag, nv

def get_nusp(image, keypoints):

    h,w, _ = image.shape
    # top-left circle
    top_left = (int(keypoints[0,0]), int(keypoints[0,1]))
    top_right = (int(keypoints[1,0]), int(keypoints[1,1]))

    # if the true top-left point is missing
    if len(keypoints) <=3 and (top_left[0]>w/2 or top_left[1]>h/2):
        x = keypoints[:,0].min()
        y = keypoints[:,1].min()
        keypoints = np.append(keypoints,[[x, y]], axis=0)
        top_left = (int(x), int(y))
        top_right = (int(keypoints[0,0]), int(keypoints[0,1]))
        keypoints = sort_kpts(keypoints)

    # if the true top-right point is missing
    if len(keypoints) <=3 and (top_right[0]<w/2 or top_right[1]>h/2):
        x = keypoints[:,0].max()
        y = keypoints[:,1].min()
        keypoints = np.append(keypoints,[[x, y]], axis=0)
        top_right = (int(x), int(y))
        keypoints = sort_kpts(keypoints)

    W = abs(top_right[0] - top_left[0])
    
    confidence = 6
    tl_nusp = (int(W*0.0456)+top_left[0]-confidence, int(W*(0.111))+top_left[1]-confidence)
    br_nusp = (int(W*0.2484)+top_left[0]+confidence, int(W*(0.4049))+top_left[1]+confidence)

    croped = np.uint8(np.mean(image, axis=-1)<127)
    croped = croped[tl_nusp[1]:br_nusp[1], tl_nusp[0]:br_nusp[0]]

    nusp = np.empty((8), dtype=str)
    for i, im in enumerate(np.array_split(croped, indices_or_sections=8, axis=1)):
        ar = np.array_split(im, indices_or_sections=10, axis=0)
        summations = np.array([np.sum(p) for p in ar])
        nusp[i] = np.argmax(summations).astype(str)

    return ''.join(nusp)

def main(path, algorithm):
    image = read_img(path)

    try:
        if algorithm=='label':
            keypoints, croped, preprocessed = detect_by_label(image)
        else:
            keypoints, croped, preprocessed = detect_by_blob(image)
    except InsuficientPointsError as e:
        print(e)
        return 0

    rotated = compute_rotation(keypoints, croped)
    
    if len(rotated) == 3:
        img,preprocessed, keypoints = rotated
        img = [img,preprocessed]
    else:
        img, keypoints = rotated

    try:
        code = find_code(keypoints, img)
    except InvertedImageException:
        M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), 180, 1)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        keypoints = np.int0(cv2.transform(np.array([keypoints]), M)).squeeze()
        keypoints = sort_kpts(keypoints)
        try:
            code = find_code(keypoints, img)
        except InvertedImageException as e:
            print(e)
            return 0
    
    # binary decoder
    try:
        test_number, pag, nv = binary_decoder(code)
    except DoesNotMatchCodeException as e:
        print(e)
        return 0

    print('Successful detection:')
    print('test number:', test_number)
    print('page:', pag)

    if pag==1:
        nusp = get_nusp(img, keypoints)
        print('nusp:', nusp)

    return 1



if __name__=='__main__':

    parser = argparse.ArgumentParser(description='AMC test reader')
    parser.add_argument('path', help="path to the image")
    parser.add_argument('--algorithm', '-a',
                        help="Algorithm to choose, can be: connected or blob",
                        choices=['connected', 'blob'],
                        default='connected')
    
    args = parser.parse_args()
    main(args.path,args.algorithm)