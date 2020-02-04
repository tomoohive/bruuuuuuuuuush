import numpy as np
import cv2
from .CannyEdgeDetector import *

def applyMorphologicalTransform(binalize_image):
    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(binalize_image, kernel, iterations = 1)
    opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite('result/morpho.png', closing)
    return closing

def applyBinalizeFilterFromImagePath(image_path):
    image = cv2.imread(image_path)
    binalize_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (thresh, result_image) = cv2.threshold(binalize_image, 0, 255, cv2.THRESH_BINARY)
    return result_image

def applyCannyEdgeDetectionFromImagePath(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    detector = CannyEdgeDetector(image, sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.19, weak_pixel=100)
    imgs_final = detector.detect()
    image = np.array(imgs_final)
    cv2.imwrite('result/CannyEdge.png', image)
    return 'result/CannyEdge.png'

def applyCannyEdgeDetectionCVFromImagePath(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.Canny(image, 100, 200)
    cv2.imwrite('result/CannyEdgeCV.png', image)
    return image

def applyGaussianFilterFromImagePath(image_path, kernel):
    blur = cv2.GaussianBlur(cv2.imread(image_path),(kernel,kernel),0)
    cv2.imwrite('result/gaussian.png', blur)
    return blur

def applyBilateralFilterFromImagePath(image_path):
    blur = cv2.bilateralFilter(cv2.imread(image_path), 9, 75, 75)
    cv2.imwrite('result/bilateral.png', blur)
    return blur