import cv2
import numpy as np

# IMAGES_PATH = '/Users/kani/Pictures'
IMAGE_NAME = 'aaaaaaaaa.png'
IMAGE_PATH = './' + IMAGE_NAME

def main1():
    originalImage = cv2.imread(IMAGE_PATH)
    hsv = cv2.cvtColor(originalImage, cv2.COLOR_BGR2HSV)
    
    h_img, s_img, v_img = cv2.split(hsv)
    cv2.imwrite('h_img.png', h_img)
    cv2.imwrite('s_img.png', s_img)
    cv2.imwrite('v_img.png', v_img)

    s_img = cv2.bitwise_not(s_img)

    hist_s_img = cv2.equalizeHist(s_img)

    _, result_bin = cv2.threshold(s_img, 200, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (4,4))

    result_morphing = cv2.morphologyEx(result_bin, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite('outmor.png', result_morphing)

    # tmp_img, contours, _ = cv2.findContours(result_morphing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # approx = approx_contour(contours)

    # cp_org_img_for_draw

def main2():
    image = cv2.imread(IMAGE_PATH)
    image = cv2.medianBlur(image, 3)
    image = cv2.medianBlur(image, 3)
    image = cv2.medianBlur(image, 3)
    
    cv2.imwrite('outblur.png', image)

def main3():
    originalImage = cv2.imread(IMAGE_PATH)
    non = cv2.fastNlMeansDenoisingColored(originalImage, None, 10, 10, 7, 21)
    cv2.imwrite('outNon.png', non)

def main4():
    image = cv2.imread('input.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    vectorized = image.reshape((-1,3))
    vectorized = np.float32(vectorized)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    attempts=100
    ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((image.shape))
    cv2.imwrite('aaaaaaaaa.png', result_image)
"""
contour line approximation
"""    
def main5():
    blur = cv2.GaussianBlur(cv2.imread('input.png'),(15,15),0)
    blur = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    cv2.imwrite('gaussian.png', blur)

def approx_contour(contours):
    approx = []
    for i in range(len(contours)):
        cnt = contours[i]
        epsilon = 0.0001*cv2.arcLength(cnt, True)
        approx.append(cv2.approxPolyDP(cnt, epsilon, True))
    return approx

if __name__ == '__main__':
    main5()