from ex2_utils import *

import numpy as np
import matplotlib.pyplot as plt


def main():
    #print("ID:", myID())
    filename1 = 'baby.jpg'
    filename2 = 'dog.jpg'

    img1 = cv2.imread(filename1, 0)
    img2 = cv2.imread(filename2, 0)

    plt.subplot(1, 2, 1), plt.imshow(img1, cmap='gray'), plt.title('original pucture 1 ')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(img2, cmap='gray'), plt.title('original pucture 2')
    plt.xticks([]), plt.yticks([])
    plt.show()
    img1 = img1 / 255
    img2 = img2 / 255

    # Convolution
    # 1D
    signal = np.array([1, 1, 1, 1])
    kernel1 = np.array([1, 1.5, 1])
    print(conv1D(signal, kernel1))
    print(np.convolve(signal, kernel1, 'full'))

    # ker = np.array([[1,1,1,1,1],
    #                [1,2,2,2,1],
    #                [1,2,3,2,1],
    #                [1,2,2,2,1],
    #                [1,1,1,1,1]])

    # 2D
    size = 11
    ker = np.full((size, size), 1)
    ker = ker / ker.sum()
    conv_res1 = conv2D(img1, ker)
    ker = ker[-1::-1, -1::-1]
    conv_res2 = cv2.filter2D(img1, -1, ker, cv2.BORDER_REPLICATE)
    plt.subplot(1, 2, 1), plt.imshow(conv_res1, cmap='gray'), plt.title('my convolution')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(conv_res2, cmap='gray'), plt.title('opencv convolution')
    plt.xticks([]), plt.yticks([])
    plt.show()

    conv_res1 = conv2D(img2, ker)
    ker = ker[-1::-1, -1::-1]
    conv_res2 = cv2.filter2D(img2, -1, ker, cv2.BORDER_REPLICATE)
    plt.subplot(1, 2, 1), plt.imshow(conv_res1, cmap='gray'), plt.title('my convolution')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(conv_res2, cmap='gray'), plt.title('opencv convolution')
    plt.xticks([]), plt.yticks([])
    plt.show()

    
    # Derivatives
    directions, magnitude, x_der, y_der = convDerivative(img1)
    plt.subplot(2, 2, 1), plt.imshow(x_der, cmap='gray')
    plt.title('x derivative'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(y_der, cmap='gray')
    plt.title('y derivative'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(directions, cmap='gray')
    plt.title('directions'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(magnitude, cmap='gray')
    plt.title('magnitude'), plt.xticks([]), plt.yticks([])
    plt.show()

    directions, magnitude, x_der, y_der = convDerivative(img2)
    plt.subplot(2, 2, 1), plt.imshow(x_der, cmap='gray')
    plt.title('x derivative'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(y_der, cmap='gray')
    plt.title('y derivative'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(directions, cmap='gray')
    plt.title('directions'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(magnitude, cmap='gray')
    plt.title('magnitude'), plt.xticks([]), plt.yticks([])
    plt.show()

    # Blur
    blur_res1 = blurImage1(img1, 11)
    blur_res2 = blurImage2(img1, 11)
    plt.subplot(1, 2, 1), plt.imshow(blur_res1, cmap='gray'), plt.title('my blur')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(blur_res2, cmap='gray'), plt.title('opencv blur')
    plt.xticks([]), plt.yticks([])
    plt.show()

    blur_res1 = blurImage1(img2, 11)
    blur_res2 = blurImage2(img2, 11)
    plt.subplot(1, 2, 1), plt.imshow(blur_res1, cmap='gray'), plt.title('my blur')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(blur_res2, cmap='gray'), plt.title('opencv blur')
    plt.xticks([]), plt.yticks([])
    plt.show()

    # Edge detection
    # sobel
    sobel_res1, sobel_res2 = edgeDetectionSobel(img1)
    plt.subplot(1, 2, 1), plt.imshow(sobel_res1, cmap='gray'), plt.title('opencv Sobel')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(sobel_res2, cmap='gray'), plt.title('my Sobel')
    plt.xticks([]), plt.yticks([])
    plt.show()

    sobel_res1, sobel_res2 = edgeDetectionSobel(img2)
    plt.subplot(1, 2, 1), plt.imshow(sobel_res1, cmap='gray'), plt.title('opencv Sobel')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(sobel_res2, cmap='gray'), plt.title('my Sobel')
    plt.xticks([]), plt.yticks([])
    plt.show()

    # ZeroCrossing
    zeroCrossing_res1 = edgeDetectionZeroCrossingSimple(img1)
    zeroCrossing_res2 = edgeDetectionZeroCrossingLOG(img1)
    plt.subplot(1, 2, 1), plt.imshow(zeroCrossing_res1, cmap='gray'), plt.title('zero crossing simple')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(zeroCrossing_res2, cmap='gray'), plt.title('zero crossing LOG')
    plt.xticks([]), plt.yticks([])
    plt.show()

    zeroCrossing_res1 = edgeDetectionZeroCrossingSimple(img2)
    zeroCrossing_res2 = edgeDetectionZeroCrossingLOG(img2)
    plt.subplot(1, 2, 1), plt.imshow(zeroCrossing_res1, cmap='gray'), plt.title('zero crossing simple')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(zeroCrossing_res2, cmap='gray'), plt.title('zero crossing LOG')
    plt.xticks([]), plt.yticks([])
    plt.show()

    # Canny
    thrs_1 = 0.2
    thrs_2 = 0.1
    canny_res1, canny_res2 = edgeDetectionCanny(img1, thrs_1, thrs_2)
    plt.subplot(1, 2, 1), plt.imshow(canny_res1, cmap='gray'), plt.title('opencv cany')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(canny_res2, cmap='gray'), plt.title('my cany')
    plt.xticks([]), plt.yticks([])
    plt.show()

    canny_res1, canny_res2 = edgeDetectionCanny(img2, thrs_1, thrs_2)
    plt.subplot(1, 2, 1), plt.imshow(canny_res1, cmap='gray'), plt.title('opencv cany')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(canny_res2, cmap='gray'), plt.title('my cany')
    plt.xticks([]), plt.yticks([])
    plt.show()

    # HoughCircle
    # create pic
    circle_img = np.zeros((400, 400))
    circle_img = cv2.circle(circle_img, (50, 50), 20, (1, 1, 1), 0)
    circle_img = cv2.circle(circle_img, (250, 250), 200, (1, 1, 1), 0)
    circle_img = cv2.circle(circle_img, (150, 50), 22, (1, 1, 1), 0)
    circle_img = cv2.circle(circle_img, (25, 25), 21, (1, 1, 1), 0)

    min_radius = 20
    max_radius = 30
    hough_res = houghCircle(circle_img, min_radius, max_radius)
    print(hough_res)


if __name__ == '__main__':
    main()
