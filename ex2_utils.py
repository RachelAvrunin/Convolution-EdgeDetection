"""
        '########:'##::::'##::::'#######::::
         ##.....::. ##::'##:::'##;;;::: ##::
         ##::::::::. ##'##::::::::::: ##::::
         ######:::::. ###:::::::::: ##::::::
         ##...:::::: ## ##::::::: ##::::::::
         ##:::::::: ##:. ##::::: ##:::::::::
         ########: ##:::. ##::'############:
        ........::..:::::..:::............::
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 304976335


def conv1D(inSignal: np.ndarray, kernel1: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param inSignal: 1-D array
    :param kernel1: 1-D array as a kernel
    :return: The convolved array
    """

    inSignalSize = inSignal.shape[0]
    kernel1Size = kernel1.shape[0]
    convSize = inSignalSize + kernel1Size - 1

    conv = np.zeros(convSize)
    for m in np.arange(inSignalSize):
        for n in np.arange(kernel1Size):
            conv[m + n] = conv[m + n] + inSignal[m] * kernel1[n]

    return conv


def conv2D(inImage: np.ndarray, kernel2: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param inImage: 2D image
    :param kernel2: A kernel
    :return: The convolved image
    """
    # assuming kernel size is odd
    if (kernel2.ndim == 1):
        width = kernel2.shape[0]
        width_pad = int((width - 1) / 2)
        height = 1
        height_pad = 0
        kernel2 = kernel2[-1::-1]
    else:
        height = kernel2.shape[0]
        height_pad = int((height - 1) / 2)  # assuming kernel size is odd
        width = kernel2.shape[1]
        width_pad = int((width - 1) / 2)  # assuming kernel size is odd
        kernel2 = kernel2[-1::-1, -1::-1]
    padded_image = cv2.copyMakeBorder(inImage, height_pad, height_pad, width_pad, width_pad, cv2.BORDER_REPLICATE)

    new_image = np.zeros(inImage.shape)
    for i in range(new_image.shape[0]):
        for j in range(new_image.shape[1]):
            sub_image = padded_image[i: i + height, j: j + width]
            new_image[i, j] = (sub_image * kernel2).sum()
    return new_image


def convDerivative(inImage: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param inImage: Grayscale iamge
    :return: (directions, magnitude,x_der,y_der)
    """
    ker = np.array([[1, 0, -1]])
    x_der = conv2D(inImage, ker)
    y_der = conv2D(inImage, ker.T)
    directions = np.arctan2(y_der, x_der)
    magnitude = np.sqrt(x_der ** 2 + y_der ** 2)

    return directions, magnitude, x_der, y_der


def blurImage1(in_image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param inImage: Input image
    :param kernelSize: Kernel size
    :return: The Blurred image
    """
    gaussian = gaussianKernel(kernel_size)
    return conv2D(in_image, gaussian)


def gaussianKernel(kernel_size: int) -> np.ndarray:
    g = np.array([1, 1])
    gaussian = np.array(g)
    for i in range(kernel_size - 2):
        gaussian = conv1D(g, gaussian)
    gaussian = np.array([gaussian])
    gaussian = gaussian.T.dot(gaussian)
    return gaussian / gaussian.sum()


def laplacianKernel() -> np.ndarray:
    laplacian = np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]])
    return laplacian


def blurImage2(in_image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param inImage: Input image
    :param kernelSize: Kernel size
    :return: The Blurred image
    """
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    gaussian = cv2.getGaussianKernel(kernel_size, sigma)
    gaussian = gaussian.dot(gaussian.T)
    return cv2.filter2D(in_image, -1, gaussian, cv2.BORDER_REPLICATE)


def edgeDetectionSobel(img: np.ndarray, thresh: float = 0.7) -> (np.ndarray, np.ndarray):
    """
    Detects edges using the Sobel method
    :param img: Input image
    :param thresh: The minimum threshold for the edge response
    :return: opencv solution, my implementation
    """
    imgX, imgY, my_magnitude = sobelEdges(img)

    mySobel = np.ones_like(imgX)
    mySobel[my_magnitude < thresh] = 0

    sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0, thresh)
    sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1, thresh)

    cv_magnitude = cv2.magnitude(sobelX, sobelY)
    openCvSobel = np.ones_like(imgX)
    openCvSobel[cv_magnitude < thresh] = 0

    return openCvSobel, mySobel


def sobelEdges(img: np.ndarray) -> (np.ndarray, np.ndarray):
    # X & Y Kernels
    kerX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    kerY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)

    imgX = conv2D(img, kerX)
    imgY = conv2D(img, kerY)

    magnitude = np.sqrt(imgX ** 2 + imgY ** 2)
    return imgX, imgY, magnitude


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> (np.ndarray):
    """
    Detecting edges using the "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """
    laplacian = laplacianKernel()
    conv_img = conv2D(img, laplacian)
    return findPattern(conv_img)


def findPattern(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges by a + - or + 0 - patterns
    :param img: Input image
    :return: Edge matrix
    """
    h, w = img.shape[:2]
    edge_image = np.zeros(img.shape)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if img[i, j] == 0:
                if (((img[i - 1, j] > 0 and img[i + 1, j] < 0)
                     or (img[i - 1, j] < 0 and img[i + 1, j] > 0))
                        or ((img[i, j - 1] > 0 and img[i, j + 1] < 0)
                            or (img[i, j - 1] < 0 and img[i, j + 1] > 0))):
                    edge_image[i, j] = 1
            elif img[i, j] > 0:
                if ((img[i - 1, j] < 0 or img[i + 1, j] < 0)
                        or (img[i, j - 1] < 0 or img[i, j + 1] < 0)):
                    edge_image[i, j] = 1
            else:  # img[i, j] < 0
                if img[i - 1, j] > 0:
                    edge_image[i - 1, j] = 1
                elif img[i + 1, j] > 0:
                    edge_image[i + 1, j] = 1
                elif img[i, j - 1] > 0:
                    edge_image[i, j - 1] = 1
                elif img[i, j + 1] > 0:
                    edge_image[i, j + 1] = 1
    return edge_image


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using the "ZeroCrossingLOG" method
    :param img: Input image
    :return: :return: Edge matrix
    """
    gaussian = gaussianKernel(15)
    laplacian = laplacianKernel()
    g_l = conv2D(gaussian, laplacian)
    conv_img = conv2D(img, g_l)
    return findPattern(conv_img)


def edgeDetectionCanny(img: np.ndarray, thrs_1: float, thrs_2: float) -> (np.ndarray, np.ndarray):
    """
    Detecting edges usint "Canny Edge" method
    :param img: Input image
    :param thrs_1: T1
    :param thrs_2: T2
    :return: opencv solution, my implementation
    """

    my_res = Canny(img,thrs_1,thrs_2)
    unnorm_img = np.uint8(img * 255)
    cv_res = cv2.Canny(unnorm_img, int(thrs_1 * 255 * 3), int(thrs_2 * 255 * 3))
    return cv_res, my_res

def Canny(img: np.ndarray, thrs_1: float, thrs_2: float) -> (np.ndarray, np.ndarray):
    """
    Detecting edges usint "Canny Edge" method
    :param img: Input image
    :param thrs_1: T1
    :param thrs_2: T2
    :return: my implementation
    """

    imgX, imgY, magnitude = sobelEdges(img)
    directions = np.arctan2(imgX, imgY)

    directions = (directions * 180) / np.pi
    magnitude = magnitude * (1 / magnitude.max())
    max_mat = nms(magnitude, directions)
    my_res = hysteresis(max_mat, thrs_1, thrs_2)
    return my_res

def nms(magnitude, directions) -> np.ndarray:
    h, w = magnitude.shape[:2]
    for x in range(1, h - 1):
        for y in range(1, w - 1):
            angle = abs(round(directions[x, y]))
            # ~0 deg
            if ((0 <= angle <= 22 or 158 <= angle <= 180)
                    and (magnitude[x - 1, y] > magnitude[x, y]
                         or magnitude[x + 1, y] > magnitude[x, y])):
                magnitude[x, y] = 0
            # ~45 deg
            elif (23 <= angle <= 67
                  and (magnitude[x - 1, y - 1] > magnitude[x, y]
                       or magnitude[x + 1, y + 1] > magnitude[x, y])):
                magnitude[x, y] = 0
            # ~90 deg
            elif (68 <= angle <= 112
                  and (magnitude[x, y - 1] > magnitude[x, y]
                       or magnitude[x, y + 1] > magnitude[x, y])):
                magnitude[x, y] = 0
            # ~135 deg
            elif (113 <= angle <= 157
                  and (magnitude[x + 1, y - 1] > magnitude[x, y]
                       or magnitude[x - 1, y + 1] > magnitude[x, y])):
                magnitude[x, y] = 0
    return magnitude


def hysteresis(magnitude, thrs_1, thrs_2):
    t1 = thrs_1 * magnitude.max()
    t2 = thrs_2 * magnitude.max()

    edges = np.zeros(magnitude.shape)
    edges[magnitude > t1] = 1
    h, w = edges.shape[:2]
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            neighbors = edges[i - 1:i + 2, j - 1:j + 2]
            if t1 > edges[i, j] > t2 and 1 in neighbors:
                edges[i, j] = 1
    return edges


def houghCircle(img: np.ndarray, min_radius: float, max_radius: float) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    :param I: Input image
    :param minRadius: Minimum circle radius
    :param maxRadius: Maximum circle radius
    :return: A list containing the detected circles,
    [(x,y,radius),(x,y,radius),...]
    """

    canny_img = Canny(img, 0.4, 0.6)
    h, w = img.shape[:2]
    r_diff = max_radius - min_radius + 1
    circle_img = np.zeros((h, w, r_diff))

    for i in range(h):
        for j in range(w):
            if canny_img[i, j] == 1:
                for r in range(min_radius, max_radius+ 1):
                    for deg in range(360):
                        y = i + r * np.sin(deg)
                        x = j + r * np.cos(deg)
                        x = int(x)
                        y = int(y)
                        if 0 <= x < h and 0 <= y < w:
                            circle_img[x, y, r-min_radius] += 1
    # clean noise
    circle_img[circle_img < 120] = 0
    for r in range(r_diff):
        scope = round(r+min_radius)
        sub_img = circle_img[:, :, r]
        circle_img[:, :, r] = localMax(sub_img, scope)

    points = []
    for i in range(h):
        for j in range(w):
            max = 0
            rad = 0
            for r in range(r_diff):
                if circle_img[i, j, r] >= max:
                    max = circle_img[i, j, r]
                    rad = r
            if circle_img[i, j, rad] > (rad * 360):
                points.append((i, j, (rad + min_radius)))

    return points


def localMax(img: np.ndarray, scope: int) -> list:
    h, w = img.shape[:2]
    for i in range(scope, h - scope):
        for j in range(scope, w - scope):
            neighbors = img[i - scope:i + scope + 1, j - scope:j + scope + 1]
            if img[i, j] < neighbors.max():
                img[i, j] = 0
    return img
