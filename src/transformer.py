import numpy as np
import cv2 as cv
import random


"""
Dev Log
10/15 Current version only support grayscale
10/16 17 Transformers implemented
10/17 Changed the input/output format requirements: input & output all must be float32
10/18 Implemented a super class for better reporting the params
10/18 Added height param. But the random threshold is still only bounded by width
"""


# TODO: Hough Lines and Hough Circles could also be useful. But need to figure out a way to integrate
# TODO: Gabor Filter could be useful
# TODO: Adaptive Threshold C param


class Transformer:

    def __init__(self, width, height):
        """
        Basic element for init a transformer
        :param width:
        """
        self.width = width
        self.height = height
        self.code = -1
        self.params = []

    def mutate(self, r=0.0005):
        """
        Dummy super class method
        :param r:
        :return:
        """
        pass

    def rep(self):
        """
        Return the description of the transformer
        :return: description
        """

        rep = [self.code, self.params]
        return rep


class AdaptiveThreshold(Transformer):
    """
    Transformer AdaptiveThreshold
    """
    def __init__(self, width, height):
        """
        Constructor, initialize params randomly
        :param width: image size
        """
        Transformer.__init__(self, width, height)
        self.code = 1
        x, y, z = np.random.choice(2, 1), np.random.choice(2, 1), np.random.choice(2, 1)
        adaptive_approach = cv.ADAPTIVE_THRESH_MEAN_C if x == 1 else cv.ADAPTIVE_THRESH_GAUSSIAN_C
        thresh_mode = cv.THRESH_BINARY if y == 1 else cv.THRESH_BINARY_INV
        block_size = random.randrange(3, width + 1, 2)
        self.params = [adaptive_approach, thresh_mode, block_size]
        self.c = 0

    def mutate(self, r=0.0005):
        """
        If mutate, regenerate the parameter
        :param r: mutate rate
        :return: void
        """

        if np.random.choice(2, 1, p=[1-r, r]) == 1:
            self.params[0] = cv.ADAPTIVE_THRESH_MEAN_C if np.random.choice(2, 1) == 0 else \
                cv.ADAPTIVE_THRESH_GAUSSIAN_C
        if np.random.choice(2, 1, p=[1-r, r]) == 1:
            self.params[1] = cv.THRESH_BINARY if np.random.choice(2, 1) == 0 else cv.THRESH_BINARY_INV
        if np.random.choice(2, 1, p=[1-r, r]) == 1:
            self.params[2] = random.randrange(3, self.width + 1, 2)

    def transform(self, img: np.ndarray):
        """
        Process the img
        :param img:
        :return: masked img in float32 format
        """
        if img.dtype == np.float32:
            img_8u = (img * 255).astype(np.uint8)
        else:
            raise Exception("Input should be {} but was {}".format(np.float32, img.dtype))
        mask = cv.adaptiveThreshold(img_8u, 255, self.params[0], self.params[1], self.params[2], self.c)
        return img * (mask == 255)


class CannyEdge(Transformer):
    """
    Transformer CannyEdge
    """
    def __init__(self, width, height):
        """
        Constructor init random params
        """

        Transformer.__init__(self, width, height)
        self.code = 2
        theta = random.uniform(0.3, 0.9)
        self.params = [theta]

    def mutate(self, r=0.0005):
        """
        Mutate
        :param r: mutate rate
        :return: void
        """
        if np.random.choice(2, 1, p=[1 - r, r]) == 1:
            self.params[0] = random.uniform(0.3, 0.9)

    def transform(self, img):
        """
        Process image
        :param img: input img
        :return: Edged Image
        """

        if img.dtype == np.float32:
            img_8u = (img * 255).astype(np.uint8)
        else:
            raise Exception("Input should be {} but was {}".format(np.float32, img.dtype))

        thresh, _ = cv.threshold(img_8u, thresh=0, maxval=255, type=(cv.THRESH_BINARY + cv.THRESH_OTSU))

        mask = cv.Canny(img_8u, threshold1=int(thresh*0.1), threshold2=int(thresh*self.params[0]))
        return img * (mask == 255)


class CensusTransformation(Transformer):
    """
    Census Transformation 3x3 patch
    """

    def __init__(self, width, height):
        """
        Constructor
        :param width: image width
        """

        Transformer.__init__(self, width, height)
        self.code = 3

    def transform(self, img: np.ndarray):
        """
        Census transformation.
        Encode each pixel according to pixel's rank
        :param img: input image
        :return: census transformed image
        """

        if img.dtype != np.float32:
            raise Exception("Input should be {} but was {}".format(np.float32, img.dtype))

        w, h = self.width, self.height

        # Initialize output array
        census = np.zeros((h - 2, w - 2), dtype=np.uint8)

        # centre pixels, which are offset by (1, 1)
        cp = img[1:h - 1, 1:w - 1]

        # offsets of non-central pixels
        offsets = [(u, v) for v in range(3) for u in range(3) if not u == 1 == v]

        # Do the pixel comparisons (encode the neighborhood compare into)
        for u, v in offsets:
            census = (census << 1) | (img[v:v + h - 2, u:u + w - 2] >= cp)

        return (census/255).astype(np.float32)


class CLAHistogram(Transformer):
    """
    Contrast Limited Adaptive Histogram Equalization.
    Enhance the contrast of an image.
    """
    def __init__(self, width, height):
        """
        Constructor
        Random clip limit and tile size
        :param width: image size
        """

        Transformer.__init__(self, width, height)
        self.code = 4
        clip_limit = np.random.uniform(10, 40)
        tile_size = random.randrange(3, width + 1)
        self.params = [clip_limit, tile_size]

    def mutate(self, r=0.0005):
        """
        Mutate
        :param r: mutate rate
        :return:
        """

        if np.random.choice(2, 1, p=[1-r, r]) == 1:
            self.params[0] = np.random.uniform(10, 40)
        if np.random.choice(2, 1, p=[1-r, r]) == 1:
            self.params[1] = random.randrange(3, self.width + 1)

    def transform(self, img: np.ndarray):
        """
        Process the image
        :param img: image to be processed
        :return:
        """

        if img.dtype == np.float32:
            img_8u = (img * 255).astype(np.uint8)
        else:
            raise Exception("Input should be {} but was {}".format(np.float32, img.dtype))

        clahe = cv.createCLAHE(clipLimit=self.params[0], tileGridSize=(self.params[1], self.params[1]))
        img = clahe.apply(img_8u, None)

        return (img/255).astype(np.float32)


class HistogramEqualization(Transformer):
    """
    Histogram Equalization to enhance the contrast of the image
    """

    def __init__(self, width, height):
        Transformer.__init__(self, width, height)
        self.code = 5

    @staticmethod
    def transform(img: np.ndarray):
        if img.dtype == np.float32:
            img_8u = (img * 255).astype(np.uint8)
        else:
            raise Exception("Input should be {} but was {}".format(np.float32, img.dtype))
        img = cv.equalizeHist(img_8u, None)
        return (img/255).astype(np.float32)


class DistanceTransformation(Transformer):
    """
    Distance Transform.
    Threshold image into binary image then compute the distance of each pixel's distance to the closest null pixel
    """

    def __init__(self, width, height):
        """
        Constructor. Random distance method and kernel size.
        :param width:
        """

        Transformer.__init__(self, width, height)
        self.code = 6
        dist = cv.DIST_L1 if random.randrange(2) == 0 else cv.DIST_L2
        k_size = 3 if random.randrange(2) == 0 else 5
        self.params = [dist, k_size]

    def mutate(self, r=0.0005):
        """
        Mutate
        :param r: mutate rate
        :return: void
        """

        if np.random.choice(2, 1, p=[1-r, r]) == 1:
            self.params[0] = cv.DIST_L1 if np.random.choice(2, 1) == 0 else cv.DIST_L2
        if np.random.choice(2, 1, p=[1-r, r]) == 1:
            self.params[1] = 3 if np.random.choice(2, 1) == 0 else 5

    def transform(self, img):
        """
        Process image: auto-threshold to binary image, then apply distance transform plus normalization
        :param img: input image
        :return: distance transformed image
        """

        if img.dtype == np.float32:
            img_8u = (img * 255).astype(np.uint8)
        else:
            raise Exception("Input should be {} but was {}".format(np.float32, img.dtype))

        # algorithm decide threshold
        _, img = cv.threshold(img_8u, 0, 1.0, cv.THRESH_BINARY | cv.THRESH_OTSU)

        # the dist image will be normalized and convert to uint8

        img = cv.distanceTransform(img, self.params[0], self.params[1], dstType=cv.CV_32F)

        # normalization
        img = cv.normalize(img, None, 0, 1.0, cv.NORM_MINMAX, cv.CV_32F)
        return img


class Dilate(Transformer):
    """
    Dilate the image.
    """

    def __init__(self, width, height):
        """
        Constructor
        One random param -> iteration. More iteration -> more dilated
        :param width: image width
        """

        Transformer.__init__(self, width, height)
        self.code = 7
        num_iter = random.randrange(1, 9)  # more iterations -> more dilated
        self.params = [num_iter]

    def mutate(self, r=0.0005):
        """
        Mutate
        :param r: mutate rate
        :return: void
        """

        if np.random.choice(2, 1, p=[1 - r, r]) == 1:
            self.params[0] = random.randrange(1, 9)

    def transform(self, img):
        """
        Process image
        :param img: input image
        :return: dilated image
        """

        if img.dtype != np.float32:
            raise Exception("Input should be {} but was {}".format(np.float32, img.dtype))

        img = cv.dilate(img, kernel=(3, 3), iterations=self.params[0])  # use default kernel size the 8 neighbors
        return img


class Erode(Transformer):
    """
    Erode the image.
    One random param -> iteration. More iteration -> more eroded
    """

    def __init__(self, width, height):
        """
        Constructor
        :param width: image width
        """

        Transformer.__init__(self, width, height)
        self.code = 8
        num_iter = random.randrange(1, 9)  # more iterations -> more eroded
        self.params = [num_iter]

    def mutate(self, r=0.0005):
        """
        Mutate
        :param r: mutate rate
        :return: void
        """

        if np.random.choice(2, 1, p=[1 - r, r]) == 1:
            self.params[0] = random.randrange(1, 9)

    def transform(self, img):
        """
        Process image
        :param img: input image
        :return: dilated image
        """

        if img.dtype != np.float32:
            raise Exception("Input should be {} but was {}".format(np.float32, img.dtype))

        img = cv.erode(img, kernel=(3, 3), iterations=self.params[0])  # use default kernel size the 8 neighbors
        return img


class DifferenceGaussian(Transformer):

    def __init__(self, width, height):
        """
        Constructor.
        Two random param: sigma value and the aspect ratio between two sigmas
        :param width:
        """

        Transformer.__init__(self, width, height)
        self.code = 9

        sigma_a = random.uniform(1, 3)  # smaller sigma
        self.ratio = random.uniform(1, 5)  # ratio between smaller sigma and the larger sigma
        sigma_b = sigma_a * self.ratio
        a, b = 3 * sigma_a, 3 * sigma_b
        a, b = int(np.ceil(a) // 2 * 2 + 1), int(np.ceil(b) // 2 * 2 + 1)  # round to nearest odd
        t = self.width if self.width % 2 == 1 else self.width - 1
        a, b = min(a, t), min(b, t)  # kernel size can not be larger than the width also odd
        k_size_a = (a, a)
        k_size_b = (b, b)

        self.params = [sigma_a, k_size_a, sigma_b, k_size_b]

    def mutate(self, r=0.0005):
        """
        Mutate
        :param r: mutate rate
        :return: void
        """

        if np.random.choice(2, 1, p=[1-r, r]) == 1:
            self.params[0] = random.uniform(1, 3)
        if np.random.choice(2, 1, p=[1-r, r]) == 1:
            self.ratio = random.uniform(1, 5)
        self.params[2] = self.params[0] * self.ratio
        a, b = 3 * self.params[0], 3 * self.params[2]
        a, b = int(np.ceil(a) // 2 * 2 + 1), int(np.ceil(b) // 2 * 2 + 1)
        t = self.width if self.width % 2 == 1 else self.width - 1
        a, b = min(a, t), min(b, t)  # kernel size can not be larger than the width also odd
        self.params[1] = (a, a)
        self.params[3] = (b, b)

    def transform(self, img):
        """
        Process the image.
        Take the difference between two gaussian
        :param img: input image
        :return: difference of gaussian
        """

        if img.dtype != np.float32:
            raise Exception("Input should be {} but was {}".format(np.float32, img.dtype))
        img_a = cv.GaussianBlur(img, ksize=self.params[1], sigmaX=self.params[0])
        img_b = cv.GaussianBlur(img, ksize=self.params[3], sigmaX=self.params[2])
        img = cv.normalize(img_b - img_a, None, 0, 1.0, cv.NORM_MINMAX, cv.CV_32F)
        return img


class GaussianBlur(Transformer):
    """
    Gaussian Blur
    """

    def __init__(self, width, height):
        """
        Constructor
        One random param: sigma
        :param width: image width
        """
        Transformer.__init__(self, width, height)
        self.code = 10
        sigma = random.uniform(1, min(width/3, 5))
        a = sigma * 3
        k_size = int(np.ceil(a) // 2 * 2 + 1)
        self.params = [sigma, k_size]

    def mutate(self, r=0.0005):
        """
        Mutate
        :param r: mutate rate
        :return: void
        """

        if np.random.choice(2, 1, p=[1-r, r]) == 1:
            self.params[0] = random.uniform(1, min(self.width/3, 5))
            a = self.params[0] * 3
            self.params[1] = int(np.ceil(a) // 2 * 2 + 1)

    def transform(self, img):
        """
        Process the image
        :param img: input image
        :return: blurred image
        """

        if img.dtype != np.float32:
            raise Exception("Input should be {} but was {}".format(np.float32, img.dtype))

        img = cv.GaussianBlur(img, ksize=(self.params[1], self.params[1]), sigmaX=self.params[0])
        return img


class Gradient(Transformer):
    """
    Gradient filters
    """
    def __init__(self, width, height):
        """
        Constructor.
        One random param decide which gradient filter to use
        :param width:
        """

        Transformer.__init__(self, width, height)
        self.code = 11
        kernel = np.ones((3, 3))
        self.params = [kernel]
        g = random.randrange(4)
        if g == 0:
            self.prewitt()
        elif g == 1:
            self.sobel()
        elif g == 2:
            self.kirsch()
        elif g == 3:
            self.scharr()
        else:
            raise Exception("Random Number out of scope {}".format(g))

    def prewitt(self):
        """
        Prewitt filter
        :return: void
        """
        if random.randrange(2) == 0:
            self.params[0] = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        else:
            self.params[0] = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])

    def sobel(self):
        """
        Sobel filter
        :return: void
        """
        if random.randrange(2) == 0:
            self.params[0] = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        else:
            self.params[0] = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

    def kirsch(self):
        """
        Kirsch filter
        :return: void
        """
        x = random.randrange(8)
        if x == 0:
            self.params[0] = np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]])
        elif x == 1:
            self.params[0] = np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])
        elif x == 2:
            self.params[0] = np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]])
        elif x == 3:
            self.params[0] = np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]])
        elif x == 4:
            self.params[0] = np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]])
        elif x == 5:
            self.params[0] = np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]])
        elif x == 6:
            self.params[0] = np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]])
        else:
            self.params[0] = np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]])

    def scharr(self):
        """
        Scharr filter
        :return: void
        """
        if random.randrange(2) == 0:
            self.params[0] = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]])
        else:
            self.params[0] = np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]])

    def mutate(self, r=0.0005):
        """
        Mutate
        :param r:
        :return:
        """
        if np.random.choice(2, 1, p=[1-r, r]) == 1:
            g = random.randrange(4)
            if g == 0:
                self.prewitt()
            elif g == 1:
                self.sobel()
            elif g == 2:
                self.kirsch()
            elif g == 3:
                self.scharr()
            else:
                raise Exception("Random number out of scope {}".format(g))

    def transform(self, img):
        """
        Process image
        :param img: input image
        :return: normalized gradient image
        """

        if img.dtype != np.float32:
            raise Exception("Input should be {} but was {}".format(np.float32, img.dtype))

        img = cv.filter2D(img, -1, kernel=self.params[0])
        img = cv.normalize(img, None, 0, 1.0, cv.NORM_MINMAX, cv.CV_32F)
        return img


class HarrisCorner(Transformer):
    """
    Harris Corner
    """

    def __init__(self, width, height):
        """
        Constructor
        Three random param: block size, kernel size, k
        :param width: image width
        """
        Transformer.__init__(self, width, height)
        self.code = 12
        block_size = random.randrange(2, width + 1)
        k_size_upper = min(width // 2 * 2 + 1, 31)
        k_size = random.randrange(3, k_size_upper + 1, step=2)
        k = random.uniform(0.01, 0.09)
        self.params = [block_size, k_size, k]

    def mutate(self, r=0.0005):
        """
        Mutate
        :param r: mutate rate
        :return:
        """
        if np.random.choice(2, 1, p=[1-r, r]) == 1:
            self.params[0] = random.randrange(2, self.width + 1)
        if np.random.choice(2, 1, p=[1-r, r]) == 1:
            k_size_upper = min(self.width // 2 * 2 + 1, 31)
            self.params[1] = random.randrange(3, k_size_upper + 1, step=2)
        if np.random.choice(2, 1, p=[1 - r, r]) == 1:
            self.params[2] = random.uniform(0.01, 0.09)

    def transform(self, img):
        """
        Process image
        :param img: input image
        :return: harris corner, normalized
        """

        if img.dtype != np.float32:
            raise Exception("Input should be {} but was {}".format(np.float32, img.dtype))

        img = cv.cornerHarris(img, blockSize=self.params[0], ksize=self.params[1], k=self.params[2])
        img = cv.normalize(img, None, 0, 1.0, cv.NORM_MINMAX, cv.CV_32F)
        return img


class IntegralTransformation(Transformer):
    """
    Image Integral
    """
    def __init__(self, width, height):
        """
        Constructor
        :param width:
        """

        Transformer.__init__(self, width, height)
        self.code = 13

    @staticmethod
    def transform(img):
        """
        Process image
        :param img: input image
        :return: normalized integral
        """

        if img.dtype != np.float32:
            raise Exception("Input should be {} but was {}".format(np.float32, img.dtype))

        img = cv.normalize(cv.integral(img), None, 0, 1.0, cv.NORM_MINMAX, cv.CV_32F)
        return img


class LaplacianEdge(Transformer):
    """
    Laplacian Filter
    """

    def __init__(self, width, height):
        """
        Constructor
        :param width: image width
        """

        Transformer.__init__(self, width, height)
        self.code = 14
        k_size = random.randrange(3, min(self.width + 1, 31), step=2)
        self.params = [k_size]

    def mutate(self, r=0.0005):
        """
        Mutate
        :param r: mutate rate
        :return: void
        """

        if np.random.choice(2, 1, p=[1-r, r]) == 1:
            self.params[0] = random.randrange(3, min(self.width + 1, 31), step=2)

    def transform(self, img):
        """
        Process image
        :param img: input image
        :return: normalized laplacian edges
        """

        if img.dtype != np.float32:
            raise Exception("Input should be {} but was {}".format(np.float32, img.dtype))

        img = cv.normalize(cv.Laplacian(img, ddepth=-1, ksize=self.params[0]), None, 0, 1.0, cv.NORM_MINMAX, cv.CV_32F)
        return img


class Log(Transformer):
    """
    Logarithm Transformation
    """
    def __init__(self, width, height):
        """
        Constructor
        :param width:
        """

        Transformer.__init__(self, width, height)
        self.code = 15

    @staticmethod
    def transform(img):
        """
        Process image with logarithm transformer to enhance contrast
        p = c * log(1 + img)
        :param img: input image
        :return:
        """

        if img.dtype == np.float32:
            img_8u = (img * 255).astype(np.uint8)
        else:
            raise Exception("Input should be {} but was {}".format(np.float32, img.dtype))

        contrast = 255/(np.log(img.max()+1))
        img = contrast * np.log(1 + img_8u.astype(np.float32))
        img = cv.normalize(img, None, 0, 1.0, cv.NORM_MINMAX, cv.CV_32F)
        return img


class MediumBlur(Transformer):
    """
    Medium Blur
    """
    def __init__(self, width, height):
        """
        Constructor
        :param width: image width
        """

        Transformer.__init__(self, width, height)
        self.code = 16
        k = random.randrange(3, min(width + 1, 15), step=2)
        self.params = [k]

    def mutate(self, r=0.0005):
        """
        Mutate
        :param r: mutate rate
        :return: void
        """
        if np.random.choice(2, 1, p=[1-r, r]) == 1:
            self.params[0] = random.randrange(3, min(self.width + 1, 15), step=2)

    def transform(self, img):
        """
        Processing Data
        :param img: input image
        :return: smoothed image
        """

        if img.dtype != np.float32:
            raise Exception("Input should be {} but was {}".format(np.float32, img.dtype))

        if self.params[0] <= 5:
            img = cv.medianBlur(img, ksize=self.params[0])
        else:
            img_8u = (img * 255).astype(np.uint8)
            img = (cv.medianBlur(img_8u, ksize=self.params[0]) / 255).astype(np.float32)
        return img


class SquareRoot(Transformer):
    """
    Square root for Gamma modification
    """
    def __init__(self, width, height):
        """
        Constructor
        :param width: image width
        """

        Transformer.__init__(self, width, height)
        self.code = 17

    @staticmethod
    def transform(img):
        """
        Must be performed on float points [0, 1] so that it have meaningful effect.
        Increase the contrast
        :param self:
        :param img:
        :return:
        """
        if img.dtype != np.float32:
            raise Exception("Input should be {} but was {}".format(np.float32, img.dtype))

        return np.sqrt(img)

