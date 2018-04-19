from PIL import Image
import cv2
import numpy as np
import random
import io
import os
from os.path import join
from torch.utils.data import Dataset
import config
from data.train_generator import claptcha
import logging

logger = logging.getLogger()


def randomString(possible_characters, lenght=5):
    rndLetters = (random.choice(possible_characters) for _ in range(lenght))
    return "".join(rndLetters)


def generate_example(possible_characters, lenght=5):
    """generate train example"""
    string = randomString(possible_characters,lenght=lenght)
    c = claptcha.Claptcha(source=string, font=config.font_path,
                      margin=(10, 10), noise=0,
                      size=(config.target_width, config.target_height))
    _, bytes = c.bytes
    bytes = bytes.read()
    image = Image.open(io.BytesIO(bytes))
    array = np.array(image)
    return array, string, bytes


def get_locations(image):
    """
    get locations in format (x, y, w, h) of characters on image
    """

    # Convert image to grayscale and find counters
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # threshold the image (convert it to pure black and white)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # find the contours (continuous blobs of pixels) the image
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    locations = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        locations.append((x, y, w, h))

    # Sort the detected letter images based on the x coordinate
    locations = sorted(locations, key=lambda x: x[0])
    return locations


def clipper(image, locations):
    """
    clip image based on locations and do some preprocessing
    """

    # case I: overlapping letters
    # split letters if they are overlapping
    locations_temp = []
    for x, y, w, h in locations:
        # Compare the width and height of the contour to detect letters that
        # are conjoined into one chunk
        if w / h > 1.75:
            # This contour is too wide to be a single letter!
            # Split it in half into two letter regions!
            half_width = int(w / 2)
            locations_temp.append((x, y, half_width, h))
            locations_temp.append((x + half_width, y, half_width, h))
        else:
            locations_temp.append((x, y, w, h))
    locations = locations_temp

    # case II: 'i/j' examples:
    # usually, character 'i' is presented as 2 small images 'dot' and 'stick'
    # in this case we should concat 2 images at vertical line
    # 1. detect such images: if x1 almost equals x2
    i = 0
    while i < len(locations) - 1:
        x1, y1, w1, h1 = locations[i]
        x2, y2, w2, h2 = locations[i + 1]
        if abs(x2 - x1) <= 5:
            # concat 2 elements
            x = min(x1, x2)
            y = min(y1, y2)
            w = max(x1 + w1, x2 + w2) - x
            h = max(y1 + h1, y2 + h2) - y
            locations[i] = (x, y, w, h)
            del locations[i + 1]
        else:
            i += 1

    # Generate images corresponded to each label in numpy format
    small_images = []
    n = 2  # number of extra pixels
    for x, y, w, h in locations:
        image_char = image[y - n:y + h + n, x - n:x + w + n]
        small_images.append(image_char)
    return small_images


def resize_one(image, shape=(32, 32, 3)):
    """ resize without stretching a figure and
    place the image to the upper-right corner of figure of shape `shape` """
    h_target, w_target = shape[:2]
    h, w = image.shape[:2]
    k = min(h_target / h, w_target / w)
    h_temp, w_temp = int(h * k), int(w * k)
    resized_temp = cv2.resize(image, (w_temp, h_temp), interpolation=cv2.INTER_NEAREST)
    empty = np.full(shape=shape, fill_value=255, dtype=np.uint8)
    empty[:h_temp, :w_temp, :] = resized_temp[:h_target, :w_target, :]
    return empty


def resize_set(small_images, shape=(40, 40, 3)):
    """ 1. place the image to the upper-right corner of figure of shape `shape`
        2. stack images into 1 numpy array """
    images = []
    for small_image in small_images:
        temp = resize_one(small_image, shape=shape)
        images.append(temp)
    return images


def OHE(string, char2ind):
    shape = (len(string), len(char2ind))
    arr = np.zeros(shape=shape, dtype=int)
    for i, el in enumerate(string):
        ind = char2ind[el]
        arr[i, ind] = 1
    return arr


def black_and_white_list(images):
    black_white = []
    for image in images:
        # gray scale image:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # convert image to binary:
        thresh, image_bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        black_white.append(image_bw)
    return black_white


def process_picture(image_array):
    locations = get_locations(image_array)
    small_images = clipper(image_array, locations)
    x = resize_set(small_images, shape=(32, 32, 3))
    x = black_and_white_list(x)  # get 1 channel images (n,n,3) -->  (n,n)
    x = [np.expand_dims(el, axis=0) for el in x]  # (n,n) --> (1,n,n)
    x = np.stack(x)
    x = x / 255.  # rescale to [0,1]
    x = x.astype(np.float32)
    return x


class TrainLoader(Dataset):
    def __init__(self):
        pass

    @staticmethod
    def generate_train_example():
        image_array, string, _ = generate_example(config.possible_characters,
                                                  lenght=config.number_chars_per_image)
        x = process_picture(image_array)
        y = OHE(string, config.char2ind)
        return x[0], y[0]

    def __getitem__(self, index):
        return self.generate_train_example()

    def __len__(self):
        return int(1e15)  # inf


class Test_Loader(Dataset):
    def __init__(self):
        X_test, y_test = self.upload_test_set(config.path_test_data)
        X = []
        Y = []
        for image_array, string in zip(X_test, y_test):
            x = process_picture(image_array)
            y = OHE(string, config.char2ind)
            if not len(x) == len(y) == config.number_chars_per_image:
                continue
            X.append(x)
            Y.append(y)
        self.X = np.vstack(X)
        self.Y = np.vstack(Y)

    @staticmethod
    def upload_test_set(path):
        x_backet = []
        y_backet = []
        files = os.listdir(path)
        logger.info("number of test examples is {}".format(len(files)))
        for f in files:
            image = Image.open(join(path, f))
            x = np.array(image)
            y = f.split('.')[0]  # cut '.png'
            x_backet.append(x)
            y_backet.append(y)
        return x_backet, y_backet

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)