
import cv2
import os
import random
import glob
import numpy


def read_data(directory):
    rand_img = random.choice(os.listdir(directory))
    img_path = directory + "/" + rand_img
    image = cv2.imread(img_path)[:, :, 0].astype(numpy.float32) / 255.
    code = rand_img.split("_")[1]

    return image, code
    #return img_array, code_array


def img_code_init(directory):
    while True:
        yield read_data(directory)
