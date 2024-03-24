#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import cv2

def read_file_list(filename, prefix=None, suffix=None):
    '''
    Reads a list of files from a line-seperated text file.

    Parameters:
        filename: Filename to load.
        prefix: File prefix. Default is None.
        suffix: File suffix. Default is None.
    '''
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.readlines()
    filelist = [x.strip() for x in content if x.strip()]
    if prefix is not None:
        filelist = [prefix + f for f in filelist]
    if suffix is not None:
        filelist = [f + suffix for f in filelist]
    return filelist

def load_png_image_as_array1(filename, return_size=False):
    # Load the image using OpenCV in grayscale
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    resized_image = cv2.resize(img, (1214, 256))

    # Convert the cropped image to a numpy array
    data = np.array(resized_image)
    # print(data.shape)  # Print the shape of the data array

    # Optionally return the size of the image
    if return_size:
        return data, resized_image.shape  # cropped_img.shape gives the height and width
    else:
        return data
