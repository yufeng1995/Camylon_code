
"""Fairly basic set of tools for real-time data augmentation on image data.
Can easily be extended to include new transformations,
new preprocessing methods, etc...
"""
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import re
from scipy import linalg
import scipy.ndimage as ndi
from six.moves import range
import os
import threading
import warnings
from os import listdir
import json
import random
import sys
import cv2
import logging
import argparse
from PIL import Image
import glob
import keras
from queue import Queue
import time


# from .. import backend as K
import keras.backend as K

try:
    from PIL import Image as pil_image
    pil_image.MAX_IMAGE_PIXELS = None
except ImportError:
    pil_image = None



def extract_foreground_mask(img, threshold=0.75, dilate_kernel=2):
    """
    Func: Get a gray image from slide
    Args: img

    Returns:gray_t

    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
    # Convert color space
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, gray_t = cv2.threshold(gray, threshold * 255, 255, cv2.THRESH_BINARY_INV)
    gray_t = cv2.dilate(gray_t, kernel)
    ret, gray_t = cv2.threshold(gray_t, threshold * 255, 255, cv2.THRESH_BINARY)

    return gray_t


def shuffle_list(list_to_be_shuffle, is_shuffle):
    """
       Args:
           list_to_be_shuffle: A list will be to shuffle.
           is_shuffle: bool, if True, list will be shuffle, if False, list will remain the same.

       Returns:
           list_to_be_shuffle:
    """
    if is_shuffle:
        shuffled_index = list(range(len(list_to_be_shuffle)))
        # random.seed(12345)
        random.shuffle(shuffled_index)
        list_to_be_shuffle = [list_to_be_shuffle[i] for i in shuffled_index]

    return list_to_be_shuffle


def transform_list_to_array(array_list, shuffle=True):
    """
        Func: transform [[image, label], [image, label], ...] to images: [batch_size, w, h, c] and labels: [batch_size, 1]
        Args:

        Returns:

    """
    assert len(array_list) != 0, logger.info('no patches to extend!')

    array_list_shuffle = shuffle_list(array_list, shuffle)

    batch_images = np.expand_dims(array_list_shuffle[0][0], axis=0)
    batch_labels = np.expand_dims(array_list_shuffle[0][1], axis=0)

    for i in range(1, len(array_list_shuffle)):
        batch_images = \
            np.concatenate((batch_images, np.expand_dims(array_list_shuffle[i][0], axis=0)))
        batch_labels = \
            np.concatenate((batch_labels, np.expand_dims(array_list_shuffle[i][1], axis=0)))

    return batch_images, batch_labels


def random_rotation(x, rg, row_axis=1, col_axis=2, channel_axis=0,
                    fill_mode='nearest', cval=0.):
    """Performs a random rotation of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        rg: Rotation range, in degrees.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        Rotated Numpy image tensor.
    """
    theta = np.pi / 180 * np.random.uniform(-rg, rg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def random_shift(x, wrg, hrg, row_axis=1, col_axis=2, channel_axis=0,
                 fill_mode='nearest', cval=0.):
    """Performs a random spatial shift of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        wrg: Width shift range, as a float fraction of the width.
        hrg: Height shift range, as a float fraction of the height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        Shifted Numpy image tensor.
    """
    h, w = x.shape[row_axis], x.shape[col_axis]
    tx = np.random.uniform(-hrg, hrg) * h
    ty = np.random.uniform(-wrg, wrg) * w
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])

    transform_matrix = translation_matrix  # no need to do offset
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def random_shear(x, intensity, row_axis=1, col_axis=2, channel_axis=0,
                 fill_mode='nearest', cval=0.):
    """Performs a random spatial shear of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        intensity: Transformation intensity.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        Sheared Numpy image tensor.
    """
    shear = np.random.uniform(-intensity, intensity)
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def random_zoom(x, zoom_range, row_axis=1, col_axis=2, channel_axis=0,
                fill_mode='nearest', cval=0.):
    """Performs a random spatial zoom of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        zoom_range: Tuple of floats; zoom range for width and height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        Zoomed Numpy image tensor.

    # Raises
        ValueError: if `zoom_range` isn't a tuple.
    """
    if len(zoom_range) != 2:
        raise ValueError('zoom_range should be a tuple or list of two floats. '
                         'Received arg: ', zoom_range)

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def random_channel_shift(x, intensity, channel_axis=0):
    x = np.rollaxis(x, channel_axis, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, channel_axis=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                                                         final_offset, order=0, mode=fill_mode, cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def standardize(x,
                preprocessing_function=None,
                rescale=None,
                channel_axis=None,
                samplewise_center=False,
                featurewise_center=False,
                samplewise_std_normalization=False,
                featurewise_std_normalization=False,
                mean=None,
                std=None,
                zca_whitening=False,
                principal_components=None,
                rng=None):
    if preprocessing_function:
        x = preprocessing_function(x)
    if rescale:
        x *= rescale
    # x is a single image, so it doesn't have image number at index 0
    img_channel_axis = channel_axis - 1
    if samplewise_center:
        x -= np.mean(x, axis=img_channel_axis, keepdims=True)
    if samplewise_std_normalization:
        x /= (np.std(x, axis=img_channel_axis, keepdims=True) + 1e-7)
    if featurewise_center:
        if mean is not None:
            x -= mean
        else:
            warnings.warn('This ImageDataGenerator specifies '
                          '`featurewise_center`, but it hasn\'t'
                          'been fit on any training data. Fit it '
                          'first by calling `.fit(numpy_data)`.')
    if featurewise_std_normalization:
        if std is not None:
            x /= (std + 1e-7)
        else:
            warnings.warn('This ImageDataGenerator specifies '
                          '`featurewise_std_normalization`, but it hasn\'t'
                          'been fit on any training data. Fit it '
                          'first by calling `.fit(numpy_data)`.')
    if zca_whitening:
        if principal_components is not None:
            flatx = np.reshape(x, (x.size))
            whitex = np.dot(flatx, principal_components)
            x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))
        else:
            warnings.warn('This ImageDataGenerator specifies '
                          '`zca_whitening`, but it hasn\'t'
                          'been fit on any training data. Fit it '
                          'first by calling `.fit(numpy_data)`.')
    return x

def random_transform(x,
                     row_axis=None,
                     col_axis=None,
                     channel_axis=None,
                     rotation_range=0.,
                     height_shift_range=0.,
                     width_shift_range=0.,
                     shear_range=0.,
                     zoom_range=0.,
                     fill_mode='nearest',
                     cval=0.,
                     channel_shift_range=0.,
                     horizontal_flip=False,
                     vertical_flip=False,
                     rng=None):

    supplied_rngs = True
    if rng is None:
        supplied_rngs = False
        rng = np.random

    # x is a single image, so it doesn't have image number at index 0
    img_row_axis = row_axis - 1
    img_col_axis = col_axis - 1
    img_channel_axis = channel_axis - 1

    # use composition of homographies
    # to generate final transform that needs to be applied
    if rotation_range:
        theta = np.pi / 180 * rng.uniform(-rotation_range, rotation_range)
    else:
        theta = 0
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    if height_shift_range:
        tx = rng.uniform(-height_shift_range, height_shift_range) * x.shape[img_row_axis]
    else:
        tx = 0

    if width_shift_range:
        ty = rng.uniform(-width_shift_range, width_shift_range) * x.shape[img_col_axis]
    else:
        ty = 0

    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])
    if shear_range:
        shear = rng.uniform(-shear_range, shear_range)
    else:
        shear = 0
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = rng.uniform(zoom_range[0], zoom_range[1], 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    transform_matrix = np.dot(np.dot(np.dot(rotation_matrix,
                                            translation_matrix),
                                     shear_matrix),
                              zoom_matrix)

    h, w = x.shape[img_row_axis], x.shape[img_col_axis]
    transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
    x = apply_transform(x, transform_matrix, img_channel_axis,
                        fill_mode=fill_mode, cval=cval)
    if channel_shift_range != 0:
        x = random_channel_shift(x,
                                 channel_shift_range,
                                 img_channel_axis)

    get_random = None
    if supplied_rngs:
        get_random = rng.rand
    else:
        get_random = np.random.random

    if horizontal_flip:
        if get_random() < 0.5:
            x = flip_axis(x, img_col_axis)

    if vertical_flip:
        if get_random() < 0.5:
            x = flip_axis(x, img_row_axis)

    return x

def array_to_img(x, dim_ordering='default', scale=True):
    """Converts a 3D Numpy array to a PIL Image instance.

    # Arguments
        x: Input Numpy array.
        dim_ordering: Image data format.
        scale: Whether to rescale image values
            to be within [0, 255].

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if invalid `x` or `dim_ordering` is passed.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    x = np.asarray(x)
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image). '
                         'Got array with shape:', x.shape)

    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Invalid dim_ordering:', dim_ordering)

    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if dim_ordering == 'th':
        x = x.transpose(1, 2, 0)
    if scale:
        x = x + max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return pil_image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return pil_image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number: ', x.shape[2])


def img_to_array(img, dim_ordering='default'):
    """Converts a PIL Image instance to a Numpy array.

    # Arguments
        img: PIL Image instance.
        dim_ordering: Image data format.

    # Returns
        A 3D Numpy array (float32).

    # Raises
        ValueError: if invalid `img` or `dim_ordering` is passed.
    """
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering: ', dim_ordering)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype='float32')
    if len(x.shape) == 3:
        if dim_ordering == 'th':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if dim_ordering == 'th':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x


def load_img(path, grayscale=False, target_size=None):
    """Loads an image into PIL format.

    # Arguments
        path: Path to image file
        grayscale: Boolean, whether to load the image as grayscale.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    img = pil_image.open(path)
    if grayscale:
        img = img.convert('L')
    else:  # Ensure 3 channel even when loaded image is grayscale
        img = img.convert('RGB')
    if target_size:
        img = img.resize((target_size[1], target_size[0]))
    return img


def list_pictures(directory, ext='jpg|jpeg|bmp|png'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match('([\w]+\.(?:' + ext + '))', f)]

class RandomCrop(object):
    def __init__(self,
                 json_path,
                 is_training=False,
                 crop_patch_nb=0,
                 max_retry_cnt=5,
                 non_zero_rate=0.7,
                 foreground_rate=1,
                 crop_width=511,
                 crop_height=511,
                 crop_channel=3,
                 is_shuffle=True,
                 readers=1,
                 num_threads=2,
                 info_maxsize=3000,
                 data_maxsize=1000):
        self.json_path = json_path
        self.is_training = is_training
        self.crop_patch_nb = crop_patch_nb
        self.max_retry_cnt = max_retry_cnt
        self.non_zero_rate = non_zero_rate
        self.foreground_rate = foreground_rate
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.crop_channel = crop_channel
        self.label = None
        self.is_shuffle = is_shuffle
        self.readers = readers
        self.num_threads = num_threads
        self.info_maxsize = info_maxsize
        self.data_maxsize = data_maxsize
        self.threads = []
        self.is_running = False
        self.info_head_lock = threading.Lock()
        self.info_tail_lock = threading.Lock()
        self.data_lock = threading.Lock()
        self.json_path_list = self.get_json_path_list()

        # create and start thread, create queue
        self._queue()
        self.start_queue_runners()

    # processing of basic patches info
    def get_json_path_list(self):
        json_path_list = glob.glob(os.path.join(self.json_path, '*.json'))

        return json_path_list

    def get_all_patch_mask_path_list(self):
        all_patches_dic_info = []
        for json_index in range(len(self.json_path_list)):
            # load json and cal numbers of patch in each json
            json_path = self.json_path_list[json_index]
            fopen = open(json_path)
            json_info = json.load(fopen)
            # json_info is a dict
            # keys:'image_id', 'data_origin', 'level', 'label' and 'patches'
            # patches is a list containing several dictionaries
            # keys: 'patch_id', 'img_path', 'mask_path' and 'patch_size'
            # print(json_path)
            self.label = json_info['label']

            nb_patches_in_each_json = len(json_info['patches'])
            for patch_index in range(nb_patches_in_each_json):
                patch_dic_info = json_info['patches'][patch_index]
                all_patches_dic_info.append(patch_dic_info)
        # all_patches_dic_info.sort()
        # print(all_patches_dic_info)
        return all_patches_dic_info

    def shuffle(self, is_shuffle):
        all_patches_dic_info = self.get_all_patch_mask_path_list()
        all_patches_dic_info = shuffle_list(all_patches_dic_info, is_shuffle)

        return all_patches_dic_info

    def get_nb_samples_per_epoch(self):
        return len(self.get_all_patch_mask_path_list())

    def get_crop_patch_np(self):
        return self.crop_patch_nb

    # random crop function of training set
    def random_crop_once(self, json_info_patch):
        # 1 get basic info of patch and load mask
        patch_width = json_info_patch['patch_size'][0]
        patch_height = json_info_patch['patch_size'][1]
        # if (patch_width > self.crop_width) and (patch_height > self.crop_height):
        if (patch_width >= self.crop_width) and (patch_height >= self.crop_height):
            logger.info('====image_path====: {}' .format(json_info_patch['img_path']))
            logger.info('=====mask_path====: {}' .format(json_info_patch['mask_path']))
            if json_info_patch['mask_path'] != 'None' :
                mask_pil = Image.open(json_info_patch['mask_path'])
            image_pil = Image.open(json_info_patch['img_path'])
            for iter in range(self.max_retry_cnt):
                # 2 Get random coordinate
                loc_x = patch_width - self.crop_width
                loc_y = patch_height - self.crop_height
                # get_x = random.randint(0, loc_x - 1)
                get_x = random.randint(0, loc_x)
                # get_y = random.randint(0, loc_y - 1)
                get_y = random.randint(0, loc_y)
                s_x = json_info_patch['patch_id'][0] + get_x
                s_y = json_info_patch['patch_id'][1] + get_y
                # 3 crop mask, cal non_zeros_rate
                if json_info_patch['mask_path'] == 'None':
                    # middle patch without annotation, regard non_zero_count in mask as 1
                    non_zero_rate = 1
                else:
                    # middle patch with annotation, random crop mask
                    mask_pil_roi = mask_pil.crop((get_x, get_y, get_x + self.crop_width, get_y + self.crop_height))
                        # mask_pil.crop((get_y, get_x, get_y + self.crop_width, get_x + self.crop_height))
                        # mask_pil.crop((get_x, get_y, get_x + self.crop_width, get_y + self.crop_height))
                    mask_pil_roi_np = np.array(mask_pil_roi)
                    non_zero_count = np.count_nonzero(mask_pil_roi_np)
                    non_zero_rate = non_zero_count / (self.crop_width * self.crop_height)
                logger.info('=====non_zero_rate====: {}' .format(non_zero_rate))
                # 4 decide actions according to non_zeros_rate
                if non_zero_rate >= self.non_zero_rate:
                    # 4.1 crop image
                    image_pil_roi = \
                        image_pil.crop((get_x, get_y, get_x + self.crop_width, get_y + self.crop_height))
                    image_pil_roi_np = np.array(image_pil_roi)

                    if self.foreground_rate != 1:
                        # 4.1.1 need to background filter
                        # image_thresh = extract_foreground_mask(image_pil_roi_np)
                        fgmask_pil = Image.open(json_info_patch['fgmask_path']).crop((get_x, get_y, get_x + self.crop_width, get_y + self.crop_height))

                        fg_count = np.count_nonzero(np.array(fgmask_pil))
                        fg_rate = fg_count / (self.crop_width * self.crop_height)
                        logger.info('train {} image, after extract_foreground, foreground_rate is: {}' .format(self.label, fg_rate))
                        if fg_rate < self.foreground_rate:
                            # 4.1.1.1 crop again
                            continue
                    else:
                        # 4.1.2 do not need to background filter, regard foreground_rate as 1
                        fg_rate = 1
                    break
                else:
                    # 4.2 crop again
                    fg_rate = 0
                    continue

            if (iter == (self.max_retry_cnt - 1)) and (fg_rate < self.foreground_rate):
                image_pil_roi_np = None
        else:
            image_pil_roi_np = None
            fg_rate = 0
            logger.info("Error! Patch size smaller that target!")
        # print(json_info_patch['img_path'].split(".")[0])
        # print(json_info_patch['img_path'].split(".p"))
        img_path = json_info_patch['img_path'].split(".p")[0] + "_sx_" + str(s_x) + "_sy_" +str(s_y) + ".png"
        return image_pil_roi_np, fg_rate , str(img_path)

    # function of eval set
    def eval_image(self, json_info_patch):
        # 1 get basic info of patch and load mask
        patch_width = json_info_patch['patch_size'][0]
        patch_height = json_info_patch['patch_size'][1]
        if (patch_width == self.crop_width) and (patch_height == self.crop_height):
            # 2 open image and transform to array
            logger.info('====eval img_path===: {}' .format(json_info_patch['img_path']))
            image_pil_roi = Image.open(json_info_patch['img_path'])
            image_pil_roi_np = np.array(image_pil_roi)
            fg_rate = 1     
            img_path = json_info_patch['img_path']
            # if self.foreground_rate != 1:
            #     # 3.1 need to filter background
            #     if json_info_patch['mask_path'] != 'None':
            #         # 3.1.1 middle patch(with mask) need to filter background
            #         fgmask_pil = Image.open(json_info_patch['fgmask_path'])
            #         # image_thresh = extract_foreground_mask(fgmask_pil)
            #         fg_count = np.count_nonzero(np.array(image_thresh))
            #         fg_rate = fg_count / (self.crop_width * self.crop_height)
            #         logger.info('eval {} image, foreground_rate: {}' .format(self.label, fg_rate))

            #     else:
            #         # 3.1.2 middle patch(without mask) do not need to filter background
            #         fg_rate = 1
            # else:
            #     # 3.2 do not need to filter background, regard foreground_rate as 1
            #     fg_rate = 1
        else:
            fg_rate = 0
            image_pil_roi_np = None
            logger.info('the size of eval image is wrong!')

        return image_pil_roi_np, fg_rate, img_path

    # queue
    def _queue(self):
        self.info_queue = Queue(maxsize=self.info_maxsize)
        self.data_queue = Queue(maxsize=self.data_maxsize)

    def get_queue(self):
        return self.data_queue

    def close_queue(self):
        # https://github.com/mwfrojdman/cpython/blob/closeable_queue/Lib/queue.py
        # https://stackoverflow.com/questions/6517953/clear-all-items-from-the-queue
        self.info_queue.mutex.acquire()
        self.info_queue.queue.clear()
        self.info_queue.not_empty.notify_all()
        self.info_queue.not_full.notify_all()
        self.info_queue.all_tasks_done.notify_all()
        self.info_queue.unfinished_tasks = 0
        self.info_queue.mutex.release()

        self.data_queue.mutex.acquire()
        self.data_queue.queue.clear()
        self.data_queue.not_empty.notify_all()
        self.data_queue.not_full.notify_all()
        self.data_queue.all_tasks_done.notify_all()
        self.data_queue.unfinished_tasks = 0
        self.data_queue.mutex.release()

    # thread
    def info_input_producer(self, info_list):
        while self.get_work_threads_status():
            for info in info_list:
                if not self.get_work_threads_status():
                    break
                self.info_tail_lock.acquire()
                self.info_queue.put(info)
                self.info_tail_lock.release()
        logger.info('*************info thread is end!**************')

    def data_input_producer(self):
        while self.get_work_threads_status():
            while (not self.info_queue.empty()) and self.get_work_threads_status():
                self.info_head_lock.acquire()
                patch_dic_info = self.info_queue.get()
                self.info_head_lock.release()
                if self.is_training:
                    image, rate ,img_path = self.random_crop_once(patch_dic_info)
                else:
                    image, rate ,img_path = self.eval_image(patch_dic_info)

                if rate >= self.foreground_rate:
                    if self.is_training :
                    # print(img_path)
                        self.data_lock.acquire()
                        self.data_queue.put([image.astype(np.float32), self.label, img_path])
                        self.data_lock.release()
                    else:
                        self.data_lock.acquire()
                        self.data_queue.put([image.astype(np.float32), self.label, img_path])
                        self.data_lock.release()
        logger.info('*************data thread is end!***************')

    def start_queue_runners(self):
        for _ in range(self.readers):
            t = threading.Thread(target=self.info_input_producer, args=(self.shuffle(self.is_shuffle), ))
            self.threads.append(t)
        for _ in range (self.num_threads):
            t = threading.Thread(target=self.data_input_producer, args=())
            self.threads.append(t)

        self.set_work_threads_status_start()
        for i in range(len(self.threads)):
            self.threads[i].setDaemon(True)
            self.threads[i].start()

    def get_threads(self):
        return self.threads

    def set_work_threads_status_start(self):
        self.is_running = True

    def set_work_threads_status_stop(self):
        self.is_running = False

    def get_work_threads_status(self):
        return self.is_running

    def end_work_threads_and_queues(self):
        self.close_queue()
        for index, thread in enumerate(self.get_threads()):
            thread.join()
        self.close_queue()


class ImageDataGenerator(object):
    """Generate minibatches of image data with real-time data augmentation.

    # Arguments
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        zca_whitening: apply ZCA whitening.
        rotation_range: degrees (0 to 180).
        width_shift_range: fraction of total width.
        height_shift_range: fraction of total height.
        shear_range: shear intensity (shear angle in radians).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        channel_shift_range: shift range for each channels.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided
            (before applying any other transformation).
        preprocessing_function: function that will be implied on each input.
            The function will run before any other modification on it.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode it is at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "tf".
        pool: an open multiprocessing.Pool that will be used to
            process multiple images in parallel. If left off or set to
            None, then the default serial processing with a single
            process will be used.
    """

    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 dim_ordering='default',
                 pool=None,
                 nb_gpu=4):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.featurewise_center = featurewise_center
        self.samplewise_center = samplewise_center
        self.featurewise_std_normalization = featurewise_std_normalization
        self.samplewise_std_normalization = samplewise_std_normalization
        self.zca_whitening = zca_whitening
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.channel_shift_range = channel_shift_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rescale = rescale
        self.preprocessing_function = preprocessing_function
        self.pool = pool
        self.nb_gpu=nb_gpu

        if dim_ordering not in {'tf', 'th'}:
            raise ValueError('dim_ordering should be "tf" (channel after row and '
                             'column) or "th" (channel before row and column). '
                             'Received arg: ', dim_ordering)
        self.dim_ordering = dim_ordering
        if dim_ordering == 'th':
            self.channel_axis = 1
            self.row_axis = 2
            self.col_axis = 3
        if dim_ordering == 'tf':
            self.channel_axis = 3
            self.row_axis = 1
            self.col_axis = 2

        self.mean = None
        self.std = None
        self.principal_components = None

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('zoom_range should be a float or '
                             'a tuple or list of two floats. '
                             'Received arg: ', zoom_range)

    def flow(self, X, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='jpeg'):
        return NumpyArrayIterator(
            X, y, self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            dim_ordering=self.dim_ordering,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            pool=self.pool)

    def flow_from_directory(self, directory,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='jpeg',
                            nb_gpu=4,
                            follow_links=False,
                            phase=None,
                            save_list_dir=None):
        return DirectoryIterator(
            directory, self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            dim_ordering=self.dim_ordering,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            pool=self.pool,
            nb_gpu=nb_gpu,
            phase=phase,
            save_list_dir=save_list_dir)

    def flow_from_json(self, json_file_path,
                            labels,
                            nb_per_class,
                            foreground_rate_per_class,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='jpeg',
                            follow_links=False,
                            nb_gpu=4,
                            is_training=True):
        return JsonIterator(
            json_file_path, self,labels,nb_per_class,
            foreground_rate_per_class,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            dim_ordering=self.dim_ordering,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            pool=self.pool,
            nb_gpu=nb_gpu,
            is_training=is_training)

    def pipeline(self):
        """A pipeline of functions to apply in order to an image.
        """
        return [
            (random_transform, dict(
                row_axis=self.row_axis,
                col_axis=self.col_axis,
                channel_axis=self.channel_axis,
                rotation_range=self.rotation_range,
                height_shift_range=self.height_shift_range,
                width_shift_range=self.width_shift_range,
                shear_range=self.shear_range,
                zoom_range=self.zoom_range,
                fill_mode=self.fill_mode,
                cval=self.cval,
                channel_shift_range=self.channel_shift_range,
                horizontal_flip=self.horizontal_flip,
                vertical_flip=self.vertical_flip)
            ),

            (standardize, dict(
                preprocessing_function=self.preprocessing_function,
                rescale=self.rescale,
                channel_axis=self.channel_axis,
                samplewise_center=self.samplewise_center,
                samplewise_std_normalization=self.samplewise_std_normalization,
                featurewise_center=self.featurewise_center,
                mean=self.mean,
                featurewise_std_normalization=self.featurewise_std_normalization,
                std=self.std,
                zca_whitening=self.zca_whitening,
                principal_components=self.principal_components)
            )
        ]

    def standardize(self, x):
        return standardize(x,
            preprocessing_function=self.preprocessing_function,
            rescale=self.rescale,
            channel_axis=self.channel_axis,
            samplewise_center=self.samplewise_center,
            samplewise_std_normalization=self.samplewise_std_normalization,
            featurewise_center=self.featurewise_center,
            mean=self.mean,
            featurewise_std_normalization=self.featurewise_std_normalization,
            std=self.std,
            zca_whitening=self.zca_whitening,
            principal_components=self.principal_components)

    def random_transform(self, x):
        return random_transform(x,
            row_axis=self.row_axis,
            col_axis=self.col_axis,
            channel_axis=self.channel_axis,
            rotation_range=self.rotation_range,
            height_shift_range=self.height_shift_range,
            width_shift_range=self.width_shift_range,
            shear_range=self.shear_range,
            zoom_range=self.zoom_range,
            fill_mode=self.fill_mode,
            cval=self.cval,
            channel_shift_range=self.channel_shift_range,
            horizontal_flip=self.horizontal_flip,
            vertical_flip=self.vertical_flip)

    def fit(self, x,
            augment=False,
            rounds=1,
            seed=None):
        """Required for featurewise_center, featurewise_std_normalization
        and zca_whitening.

        # Arguments
            x: Numpy array, the data to fit on. Should have rank 4.
                In case of grayscale data,
                the channels axis should have value 1, and in case
                of RGB data, it should have value 3.
            augment: Whether to fit on randomly augmented samples
            rounds: If `augment`,
                how many augmentation passes to do over the data
            seed: random seed.

        # Raises
            ValueError: in case of invalid input `x`.
        """
        x = np.asarray(x)
        if x.ndim != 4:
            raise ValueError('Input to `.fit()` should have rank 4. '
                             'Got array with shape: ' + str(x.shape))
        if x.shape[self.channel_axis] not in {1, 3, 4}:
            raise ValueError(
                'Expected input to be images (as Numpy array) '
                'following the dimension ordering convention "' + self.dim_ordering + '" '
                '(channels on axis ' + str(self.channel_axis) + '), i.e. expected '
                'either 1, 3 or 4 channels on axis ' + str(self.channel_axis) + '. '
                'However, it was passed an array with shape ' + str(x.shape) +
                ' (' + str(x.shape[self.channel_axis]) + ' channels).')

        if seed is not None:
            np.random.seed(seed)

        x = np.copy(x)
        if augment:
            ax = np.zeros(tuple([rounds * x.shape[0]] + list(x.shape)[1:]))
            for r in range(rounds):
                for i in range(x.shape[0]):
                    ax[i + r * x.shape[0]] = self.random_transform(x[i])
            x = ax

        if self.featurewise_center:
            self.mean = np.mean(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.mean = np.reshape(self.mean, broadcast_shape)
            x -= self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.std = np.reshape(self.std, broadcast_shape)
            x /= (self.std + K.epsilon())

        if self.zca_whitening:
            flat_x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
            sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
            u, s, _ = linalg.svd(sigma)
            self.principal_components = np.dot(np.dot(u, np.diag(1. / np.sqrt(s + 10e-7))), u.T)


class Iterator(object):

    def __init__(self, batch_size, shuffle, seed):
        # self.n_total = n_total
        # self.n_neg = n_negative
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        # self.index_generator_n_total = self._flow_index(n_total, batch_size, shuffle, seed)
        # self.index_generator_negative = self._flow_index(n_negative, batch_size, shuffle, seed)

        # create multiple random number generators to be used separately in
        # each process when using a multiprocessing.Pool
        if seed:
            self.rngs = [np.random.RandomState(seed + i) for i in range(batch_size)]
        else:
            self.rngs = [np.random.RandomState(i) for i in range(batch_size)]



    def end_subthreads(objects):
        """
            Args:
                objects: object of RandomCrop class
            Returns:
                None
        """
        for obj in objects:
            obj.set_work_threads_status_stop()
            obj.end_work_threads_and_queues()


    def image_data_generator(objects, batch_size):
        """
            Args:
                objects:
            Returns:
                batch_images_and_labels: [[iamge, label], [image, label], ...]
        """
        # creat RandomCrop obj for each label and generate samples [[iamge, label], [image, label], ...]
        while True:
            start_time = time.time()
            batch_images_and_labels = []
            for obj in objects:
                queue = obj.get_queue()
                while not queue.empty():
                    for _ in range(obj.get_crop_patch_np()):
                        batch_images_and_labels.append(queue.get())
                    break

            if len(batch_images_and_labels) == batch_size:
                end_time = time.time()
                logger.info('=============================================')
                logger.info('batch time: {}' .format(end_time - start_time))
                logger.info('=============================================')
                yield batch_images_and_labels


    def get_instance_objects(train_base_path,
                            eval_base_path,
                            labels,
                            nb_per_class,
                            max_retry_cnt=5,
                            non_zero_rate=0.5,
                            crop_width=511,
                            crop_height=511,
                            crop_channel=3,
                            is_training=False,
                            is_shuffle=True,
                            readers=1,
                            num_threads=2,
                            info_maxsize=3000,
                            data_maxsize=1000,
                            foreground_rate_per_class=[]):
        """
            Args:
                train_base_path: Base path of training set, for detail info, see README.md.
                train_base_path: Base path of eval set, for detail info, see README.md.
                labels: List of all classes need to be generator.
                nb_per_class: List of numbers of samples for each class.
                max_retry_cnt: The max retry counts of random crop, default 5.
                non_zero_rate: The rate of non zero in mask after random crop, default 0.7,
                    set non_zero_rate=1 for middle patch without mask.
                crop_width: The width of random crop image.
                crop_height: The height of random crop image.
                crop_channel: The channels of random crop image.
                is_training: Default 'False', if set to 'True', generator data for training, or for eval.
                is_shuffle: Default 'True', If set to 'True', it will be shuffle.
                readers: The number of threads which push info into info queue.
                num_threads: The number of threads which push data into data queue.
                info_maxsize: The capacity of info queue.
                data_maxsize: The capacity of data queue.
                foreground_rate_per_class: Default [], if len(foreground_rate_per_class) != len(labels), it
                    will be extend to list of 1 with len(labels), 1 stands for no background filter.
                    e.g. labels = [tumor, benign, insitu, invasive], foreground_rate_per_class = [0.3, 0.5]
                    then, foreground_rate_per_class will be extend to [0.3, 0.5, 1, 1].
            Returns:
                List of generators for each class in labels list.

        """
        assert len(labels) == len(nb_per_class), \
            logger.info('the length of labels is unequal to the length of nb_per_class!')

        if len(foreground_rate_per_class) != len(labels):
            len_need_to_extend = len(labels) - len(foreground_rate_per_class)
            foreground_rate_per_class.extend([1] * len_need_to_extend)

        objs = []
        nb_samples_per_epoch = []
        for label, num, foreground_rate in zip(labels, nb_per_class, foreground_rate_per_class):
            if is_training:
                if label == "normal":
                    # json_path = "/mnt/disk_share/data/breast_cancer/lymph_node/postoperative/middel_patch/normal/fg_mask_litter"
                    json_path = "/mnt/disk_share/data/breast_cancer/lymph_node/postoperative/new_normal_train_set/json_fgmask/"
                elif label == "tumor":
                    # json_path = "/home/zhangyufeng/json_fgmask/"
                    # json_path = "/mnt/disk_share/data/breast_cancer/lymph_node/postoperative/middel_patch/hard_tumor/json_mask/"
                    json_path = "/mnt/disk_share/data/breast_cancer/lymph_node/postoperative/new_train_set/json_fgmask/"
                    # json_path = "/mnt/disk_share/data/breast_cancer/lymph_node/postoperative/test_wsi_xml_one/save/tumor/clean_fg_mask_json"
                    # json_path =  "/mnt/disk_share/data/breast_cancer/lymph_node/postoperative/middel_patch/tumor/third_clean_data_2000_to_224_0.7_mask/json/"
                elif label == "hard_normal":
                    # json_path = "/mnt/disk_share/data/breast_cancer/lymph_node/postoperative/middel_patch/hard_normal/"
                    json_path = "/mnt/disk_share/data/breast_cancer/lymph_node/postoperative/new_hard_example/save_patch_not_clear/1/normal_fg_mask_json/"
                elif label == "hard_tumor":
                    json_path = "/mnt/disk_share/data/breast_cancer/lymph_node/postoperative/middel_patch/hard_tumor/json_mask/"
                    # json_path = "/mnt/disk_share/data/breast_cancer/lymph_node/postoperative/new_hard_example/save_patch_not_clear/1/tumor_fg_mask_json/"
                elif label == "lymphatic_sinusoid":
                    json_path = "/mnt/disk_share/data/breast_cancer/lymph_node/postoperative/linbadou/save/linbadou/json_fgmask"
                # json_path = os.path.join(train_base_path, label + '/third_cleaned_hard_sample_json')
            else:
                if label == "normal":
                    # json_path = "/mnt/disk_share/data/breast_cancer/lymph_node/postoperative/new_normal_valid_set/224_224_debug/"
                    json_path = "/mnt/disk_share/data/breast_cancer/lymph_node/postoperative/new_normal_valid_set/224_224_json_3W/"
                elif label == "tumor":
                    # json_path = "/mnt/disk_share/data/breast_cancer/lymph_node/postoperative/new_valid_set/224_debug/"
                    json_path = "/mnt/disk_share/data/breast_cancer/lymph_node/postoperative/new_valid_set/224_json_first_cleaned/"
                elif label == "hard_normal":
                    json_path = "/mnt/disk_share/yufeng/train_code_Camylon/Camelyon/test_wrong_picture/only_hard/normal_pred_tumor_json/"
                elif label == "hard_tumor":
                    json_path = "/mnt/disk_share/yufeng/train_code_Camylon/Camelyon/test_wrong_picture/only_hard/tumor_pred_normal_json/"
                # json_path = os.path.join(eval_base_path, label + '/json')
            # import pdb
            # pdb.set_trace()
            obj = RandomCrop(json_path=json_path,
                            is_training=is_training,
                            crop_patch_nb=num,
                            max_retry_cnt=max_retry_cnt,
                            non_zero_rate=non_zero_rate,
                            foreground_rate=foreground_rate,
                            crop_width=crop_width,
                            crop_height=crop_height,
                            crop_channel=crop_channel,
                            is_shuffle=is_shuffle,
                            readers=readers,
                            num_threads=num_threads,                        
                            info_maxsize=info_maxsize,
                            data_maxsize=data_maxsize)
            objs.append(obj)
            nb_samples_per_epoch.append(obj.get_nb_samples_per_epoch())

        return objs, nb_samples_per_epoch
    def reset(self):
        self.batch_index = 0

    def _flow_index(self, n, batch_size=32, shuffle=False, seed=None):
        # ensure self.batch_index is 0
        self.reset()
        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(n)
                if shuffle:
                    index_array = np.random.permutation(n)
                else:
                    index_array = index_array[: self.n]

            current_index = (self.batch_index * batch_size) % n

            if n >= current_index + int(batch_size/2):
                current_batch_size = int(batch_size/2)
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0
            self.total_batches_seen += 1

            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)


    def __iter__(self):
        # needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


def process_image_pipeline(tup):
    """ Worker function for NumpyArrayIterator multiprocessing.Pool
    """
    (pipeline, x, rng) = tup
    x = x.astype('float32')
    for (func, kwargs) in pipeline:
        x = func(x, rng=rng, **kwargs)
    return x

def process_image_pipeline_dir(tup):
    """ Worker function for DirectoryIterator multiprocessing.Pool
    """
    (pipeline, fname, directory, grayscale,
    target_size, dim_ordering, rng) = tup
    img = load_img(os.path.join(directory, fname),
                   grayscale=grayscale,
                   target_size=target_size)
    x = img_to_array(img, dim_ordering=dim_ordering)
    for (func, kwargs) in pipeline:
        x = func(x, rng=rng, **kwargs)
    return x

def process_image_pipeline_img(tup):
    """ Worker function for DirectoryIterator multiprocessing.Pool
    """
    (pipeline,batch_generator, grayscale,
    target_size, dim_ordering, rng) = tup
    x = batch_generator[0].astype(np.float32)
    for (func, kwargs) in pipeline:
        x = func(x, rng=rng, **kwargs)
    # print(batch_generator[1])
    if batch_generator[1] == "normal":
        y=0
    elif batch_generator[1] == "tumor" :
        y=1
    elif batch_generator[1] == "hard_normal" :
        y=0
    elif batch_generator[1] == "hard_tumor":
        y=1
    elif batch_generator[1] == "linbadou":
        y =0
    # elif batch_generator[1] == "lymphocyte" :
    #     y=0
    path =  batch_generator[2]
    return x , y , path




########################################
# Functions for filtering
########################################
def extract_foreground_mask(img):
    '''
    Extract the slide foreground as the binary mask 255 or 0
    threshold -> dilate -> threshold
    Input:
        slide: h*w*3 array, the downsampled slided image
    Output:
        gray_t: h*w array, the foreground is labeled 1 and the other region is labeled 0
    '''
    threshold = 0.8
    dilate_kernel = 2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))

    # Convert color space
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    ret, gray_t = cv2.threshold(gray, threshold * 255, 255, cv2.THRESH_BINARY_INV)
    gray_t = cv2.dilate(gray_t, kernel)
    ret, gray_t = cv2.threshold(gray_t, threshold * 255, 255, cv2.THRESH_BINARY)

    return gray_t

def process_bgs(img):
    img_binary = extract_foreground_mask(img)
    img_thresh = img_binary

    return img_thresh

## TODO: cut a patch from give image, location=(x,y), target_size=(w,h)
def process_image_pipeline_json(tup):

    (pipeline, patch_meta, grayscale,
     target_size, dim_ordering, rng) = tup   #每个batch中对应位置的随机性是一样的。

    fname = patch_meta["img_path"]
    fmaskname = patch_meta["mask_path"]
    fstatus = patch_meta["label"]

    nb_crop = 7
    threshold = 0.5

    # return x and y
    # A tumor patch or a normal patch?
    # If it is a tumor patch:
    if fstatus == "tumor":
        # Load img and corresponding mask.
        if fname.startswith('/disk1'):
            fname = '/mnt/data/jin_data/lymph_private/0124_finetune_processed/middle_patch/train/tumor/' + os.path.basename(fname)
        if fmaskname.startswith('/disk1'):
            fmaskname = '/mnt/data/jin_data/lymph_private/0124_finetune_processed/middle_patch/train/mask/' + os.path.basename(fmaskname)

        img_original = load_img(fname,
                       grayscale=grayscale,
                       target_size=None)
        mask_original = load_img(fmaskname,
                        grayscale=grayscale,
                        target_size=None)

        # mask_verify = np.array(mask)
        # mask_np = mask_verify[:, :, 0]

        # Check the size of the image:
        width = img_original.size[0]
        height = img_original.size[1]

        # if the image is larger than target size
        if (width > target_size[0]) and (height > target_size[1]):
            for i in range(10):
                # 1. Get random coordinate
                loc_x = width - target_size[0]
                loc_y = height - target_size[1]

                get_x = random.randint(0, loc_x - 1)
                get_y = random.randint(0, loc_y - 1)

                # 2. Crop the image
                img = img_original.crop((get_x, get_y,
                                        get_x + target_size[0],
                                        get_y + target_size[1]))
                x = img_to_array(img, dim_ordering=dim_ordering)

                for (func, kwargs) in pipeline:
                    x = func(x, rng=rng, **kwargs)

                # 2.5 Check the ratio of white pixels in the image
                img_thresh = process_bgs(img=img_to_array(img, dim_ordering=dim_ordering))
                total_pixel_m = float(img_thresh.shape[0] * img_thresh.shape[1])
                nb_foreground = float(np.count_nonzero(img_thresh))
                foreground_ratio = float(nb_foreground / total_pixel_m)

                # 3. Crop the mask
                get_mask = mask_original.crop((get_x, get_y,
                                            get_x + target_size[0],
                                            get_y + target_size[1]))
                get_mask = img_to_array(get_mask, dim_ordering=dim_ordering)

                # 4. Calculate mask label
                total_pixel = float(get_mask.shape[0] * get_mask.shape[1])
                tumor_pixel = float(np.count_nonzero(get_mask[:, :, 0]))
                tumor_rate = float(tumor_pixel / total_pixel)

                if (tumor_rate >= threshold) and (foreground_ratio >= threshold):
                    y = 1
                    return x, y
                elif ((tumor_rate < threshold) or (foreground_ratio < threshold)) and (i < (nb_crop-1)):
                    continue
                elif ((tumor_rate < threshold) or (foreground_ratio < threshold)) and (i == (nb_crop-1)):
                    y = 0
                    return x, y

        # If the image is already smaller than target, there should be sth wrong
        else:
            print ("Error! Patch size smaller that target!")

    # If it is a normal patch
    elif fstatus == "lymphocyte":
    # Load img and corresponding mask.
        if fname.startswith('/disk1'):
            fname = '/mnt/data/jin_data/lymph_private/0124_finetune_processed/middle_patch/train/tumor/' + os.path.basename(fname)
        if fmaskname.startswith('/disk1'):
            fmaskname = '/mnt/data/jin_data/lymph_private/0124_finetune_processed/middle_patch/train/mask/' + os.path.basename(fmaskname)

        img_original = load_img(fname,
                        grayscale=grayscale,
                        target_size=None)
        mask_original = load_img(fmaskname,
                        grayscale=grayscale,
                        target_size=None)

        # mask_verify = np.array(mask)
        # mask_np = mask_verify[:, :, 0]

        # Check the size of the image:
        width = img_original.size[0]
        height = img_original.size[1]

        # if the image is larger than target size
        if (width > target_size[0]) and (height > target_size[1]):
            for i in range(10):
                # 1. Get random coordinate
                loc_x = width - target_size[0]
                loc_y = height - target_size[1]

                get_x = random.randint(0, loc_x - 1)
                get_y = random.randint(0, loc_y - 1)

                # 2. Crop the image
                img = img_original.crop((get_x, get_y,
                                        get_x + target_size[0],
                                        get_y + target_size[1]))
                x = img_to_array(img, dim_ordering=dim_ordering)

                for (func, kwargs) in pipeline:
                    x = func(x, rng=rng, **kwargs)

                # 2.5 Check the ratio of white pixels in the image
                img_thresh = process_bgs(img=img_to_array(img, dim_ordering=dim_ordering))
                total_pixel_m = float(img_thresh.shape[0] * img_thresh.shape[1])
                nb_foreground = float(np.count_nonzero(img_thresh))
                foreground_ratio = float(nb_foreground / total_pixel_m)

                # 3. Crop the mask
                get_mask = mask_original.crop((get_x, get_y,
                                            get_x + target_size[0],
                                            get_y + target_size[1]))
                get_mask = img_to_array(get_mask, dim_ordering=dim_ordering)

                # 4. Calculate mask label
                total_pixel = float(get_mask.shape[0] * get_mask.shape[1])
                lymphocyte_pixel = float(np.count_nonzero(get_mask[:, :, 0]))
                lymphocyte_rate = float(lymphocyte_pixel / total_pixel)

                if (lymphocyte_rate >= threshold) and (foreground_ratio >= threshold):
                    y = 0
                    return x, y
                elif ((lymphocyte_rate < threshold) or (foreground_ratio < threshold)) and (i < (nb_crop-1)):
                    continue
                elif ((lymphocyte_rate < threshold) or (foreground_ratio < threshold)) and (i == (nb_crop-1)):
                    y = 0
                    return x, y

            # If the image is already smaller than target, there should be sth wrong
            else:
                print ("Error! Patch size smaller that target!")

        
        # if fname.startswith('/disk1'):
        #     fname = '/mnt/data/jin_data/lymph_private/0124_finetune_processed/middle_patch/train/normal/' + os.path.basename(fname)

        # img_original = load_img(fname,
        #                grayscale=grayscale,
        #                target_size=None)

        # # The label for a normal patch must be zero
        # y = 0

        # # Check the size of the image:
        # width = img_original.size[0]
        # height = img_original.size[1]

        # # If the image is larger than target, random crop
        # if (width > target_size[0]) and (height > target_size[1]):
        #     for i in range(nb_crop):
        #         loc_x = width - target_size[0]
        #         loc_y = height - target_size[1]

        #         get_x = random.randint(0, loc_x - 1)
        #         get_y = random.randint(0, loc_y - 1)

        #         img = img_original.crop((get_x, get_y,
        #                         get_x + target_size[0],
        #                         get_y + target_size[1]))

        #         # Img to array and use the functions in pipeline for augmentation
        #         x = img_to_array(img, dim_ordering=dim_ordering)

        #         for (func, kwargs) in pipeline:
        #             x = func(x, rng=rng, **kwargs)

        #         # Check the ratio of white pixels in the image
        #         img_thresh = process_bgs(img=img_to_array(img, dim_ordering=dim_ordering))
        #         total_pixel_m = float(img_thresh.shape[0] * img_thresh.shape[1])
        #         nb_foreground = float(np.count_nonzero(img_thresh))
        #         foreground_ratio = float(nb_foreground / total_pixel_m)

        #         if (foreground_ratio >= threshold):
        #             return x, y
        #         elif (foreground_ratio < threshold) and (i < (nb_crop-1)):
        #             continue
        #         elif (foreground_ratio < threshold) and (i == (nb_crop-1)):
        #             return x, y

        else:
            print("Error! Patch size smaller than target!")

        return x, y



class NumpyArrayIterator(Iterator):

    def __init__(self, x, y, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 dim_ordering='default',
                 save_to_dir=None, save_prefix='', save_format='jpeg',
                 pool=None):
        if y is not None and len(x) != len(y):
            raise ValueError('X (images tensor) and y (labels) '
                             'should have the same length. '
                             'Found: X.shape = %s, y.shape = %s' %
                             (np.asarray(x).shape, np.asarray(y).shape))
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.x = np.asarray(x)
        if self.x.ndim != 4:
            raise ValueError('Input data in `NumpyArrayIterator` '
                             'should have rank 4. You passed an array '
                             'with shape', self.x.shape)
        channels_axis = 3 if dim_ordering == 'tf' else 1
        if self.x.shape[channels_axis] not in {1, 3, 4}:
            raise ValueError('NumpyArrayIterator is set to use the '
                             'dimension ordering convention "' + dim_ordering + '" '
                             '(channels on axis ' + str(channels_axis) + '), i.e. expected '
                             'either 1, 3 or 4 channels on axis ' + str(channels_axis) + '. '
                             'However, it was passed an array with shape ' + str(self.x.shape) +
                             ' (' + str(self.x.shape[channels_axis]) + ' channels).')
        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = None
        self.image_data_generator = image_data_generator
        self.dim_ordering = dim_ordering
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.pool = pool

        super(NumpyArrayIterator, self).__init__(x.shape[0], batch_size, shuffle, seed)

    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel

        batch_x = None

        if self.pool:
            pipeline = self.image_data_generator.pipeline()
            result = self.pool.map(process_image_pipeline, (
                (pipeline, self.x[j], self.rngs[i%self.batch_size])
                for i, j in enumerate(index_array)))
            batch_x = np.array(result)
        else:
            batch_x = np.zeros(tuple([current_batch_size] + list(self.x.shape)[1:]))
            for i, j in enumerate(index_array):
                x = self.x[j]
                x = self.image_data_generator.random_transform(x.astype('float32'))
                x = self.image_data_generator.standardize(x)
                batch_x[i] = x

        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        if self.y is None:
            return batch_x
        batch_y = self.y[index_array]
        return batch_x, batch_y


class DirectoryIterator(Iterator):

    def __init__(self, directory, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 dim_ordering='default',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg',
                 follow_links=False, pool=None, save_list_dir=None,
                 nb_gpu=4, phase="train"):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.dim_ordering = dim_ordering
        if self.color_mode == 'rgb':
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.pool = pool
        self.batch_size = batch_size
        self.nb_gpu = nb_gpu
        self.save_list_dir = save_list_dir
        self.phase = phase

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}

        # first, count the number of samples and classes
        # self.nb_sample = 0
        self.nb_sample_init = 0

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)
        self.nb_class = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        def _recursive_list(subpath):
            return sorted(os.walk(subpath, followlinks=follow_links), key=lambda tpl: tpl[0])

        for subdir in classes:
            subpath = os.path.join(directory, subdir)
            for root, _, files in _recursive_list(subpath):
                for fname in files:
                    is_valid = False
                    for extension in white_list_formats:
                        if fname.lower().endswith('.' + extension):
                            is_valid = True
                            break
                    if is_valid:
                        self.nb_sample_init += 1
        print ("Using this processing code...")
        print('Found %d images belonging to %d classes.' % (self.nb_sample_init, self.nb_class))

        # second, build an index of the images in the different class subfolders
        self.filenames = []
        self.classes = np.zeros((self.nb_sample_init,), dtype='int32')
        i = 0
        for subdir in classes:
            subpath = os.path.join(directory, subdir)
            for root, _, files in _recursive_list(subpath):
                for fname in files:
                    is_valid = False
                    for extension in white_list_formats:
                        if fname.lower().endswith('.' + extension):
                            is_valid = True
                            break
                    if is_valid:
                        self.classes[i] = self.class_indices[subdir]
                        i += 1
                        # add filename relative to directory
                        absolute_path = os.path.join(root, fname)
                        self.filenames.append(os.path.relpath(absolute_path, directory))

        # Save the list of img_paths when doing the test
        print("Current phase:")
        print(self.phase)

        if self.phase == "test":
            files_np = np.array(self.filenames)
            np.save(self.save_list_dir, files_np)
            print("Image paths list saved for testing!")

        # Pop the remainder according to the nb_gpu
        multiple = self.nb_gpu * self.batch_size
        print("The multiple is: %d" % multiple)
        quotient = self.nb_sample_init // multiple
        print("Quotient: %d" % quotient)
        nb_excess_patch = self.nb_sample_init - quotient * multiple
        print("Excess patches: %d" % nb_excess_patch)
        self.nb_sample = self.nb_sample_init - nb_excess_patch

        # Deal with excess patches
        if nb_excess_patch == 0:
            print("There is no excessing patches.")
        else:
            for i in range(nb_excess_patch):
                np.delete(self.classes, -1)
                self.filenames.pop(-1)
        # print("Lenth of the patch meta: %d" % len(self.patch_meta))
        print("[!] After pop the total number of patches is %d" % self.nb_sample)

        super(DirectoryIterator, self).__init__(self.nb_sample, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel

        batch_x = None
        grayscale = self.color_mode == 'grayscale'

        if self.pool:
            pipeline = self.image_data_generator.pipeline()
            result = self.pool.map(process_image_pipeline_dir, ((pipeline,
                self.filenames[j],
                self.directory,
                grayscale,
                self.target_size,
                self.dim_ordering,
                self.rngs[i%self.batch_size]) for i, j in enumerate(index_array)))
            batch_x = np.array(result)
        else:
            batch_x = np.zeros((current_batch_size,) + self.image_shape)
            # build batch of image data
            for i, j in enumerate(index_array):
                fname = self.filenames[j]
                img = load_img(os.path.join(self.directory, fname),
                               grayscale=grayscale,
                               target_size=self.target_size)
                x = img_to_array(img, dim_ordering=self.dim_ordering)
                x = self.image_data_generator.random_transform(x)
                x = self.image_data_generator.standardize(x)
                batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        # if self.save_to_dir:
        #     for i in range(current_batch_size):
        #         img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
        #         fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
        #                                                           index=current_index + i,
        #                                                           hash=np.random.randint(1e4),
        #                                                           format=self.save_format)
        #         img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype('float32')
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.nb_class), dtype='float32')
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y


##############################
# TODO: follow the implementation of DirectoryIterator
##############################
class JsonIterator(Iterator,RandomCrop):

    def __init__(self, json_file_path, image_data_generator,labels,nb_per_class,
                 foreground_rate_per_class,
                 target_size=(256, 256), color_mode='rgb',
                 dim_ordering='default',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg',
                 follow_links=False, pool=None, nb_gpu=4, is_training=True):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.json_file_path = json_file_path
        self.nb_per_class = nb_per_class
        self.labels = labels 
        self.foreground_rate_per_class = foreground_rate_per_class
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.dim_ordering = dim_ordering
        if self.color_mode == 'rgb':
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.pool = pool
        self.batch_size = batch_size
        self.nb_gpu = nb_gpu
        self.is_training = is_training

        # TODO: load pre-processed patches into patch_meta (Finished)
        # self.total = []
        # self.patch_meta_tumor = []
        # self.patch_meta_normal_mucosa=[]
        global logger
        logger = logging.getLogger(__name__)

        enum = threading.enumerate()
        logger.info('============before, num of threads is:===========')
        logger.info(len(enum))

        # set parameters
        # labels = ['tumor', 'normal','hard_normal',"hard_tumor","lymphatic_sinusoid"]
        # labels = ['tumor', 'normal','hard_normal',"hard_tumor"]
        # labels = ['tumor','normal']
        # nb_per_class = [80,48]
        # json_file_path=self.json_file_path
        # nb_per_class = [77, 33, 8, 1, 5]
        # foreground_rate_per_class = [0.1,0.7]
        # foreground_rate_per_class = [0.7, 0.2, 0.001,0.001,0.001]
        objs, nb_samples_per_epoch = Iterator.get_instance_objects(train_base_path=json_file_path,
                                                        labels=labels,
                                                        nb_per_class=nb_per_class,
                                                        foreground_rate_per_class=foreground_rate_per_class,
                                                        eval_base_path='',
                                                        non_zero_rate=0.7,
                                                        is_training=is_training,
                                                        is_shuffle=False,
                                                        crop_width=224,
                                                        crop_height=224,
                                                        readers=1,
                                                        num_threads=8,
                                                        info_maxsize=3000,
                                                        data_maxsize=1000)
        logger.info('*********nb_samples_per_epoch: {}**********' .format(nb_samples_per_epoch))

        self.batch_generator = Iterator.image_data_generator(objs, sum(nb_per_class))

        enum = threading.enumerate()
        logger.info('============before, num of threads is:===========')
        logger.info(len(enum))

        self.stop_training = False

        if self.stop_training == True:
            end_subthreads(objs)

            # count the numbers of threads after task is done
            enum = threading.enumerate()
            print("done")



        super(JsonIterator, self).__init__(batch_size, shuffle, seed)

    #TODO: use this function to generate batch of image data: batch_x and label: batch_y
    def next(self):
        grayscale = self.color_mode == 'grayscale'
        # a=time.time()
        batch_generator = next(self.batch_generator)
        random.shuffle(batch_generator)
        #TODO: implement process_image_pipeline_json, this function takes input of filenames
        if self.pool:
            pipeline = self.image_data_generator.pipeline()
            # map function: use the "process_image_pipeline_json" function to process the second term
            results = self.pool.map(process_image_pipeline_img,
                    ((pipeline,
                    image_labels,  # change 成index_array_neg(j)
                    grayscale,
                    self.target_size,
                    self.dim_ordering,
                    self.rngs[index%(self.batch_size)]) for index, image_labels in enumerate(batch_generator)))
            results_normal_np = np.asarray(results)
            nb_sample = results_normal_np.shape[0]
            #TODO: get the X and Y from results ()
            batch_x = np.asarray(results_normal_np[:, 0])   
            batch_y = np.asarray(results_normal_np[:, 1])
            batch_img_path = np.asarray(results_normal_np[:, 2])

            new_batch_x = []
            new_batch_y = []
            new_batch_img_path = []

            for i in range (nb_sample):
                new_batch_x.append(batch_x[i]) 
                new_batch_y.append(batch_y[i])
                new_batch_img_path.append(batch_img_path[i])

            new_batch_x = np.array(new_batch_x)   #(224,224,3)*56
            new_batch_y = np.array(new_batch_y)
            new_batch_y = np.reshape(new_batch_y, (nb_sample, 1))
            new_batch_img_path = np.reshape(new_batch_img_path, (nb_sample, 1))
            # new_batch_y = keras.utils.to_categorical(new_batch_y,2)
            # print(new_batch_y)

            # print (new_batch_x.shape)
            # print (new_batch_y.shape)
            # print ("Value of y: ")
            # print (batch_y)
        else:
            print ("#######This is the else ...######")
            print ("#######Need to debug!!###########")

        # optionally save augmented images to disk for debugging purposes
        """
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        """
        # build batch of labels
        """
        if self.class_mode == 'sparse':
            new_batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            new_batch_y = self.classes[index_array].astype('float32')
        elif self.class_mode == 'categorical':
            new_batch_y = np.zeros((len(batch_x), self.nb_class), dtype='float32')
            for i, label in enumerate(self.classes[index_array]):
                new_batch_y[i, label] = 1.
        """
        # import pdb
        # pdb.set_trace()
        return new_batch_x, new_batch_y,new_batch_img_path
        # return new_batch_x, new_batch_y