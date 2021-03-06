"""
# Copyright (c) 2021-2022 Raphael Baumann and Ostbayerische Technische Hochschule Regensburg.
#
# This file is part of house-brackmann-medical-processing
# Author: Raphael Baumann
#
# License:
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# Changelog:
# - 2021-12-15 Initial (~Raphael Baumann)
# - 2022-03-12 Final Version 1.0.0 (~Raphael Baumann)
"""

import sys
import os
import re
from copy import deepcopy
import numpy as np
from PIL import Image, ImageOps
#import matplotlib.pyplot as plt

import face_alignment

from .config import LOGGER, IMG_FORMATS  #pylint: disable=import-error
from .templates import house_brackmann_template #pylint: disable=import-error
from .singleton import Singleton #pylint: disable=import-error
from .decorators import try_except_none #pylint: disable=import-error

if sys.platform == 'win32': #pylint: disable=import-error #pyheif does not work on Windows. So dummy is import
    from .errorimports import read_heif #pylint: disable=import-error
if sys.platform == 'linux':
    from pyheif import read_heif #pylint: disable=import-error

@try_except_none
def load_image(path, img_name):
    """
    Loading images in correct Format

    :param path: Path to folder (str)
    :param img_name: Name of the Image (str)
    :returns  Image (Image)
    """
    #TODO add Other Timestamps after Preop T000 for example T001,T002, T003 ... as seperate
    path_list = os.listdir(path)
    matching_folders = [os.path.join(path, s) for s in path_list if ("T000" in s) and ("postop" not in s)]
    LOGGER.debug("Matching Folder list -> %s", matching_folders)

    if not matching_folders:
        matching_folders = [path]

    LOGGER.debug("Matching Folder list after checking if empty -> %s", matching_folders)

    matching_img_path = []
    for i in matching_folders:
        matching_img_path += [os.path.join(i, s) for s in os.listdir(i)]
    LOGGER.debug("Matching Imagepath  -> %s", matching_img_path)

    matching_img_path_format = [x for x in matching_img_path if x.split('.')[-1].lower() in IMG_FORMATS and not "IMG" in x]
    image   = [x for x in matching_img_path_format if img_name in re.split(r'\\|/',x)[-1]]
    image.sort()
    LOGGER.debug("Matching Imagepath after checking Formats and sorting  -> %s", image)


    #print(img_name, path, matching_folders, image, )

    if not image:
        LOGGER.debug("No image found return None!")
        return None

    #loading images
    #using the first image in the array
    if image[0].split('.')[-1].lower() in ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']:
        return Image.open(image[0]).convert('RGB')

    if image[0].split('.')[-1].lower() in ['heic']:
        temp = read_heif(image[0]) #pyheif read
        return Image.frombytes(mode=temp.mode, size=temp.size, data=temp.data).convert('RGB')

    return None #Faisave

@Singleton
class Cutter():
    """
    Cutter Class

    Handles the cutting for the images into the different types
    """
    def __init__(self):
        """
        Initializes Cutter Class and Face Alignment module

        :param device: cuda device (cpu or cuda:0)
        :param prefix_for_log: logger output prefix (str)
        """
        self.prefix_for_log = ""
        self.fn_landmarks = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                                         flip_input=False,
                                                         device='cpu',
                                                         face_detector='sfd',
                                                         network_size=4)

    def set(self, device, prefix_for_log:str):
        """
        set Class items
        :param device: cuda device (cpu or cuda)
        :param prefix_for_log: logger output prefix (str)
        """
        self.prefix_for_log = prefix_for_log
        device = 'cpu' if str(device) == "cpu" else 'cuda'

        #Documentation for Framework: https://github.com/1adrianb/face-alignment
        #pylint: disable=protected-access
        LOGGER.debug("%sSetting up Framework for generating the Markers", self.prefix_for_log)
        self.fn_landmarks = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                                         flip_input=False,
                                                         device=device,
                                                         face_detector='sfd',
                                                         network_size=4)

    def generate_marker(self, img):
        """
        Generates the Landmarks from given input (Points [x, y])

        :param img: input image (Image)
        :returns  Array of Landmarks ([x, y], x and y from type int)
        """
        #Documentation for Framework: https://github.com/1adrianb/face-alignment
        assert self.fn_landmarks, "Framework is not defined! Use Cutter.instanche().set(<properties>) to set the Values!"
        landmarks = self.fn_landmarks.get_landmarks(np.array(img))
        return landmarks[0]

    def flip_image_and_return_landmarks(self, img_input, dyn_factor):
        """
        Flip images to correct Rotation

        :param image: input Image (Image)
        :param dyn_factor: Factor for resize (int)
        :returns  landmarks and cropped image (array, Image)
        """
         #try to flip image if exif tag is available see https://pillow.readthedocs.io/en/stable/reference/ImageOps.html
        img_input = ImageOps.exif_transpose(img_input)
        img_resized = img_input.resize((int(img_input.size[0]/dyn_factor), int(img_input.size[1]/dyn_factor)))
        #Generate Landmarks
        det = self.generate_marker(img_resized)

        #only executes if rotation with exif tags did not work or rotation is wrong
        #checks rotation from two marker in face and their relative position
        exit_condition = 0
        while not ((det[0, 0] < det[10, 0]) and (det[0, 1] < det[10, 1])):
            if(det[0, 0] < det[10, 0]) and (det[0, 1] > det[10, 1]): #image is turned 90 Degree counterclockwise
                img_input = img_input.transpose(Image.ROTATE_270)
            if(det[0, 0] > det[10, 0]) and (det[0, 1] < det[10, 1]): #image is turned 90 Degree clockwise
                img_input = img_input.transpose(Image.ROTATE_90)
            if(det[0, 0] > det[10, 0]) and (det[0, 1] > det[10, 1]): #image is turned 180 Degree
                img_input = img_input.transpose(Image.ROTATE_180)

            img_resized = img_input.resize((int(img_input.size[0]/dyn_factor), int(img_input.size[1]/dyn_factor)))
            det = self.generate_marker(img_resized)

            exit_condition += 1
            assert exit_condition!=10, "Can not turn Images automatically!!"

        #returns the originam image only rotated and the landmarks (det)
        return det, img_input

    def crop_image(self, img):
        """
        Loading Images

        :param img: Image (Image)
        :returns  landmarks and cropped image (array, Image)
        """
        #Setting the factor for resizing the image
        #Marker generation does not work for high resolution Images!
        dyn_factor = max(int(img.size[0]/1000), int(img.size[1]/1000), 1)
        dyn_factor = dyn_factor+1 if dyn_factor%2 else dyn_factor

        #landmarks and the image is not affected from the factor!
        det, img_flip_org = self.flip_image_and_return_landmarks(img, dyn_factor)
        assert len(det), "Marker Detection Failture"

        #Setting the minimal and maximal x,y value
        x_min_det = det[:,0].min()*dyn_factor
        x_max_det = det[:,0].max()*dyn_factor
        y_min_det = det[:,1].min()*dyn_factor
        y_max_det = det[:,1].max()*dyn_factor

        #calculating max width, height of the face
        x_diff = abs(x_max_det-x_min_det)
        y_diff = abs(y_max_det-y_min_det)

        #Offsetting the Frame. Leaving space between the border and the Face
        x_min = int(x_min_det - (x_diff/8)) if int(x_min_det - (x_diff/8)) > 0 else 0
        x_max = int(x_max_det + (x_diff/8)) if int(x_max_det + (x_diff/8)) < img_flip_org.size[0] else img_flip_org.size[0]
        y_min = int(y_min_det - (y_diff/2)) if int(y_min_det - (y_diff/2)) > 0 else 0
        y_max = int(y_max_det + (y_diff/4)) if int(y_max_det + (y_diff/4)) < img_flip_org.size[1] else img_flip_org.size[1]

        # Cropping the image. Cut of all unneccessary
        img_crop = img_flip_org.crop((x_min,y_min,x_max,y_max))
        #Correcting the landmarks to the new position
        det = det*dyn_factor - [x_min, y_min]

        return det, img_crop

    def cut_wrapper(self):
        """
        Function Wrapper

        :returns  Dictionary with the Functions (dict))
        """

        struct_func_list = deepcopy(house_brackmann_template)
        struct_func_list["symmetry"] = self.cut_symmetry
        struct_func_list["eye"] = self.cut_eye
        struct_func_list["mouth"] = self.cut_mouth
        struct_func_list["forehead"] = self.cut_forehead
        struct_func_list["hb_direct"] = self.cut_symmetry

        return  struct_func_list

    def cut_symmetry(self, path, img_name):
        """
        Cutter Module for the Symmetry. Cropping the input image to the Specs.

        :param path: input path (str)
        :param img_name: name of the image (str)
        :returns  cropped image
        """
        img = load_image(path, img_name)
        if not img:
            return None
        _, img = self.crop_image(img)
        # plt.imshow(img)
        # plt.scatter(landmarks[:,0], landmarks[:,1],10, color=[1, 0, 0, 1])
        # plt.show()
        return img

    def cut_eye(self, path, img_name):
        """
        Cutter Module for the Eye. Cropping the input image to the Specs.

        :param path: input path (str)
        :param img_name: name of the image (str)
        :returns  cropped image
        """
        img = load_image(path, img_name)
        if not img:
            return None
        landmarks, img = self.crop_image(img)


        eye = landmarks[slice(36, 42)]
        fac = (landmarks[:,0].min())/4

        x_min_eye = int(eye[:,0].min() -fac)
        x_max_eye = int(eye[:,0].max() +fac)
        y_min_eye = int(eye[:,1].min() -fac)
        y_max_eye = int(eye[:,1].max() +fac)

        img_slice_left = img.crop((x_min_eye,y_min_eye,x_max_eye,y_max_eye))

        eye = landmarks[slice(42, 48)]
        x_min_eye = int(eye[:,0].min() -fac)
        x_max_eye = int(eye[:,0].max() +fac)
        y_min_eye = int(eye[:,1].min() -fac)
        y_max_eye = int(eye[:,1].max() +fac)

        img_slice_right = img.crop((x_min_eye,y_min_eye,x_max_eye,y_max_eye))


        width = img_slice_right.width   if img_slice_right.width  >= img_slice_left.width  else img_slice_left.width
        height = img_slice_right.height if img_slice_right.height >= img_slice_left.height else img_slice_left.height

        img_slice_right = img_slice_right.resize((width,height))
        img_slice_left  =  img_slice_left.resize((width,height))

        dst = Image.new('RGB', (img_slice_left.width + img_slice_right.width, img_slice_left.height))
        dst.paste(img_slice_left, (0, 0))
        dst.paste(img_slice_right, (img_slice_left.width, 0))

        # plt.imshow(dst)
        # plt.show()

        return dst

    def cut_mouth(self, path, img_name):
        """
        Cutter Module for the Mouth. Cropping the input image to the Specs.

        :param path: input path (str)
        :param img_name: name of the image (str)
        :returns  cropped image
        """
        img = load_image(path, img_name)
        if not img:
            return None
        landmarks, img = self.crop_image(img)


        landmarks = landmarks[slice(48, 68)]
        fac = (landmarks[:,0].min())/4

        x_min_mouth = int(landmarks[:,0].min() -fac)
        x_max_mouth = int(landmarks[:,0].max() +fac)
        y_min_mouth = int(landmarks[:,1].min() -fac)
        y_max_mouth = int(landmarks[:,1].max() +fac)

        img_slice = img.crop((x_min_mouth,y_min_mouth,x_max_mouth,y_max_mouth))
        # plt.imshow(img_slice)
        # plt.scatter(landmarks[:,0]-x_min_mouth, landmarks[:,1]-y_min_mouth,10, color=[1, 0, 0, 1])
        # plt.show()
        return img_slice

    def cut_forehead(self, path, img_name):
        """
        Cutter Module for the Forehead. Cropping the input image to the Specs.

        :param path: input path (str)
        :param img_name: name of the image (str)
        :returns  cropped image
        """
        img = load_image(path, img_name)
        if not img:
            return None
        landmarks, img = self.crop_image(img)


        landmarks = landmarks[slice(17, 27)]
        fac = (landmarks[:,0].min())/16

        x_min_forehead = 0
        x_max_forehead = img.size[0]
        y_min_forehead = 0
        y_max_forehead = int(landmarks[:,1].max()+fac)

        img_slice = img.crop((x_min_forehead,y_min_forehead,x_max_forehead,y_max_forehead))
        # plt.imshow(img_slice)
        # plt.scatter(landmarks[:,0]-x_min_forehead, landmarks[:,1]-y_min_forehead,10, color=[1, 0, 0, 1])
        # plt.show()
        return img_slice
