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

import torch
import torchvision.transforms as T

from .config import LOGGER
from .cutter import Cutter #pylint: disable=import-error
from .dataloader import transform_resize_and_to_tensor #pylint: disable=import-error
from .singleton import Singleton #pylint: disable=import-error
from .templates import house_brackmann_template #pylint: disable=import-error


@Singleton
class Wrapper():
    """
    Wrapper Class

    Handles the wrapper of the most interesting function of the module
    """
    def __init__(self):
        """
        Initializes Wrapper Class

        :param device: cuda device (cpu or cuda:0)
        :param prefix_for_log: logger output prefix (str)
        """
        self.prefix_for_log = ""
        self.cutter_class = Cutter.instance() #pylint: disable=no-member

    def augmentation(self, img_tensor):
        """
        do Augmentation

        :param img_tensor: Tensor (Tensor)
        :return Transformed Tensor (Tensor)

        Info:
        https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
        """
        valid_transforms = T.Compose([ T.Normalize(mean=[0.5, 0.5, 0.5],
                                                   std= [0.5, 0.5, 0.5])])

        LOGGER.debug("%sAugmentation set to:\n %s", self.prefix_for_log, valid_transforms) #pylint: disable=no-member
        return valid_transforms(img_tensor)

    def get_img_from_module(self, path, module):
        """
        Get all9 images of one Patient

        :param path: Path to Patient including all 9 images (str)
        :param module_list: module for operation, one or multipe of ["symmetry", "eye", "mouth", "forehead", "hb_direct"] (list of str)
        :returns list of Images (list of type Image)
        """
        if module not in list(house_brackmann_template):
            return []

        func_list = self.cutter_class.cut_wrapper()

        #Documentation for Framework: https://github.com/1adrianb/face-alignment
        return [transform_resize_and_to_tensor(func_list[module](path, "01"), module  ),
                transform_resize_and_to_tensor(func_list[module](path, "02"), module  ),
                transform_resize_and_to_tensor(func_list[module](path, "03"), module  ),
                transform_resize_and_to_tensor(func_list[module](path, "04"), module  ),
                transform_resize_and_to_tensor(func_list[module](path, "05"), module  ),
                transform_resize_and_to_tensor(func_list[module](path, "06"), module  ),
                transform_resize_and_to_tensor(func_list[module](path, "07"), module  ),
                transform_resize_and_to_tensor(func_list[module](path, "08"), module  ),
                transform_resize_and_to_tensor(func_list[module](path, "09"), module  )]

    def get_cat_tensor_from_module(self, path, module):
        """
        Get Concatenated Tensor from all 9 images of one Patient

        :param path: Path to Patient including all 9 images (str)
        :param module_list: module for operation, one or multipe of ["symmetry", "eye", "mouth", "forehead", "hb_direct"] (list of str)
        :returns  Tensor of size [27, a, b] (Tensor)
        """

        img_list = self.get_img_from_module(path, module)
        if not img_list:
            return None

        #Concatenates all 9 images to one Tensor
        return torch.cat(  [self.augmentation(j) for j in img_list]  )

    def get_cut_images_from_one(self, path, img_number, module_list):
        """
        Get cut images from the Patient

        :param path: Path to Patient including all 9 images (str)
        :param img_number one of ["01", "02", "03", "04", "05", "06", "07", "08", "09"] (str)
        :param module_list: module for operation, one or multipe of ["symmetry", "eye", "mouth", "forehead", "hb_direct"] (list of str)
        :returns  image list (Image)
        """
        for i in module_list:
            if i not in ["symmetry", "eye", "mouth", "forehead", "hb_direct"]:
                return None

        if img_number not in ["01", "02", "03", "04", "05", "06", "07", "08", "09"]:
            return None

        func_list = self.cutter_class.cut_wrapper()

        #Concatenates all 9 images to one Tensor
        return [func_list[i](path, img_number) for i in module_list]
