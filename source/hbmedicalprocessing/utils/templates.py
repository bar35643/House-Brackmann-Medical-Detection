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

from torch import nn
from torchvision.models import resnet18


#https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html


#Modyfying the Models for the Modules
#first layer hs to be istead of 3, 27 because of concartenation
#last layer has to be the size of the len(label)
model_2_label = resnet18(pretrained=True)
model_2_label.conv1 = nn.Conv2d(27, model_2_label.conv1.out_channels,
                            kernel_size=model_2_label.conv1.kernel_size,
                            stride=model_2_label.conv1.stride,
                            padding=model_2_label.conv1.padding,
                            bias=model_2_label.conv1.bias)
model_2_label.fc = nn.Linear(model_2_label.fc.in_features, 2) #2 classes

model_3_label = resnet18(pretrained=True)
model_3_label.conv1 = nn.Conv2d(27, model_3_label.conv1.out_channels,
                            kernel_size=model_3_label.conv1.kernel_size,
                            stride=model_3_label.conv1.stride,
                            padding=model_3_label.conv1.padding,
                            bias=model_3_label.conv1.bias)
model_3_label.fc = nn.Linear(model_3_label.fc.in_features, 3) #3 classes

model_4_label = resnet18(pretrained=True)
model_4_label.conv1 = nn.Conv2d(27, model_4_label.conv1.out_channels,
                            kernel_size=model_4_label.conv1.kernel_size,
                            stride=model_4_label.conv1.stride,
                            padding=model_4_label.conv1.padding,
                            bias=model_4_label.conv1.bias)
model_4_label.fc = nn.Linear(model_4_label.fc.in_features, 4) #4 classes

special = resnet18(pretrained=True)
special.conv1 = nn.Conv2d(27, special.conv1.out_channels,
                            kernel_size=special.conv1.kernel_size,
                            stride=special.conv1.stride,
                            padding=special.conv1.padding,
                            bias=special.conv1.bias)
special.fc = nn.Linear(special.fc.in_features, 6) #6 classes



#Lookuptable for the modules
#Enum reprenentates the correlation between label and a Number
#model is the representaing Neural Net
house_brackmann_lookup = {
    "symmetry": {
        "enum":{
            "normal":0,
            "asymm" :1,
            "none"  :2},
        "model": model_3_label
    },
    "eye": {
        "enum":{
            "complete"  :0,
            "incomplete":1},
        "model": model_2_label
    },
    "mouth": {
        "enum":{
            "normal"   :0,
            "min_asymm":1,
            "asymm"    :2,
            "none"     :3},
        "model":model_4_label
    },
    "forehead": {
        "enum":{
            "normal"   :0,
            "min_asymm":1,
            "none"     :2},
        "model": model_3_label
    },
    "hb_direct": {
        "enum":{
            "I"  :0,
            "II" :1,
            "III":2,
            "IV" :3,
            "V"  :4,
            "VI" :5,},
        "model": special
    },

}

#Template used for everything else.
# e.g copying results
# access via tmp=deepcopy(house_brackmann_template) for no overwriting
house_brackmann_template = {
    "symmetry": None,
    "eye": None,
    "mouth": None,
    "forehead": None,
    "hb_direct": None,
}

#Relation between Grade and the Modules/Labels
house_brackmann_grading = {
    "I"    : {"symmetry":"normal",   "eye":"complete"  ,   "forehead":"normal"   ,   "mouth":"normal"   },
    "II"   : {"symmetry":"normal",   "eye":"complete"  ,   "forehead":"normal"   ,   "mouth":"min_asymm"},
    "III"  : {"symmetry":"normal",   "eye":"complete"  ,   "forehead":"min_asymm",   "mouth":"min_asymm"},
    "IV"   : {"symmetry":"normal",   "eye":"incomplete",   "forehead":"none"     ,   "mouth":"asymm"    },
    "V"    : {"symmetry":"asymm" ,   "eye":"incomplete",   "forehead":"none"     ,   "mouth":"asymm"    },
    "VI"   : {"symmetry":"none"  ,   "eye":"incomplete",   "forehead":"none"     ,   "mouth":"none"     },
}
