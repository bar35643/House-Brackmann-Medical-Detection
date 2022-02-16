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
"""
from torch import nn
from torchvision.models import resnet18


#https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html


#Modyfying the Models for the Modules
#first layer hs to be istead of 3, 27 because of concartenation
#last layer has to be the size of the len(label)
special = resnet18(pretrained=True)
special.conv1 = nn.Conv2d(3, special.conv1.out_channels,
                            kernel_size=special.conv1.kernel_size,
                            stride=special.conv1.stride,
                            padding=special.conv1.padding,
                            bias=special.conv1.bias)
special.fc = nn.Linear(special.fc.in_features, 6)


#Lookuptable for the modules
#Enum reprenentates the correlation between nabe and a Number
#model is the representaing Neural Net
house_brackmann_lookup = {
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

house_brackmann_template = {
"hb_direct" : None,
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
