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

import os
from pathlib import Path
import logging


#General config Parameters for the Project
#CHANGEABLE
LRU_MAX_SIZE=100
MAX_THREADS=8

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo', 'heic']
WORKDIR = Path('./static/upload_temporary/')


#General config Parameters for the Project
#NOT CHANGEABLE
FILE = Path(__file__).resolve()

THREADPOOL_NUM_THREADS = min(MAX_THREADS, os.cpu_count())  # number of multiprocessing threads

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "-1"))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", "-1"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))

LOGGER = logging.getLogger(__name__)
