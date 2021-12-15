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
#python env_test.py
#python -m torch.distributed.run --nproc_per_node 2 env_test.py

#pylint: skip-file
import logging
from argparse import Namespace
import torch
import io
import numpy as np
import torchvision.transforms as T
import timeit

from utils.config import ROOT, ROOT_RELATIVE, LOCAL_RANK, RANK, WORLD_SIZE, LOGGER

from utils.general import set_logging, OptArgs, check_requirements
from utils.pytorch_utils import select_device, is_process_group, is_master_process
from utils.dataloader import CreateDataset, LoadImages
from utils.templates import allowed_fn
from train import run


if __name__ == "__main__":
    opt_args = vars(Namespace())
    OptArgs.instance()(opt_args)
    check_requirements()
    set_logging(logging.INFO, "env_test: ")
    print("-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-")
    print("ROOT: ", ROOT)
    print("ROOT_RELATIVE: ", ROOT_RELATIVE)
    print("LOCAL_RANK: ", LOCAL_RANK)
    print("RANK: ", RANK)
    print("WORLD_SIZE: ", WORLD_SIZE)

    print("Cuda Avialable: ", torch.cuda.is_available())
    print("Cuda device count: ", torch.cuda.device_count())
    print("Selected device 1: ", torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    if is_process_group(LOCAL_RANK):
        print("Selected device 2", torch.device("cuda" if torch.cuda.is_available() else "cpu", LOCAL_RANK))
    print("Distributed is available: ", torch.distributed.is_available())

    if torch.cuda.is_available():
        print("Device Properties: ", torch.cuda.get_device_properties("cuda:0"))


    print(select_device('cpu'))
    #print(select_device('0, 1, 2'))

    print(is_process_group(LOCAL_RANK))
    print(is_master_process(LOCAL_RANK))

    print("\n\ntesting LoadImages (All Categories, Single Category, Single Patient)")
    tst = LoadImages(path='../test_data', prefix_for_log='')
    print("length: ", len(tst))
    tst = LoadImages(path='../test_data/Muskeltransplantation', prefix_for_log='')
    print("length: ", len(tst))
    tst = LoadImages(path='../test_data/Muskeltransplantation/0001', prefix_for_log='')
    print("length: ", len(tst))

    print("\n\nTestin Caching and Lru Cache\n")
    """
    for x in range(1, 300):
        print(f"cached {x}x (lru_cache):"    , timeit.timeit(lambda: run(source="../test_data",cache=True,nosave=False,batch_size=4,device="cpu"), number=x))
        print(f"not cached {x}x (lru_cache):", timeit.timeit(lambda: run(source="../test_data",cache=False,nosave=False,batch_size=4,device="cpu"), number=x))
    """
    print("-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-")
