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
import argparse
import time
import timeit
from copy import deepcopy

import torch

from utils.config import LOGGER
from utils.general import check_requirements, set_logging, init_dict, OptArgs
from utils.pytorch_utils import select_device, load_model
from utils.dataloader import create_dataloader_only_images
from utils.automata import hb_automata

PREFIX = "detect: "

@torch.no_grad()
def run(weights="models", #pylint: disable=too-many-arguments, too-many-locals
        source="../data",
        batch_size=16,
        device="cpu",
        half=False,):
    """
    Calculates the Grade or the Modules for the House-Brackmann score

    :param weights: path to Models (str)
    :param source: Data Source Directory (str)
    :param batch_size: max size of the Batch (int)
    :param device: CPU or 0 or 0,1 (int)
    :param half: Half Precsiosn Calculation (bool)
    :return Dictionary of Result (Dict)
    """
    LOGGER.info("%sStarting Detection...",PREFIX)

    #Init Device
    device = select_device(device)
    half &= device.type != "cpu"  # half precision only supported on CUDA

    #Loading Data
    dataloader= create_dataloader_only_images(path=source, device=device, batch_size=batch_size, prefix_for_log=PREFIX)

    selected_module = "hb_direct"


    model = load_model(weights, selected_module)
#-#-#-#-#-#-#-#-#-#-#-Calculating Operation-#-#-#-#-#-#-#-#-#-#-#-#
    result_list = {}
    for batch, item_struct in enumerate(dataloader):
        #------------------------------BATCH------------------------------#
        i_name, img = item_struct
        model.eval()
        if half:
            model.half()  # to FP16

        img = (img.half() if half else img.float()) # uint8 to fp16/32
        pred = model(img.to(device))

        LOGGER.debug("%sMINIBATCH --> Batch-Nr=%s, names=%s, resuts=%s", PREFIX, batch, i_name, pred.max(1)[1].cpu().numpy())

        for idx, name in enumerate(i_name):
            result_list[name] = {"grade": pred.max(1)[1].cpu().numpy()}
        #----------------------------END BATCH----------------------------#
#-#-#-#-#-#-#-#-#-#-#End Calculating Operation-#-#-#-#-#-#-#-#-#-#
    LOGGER.info("%sFinal Results ---> %s", PREFIX, result_list)
    return result_list










def parse_opt():
    """
    Command line Parser Options see >> python detect.py -h for more about
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="models",
                        help="model folder")
    parser.add_argument("--source", type=str, default="../test_data",
                        help="file/dir")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="total batch size for all GPUs")
    parser.add_argument("--device", default="cpu",
                        help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--half", action="store_true",
                        help="use FP16 half-precision inference")
    return parser.parse_args()

if __name__ == "__main__":
    opt_args = vars(parse_opt())
    OptArgs.instance()(opt_args)

    set_logging(PREFIX)
    check_requirements()

    time = timeit.timeit(lambda: run(**opt_args), number=1) #pylint: disable=unnecessary-lambda
    LOGGER.info("Done with Detection. Finished in %s s", time)
