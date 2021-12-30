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
import logging
import time
import timeit

import torch


#time tracking with timeit.timeit(lambda: func, number=1)
#or
# start_time = time.time()
# fn()
# elapsed = time.time() - start_time
from utils.config import LOGGER
from utils.general import check_requirements, set_logging, init_dict, OptArgs
from utils.pytorch_utils import select_device, load_model
from utils.templates import allowed_fn, house_brackmann_template
from utils.dataloader import create_dataloader_only_images

PREFIX = "detect: "

@torch.no_grad()
def run(weights="models", #pylint: disable=too-many-arguments, too-many-locals
        source="../data",
        batch_size=16,
        device="cpu",
        half=False,
        function_selector="all"):
    """
    TODO
    Check internet connectivity
    """


    #Selecting the Functions
    fn_ptr = []
    function_selector = function_selector.strip().lower().replace(" ", "").split(",")
    for i in function_selector:
        if i == "all":
            fn_ptr = allowed_fn
        else:
            assert i in allowed_fn, "given Function not in the list of the allowed Functions! Only use all, symmetry, eye, mouth or forehead"
            fn_ptr.append(i)

    #Init
    device = select_device(device)
    half &= device.type != "cpu"  # half precision only supported on CUDA

    dataloader= create_dataloader_only_images(path=source, device=device, batch_size=batch_size, prefix_for_log=PREFIX)
    #Calculating
    result_list = []
    for batch, item_struct in enumerate(dataloader):
        i_name, img_struct = item_struct

        results = init_dict(house_brackmann_template, [])
        for selected_function in fn_ptr:
            model = load_model(weights, selected_function)
            if half:
                model.half()  # to FP16
            for idx, img in enumerate(img_struct[selected_function]):

                img = (img.half() if half else img.float()) # uint8 to fp16/32

                #TODO mean of both prediction and lookup
                pred = model(img.to(device))
                #print(pred.shape)
                # pred_true = []
                # for j in torch.tensor_split(pred, len(i_name)):
                #     pred_true.append(np.argmax(j.detach().numpy()))

                results[selected_function].append({"batch": str(batch),
                                                   "idx": str(idx),
                                                   "pred": pred.shape})

                    #predicted.append(np.argmax(pred.detach().numpy()))
                    #
                    # pred_inv = model(img_inv.to(device))
                    # predicted.append(np.argmax(pred_inv.detach().numpy()))


                    #print(pred.max(1))
                    #print(pred.max(1)[1])
                    #print(np.argmax(pred.detach().numpy()))


        print(i_name, results)

        if function_selector == "all":
        #TODO Desicion Tree
            pass



    return result_list
















def parse_opt():
    """
    TODO
    Check internet connectivity
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
    parser.add_argument("--function_selector", type=str, default="all",
                        help="funchtions which an be executed or multiple of the list (all, symmetry, eye, mouth, forehead)")
    return parser.parse_args()

if __name__ == "__main__":
    opt_args = vars(parse_opt())
    OptArgs.instance()(opt_args)

    set_logging(PREFIX)
    check_requirements()

    time = timeit.timeit(lambda: run(**opt_args), number=1) #pylint: disable=unnecessary-lambda
    LOGGER.info("Done with Detection. Finished in %s s", time)
