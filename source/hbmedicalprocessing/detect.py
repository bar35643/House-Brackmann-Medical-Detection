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
import os
import sys
from pathlib import Path

import numpy as np
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.config import LOGGER #pylint: disable=import-error
from utils.general import check_requirements, set_logging, init_dict, OptArgs, get_key_from_dict #pylint: disable=import-error
from utils.pytorch_utils import select_device, load_model #pylint: disable=import-error
from utils.templates import house_brackmann_template, house_brackmann_lookup #pylint: disable=import-error
from utils.dataloader import create_dataloader_only_images #pylint: disable=import-error
from utils.automata import hb_automata #pylint: disable=import-error

PREFIX = "detect: "

@torch.no_grad()
def run(weights="models", #pylint: disable=too-many-arguments, too-many-locals, too-many-statements
        source="../data",
        batch_size=16,
        device="cpu",
        half=False,
        function_selector="all",
        convert=False):
    """
    Calculates the Grade or the Modules for the House-Brackmann score

    :param weights: path to Models (str)
    :param source: Data Source Directory (str)
    :param batch_size: max size of the Batch (int)
    :param device: CPU or 0 or 0,1 (int)
    :param half: Half Precsiosn Calculation (bool)
    :param function_selector: Which function should be calculated Example: (str)
                              function_selector=symmetry,eye,mouth,forehead,hb_direct
                              function_selector=symmetry,eye
                              function_selector=forehead
                              function_selector=all
    :param convert: converts Classes to Labels (bool)
    :return Dictionary of Result (Dict)
    """
    LOGGER.info("%sStarting Detection...",PREFIX)


    #Selecting the Moudles
    fn_ptr = []
    function_selector = function_selector.strip().lower().replace(" ", "").split(",")
    LOGGER.debug("%sSelected Functions %s", PREFIX, function_selector)
    for i in function_selector:
        if i == "all":
            fn_ptr = list(house_brackmann_template)
        else:
            assert i in list(house_brackmann_template), "given Function not in the list of the allowed Functions! Only use all, symmetry, eye, mouth or forehead"
            fn_ptr.append(i)
    fn_ptr = list(dict.fromkeys(fn_ptr))
    LOGGER.debug("%sSelected Functions after deleting Duplicates %s", PREFIX, fn_ptr)

    #Init Device
    device = select_device(device)
    half &= device.type != "cpu"  # half precision only supported on CUDA

    #Loading Data
    dataloader= create_dataloader_only_images(path=source, device=device, batch_size=batch_size, prefix_for_log=PREFIX)

#-#-#-#-#-#-#-#-#-#-#-Calculating Operation-#-#-#-#-#-#-#-#-#-#-#-#
    result_list = {}
    for batch, item_struct in enumerate(dataloader):
        #------------------------------BATCH------------------------------#
        i_name, img_struct = item_struct
        results = init_dict(house_brackmann_template, [])
        for selected_function in fn_ptr:
            model = load_model(weights, selected_function) #Load Model
            model.eval()

            if half:
                model.half()  # to FP16

            img = img_struct[selected_function]
            img = (img.half() if half else img.float()) # uint8 to fp16/32

            pred = model(img.to(device)) #Predict image
            results[selected_function] = pred.cpu().numpy()# COnvert Prediction to Numpy Array

        LOGGER.debug("%sMINIBATCH --> Batch-Nr=%s, names=%s, resuts=%s", PREFIX, batch, i_name, results)

        #Calculates the Grade from the seperate Modules
        if function_selector[0] == "all":
            for idx, name in enumerate(i_name):
                tmp = deepcopy(house_brackmann_template)
                for func in results:
                    tmp[func] = results[func][idx]

                #Fusionate the Moduels with Rowsum
                grade1 = tmp["symmetry"][0] + tmp["eye"][0] + tmp["forehead"][0] + tmp["mouth"][0]
                grade2 = tmp["symmetry"][0] + tmp["eye"][0] + tmp["forehead"][0] + tmp["mouth"][1]
                grade3 = tmp["symmetry"][0] + tmp["eye"][0] + tmp["forehead"][1] + tmp["mouth"][1]
                grade4 = tmp["symmetry"][0] + tmp["eye"][1] + tmp["forehead"][2] + tmp["mouth"][2]
                grade5 = tmp["symmetry"][1] + tmp["eye"][1] + tmp["forehead"][2] + tmp["mouth"][2]
                grade6 = tmp["symmetry"][2] + tmp["eye"][1] + tmp["forehead"][2] + tmp["mouth"][3]

                out = [grade1, grade2, grade3, grade4, grade5, grade6]
                LOGGER.debug("%sRowsum from 0-5", PREFIX, out)
                tmp["grade_rowsum"] = out.index(max(out))


                #Fusionate the Moduels with Automata
                tmp["symmetry"] = np.argmax(tmp["symmetry"])
                tmp["eye"]      = np.argmax(tmp["eye"])
                tmp["mouth"]    = np.argmax(tmp["mouth"])
                tmp["forehead"] = np.argmax(tmp["forehead"])
                tmp["grade_automata"] = hb_automata(tmp["symmetry"], tmp["eye"], tmp["mouth"], tmp["forehead"])

                #Direct Module to detect Grade
                tmp["grade_direct"] = np.argmax(tmp["hb_direct"])
                del tmp["hb_direct"]


                # Convert Class to Label
                if convert:
                    tmp["symmetry"]       = get_key_from_dict(house_brackmann_lookup["symmetry"]["enum"] , tmp["symmetry"])
                    tmp["eye"]            = get_key_from_dict(house_brackmann_lookup["eye"]["enum"]      , tmp["eye"])
                    tmp["mouth"]          = get_key_from_dict(house_brackmann_lookup["mouth"]["enum"]    , tmp["mouth"])
                    tmp["forehead"]       = get_key_from_dict(house_brackmann_lookup["forehead"]["enum"] , tmp["forehead"])
                    tmp["grade_rowsum"]   = get_key_from_dict(house_brackmann_lookup["hb_direct"]["enum"], tmp["grade_rowsum"])
                    tmp["grade_direct"]   = get_key_from_dict(house_brackmann_lookup["hb_direct"]["enum"], tmp["grade_direct"])
                    tmp["grade_automata"] = get_key_from_dict(house_brackmann_lookup["hb_direct"]["enum"], tmp["grade_automata"])
                result_list[name] = tmp
                LOGGER.debug("%sResults for %s:", PREFIX, name, result_list[name])
                del tmp
        else:
            for idx, name in enumerate(i_name):
                tmp = deepcopy(house_brackmann_template)
                for func in results:
                    if len(results[func]) != 0: #if one Module is empty

                        tmp[func] = np.argmax(results[func][idx])
                        if convert:
                            tmp[func] = get_key_from_dict(house_brackmann_lookup[func]["enum"] , tmp[func])

                tmp["grade_rowsum"] = None
                tmp["grade_direct"] = tmp["hb_direct"]
                tmp["grade_automata"] = None
                del tmp["hb_direct"]

                result_list[name] = tmp
                LOGGER.debug("%sResults for %s:", PREFIX, name, result_list[name])
                del tmp
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
    parser.add_argument("--function-selector", type=str, default="all",
                        help="funchtions which an be executed or multiple of the list (all or symmetry, eye, mouth, forehead, hb_direct)")
    parser.add_argument("--convert", action="store_true",
                        help="convert the Classes to their Labels")
    parser.add_argument("--log", action="store_true", help="activates log")
    return parser.parse_args()

if __name__ == "__main__":
    opt_args = vars(parse_opt())
    OptArgs.instance()(opt_args)

    set_logging(PREFIX)
    check_requirements()

    time = timeit.timeit(lambda: run(**opt_args), number=1) #pylint: disable=unnecessary-lambda
    LOGGER.info("Done with Detection. Finished in %s s", time)
