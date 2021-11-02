#TODO Docstring
"""
TODO
"""

#import os
import argparse
import logging
import time
import timeit
from copy import deepcopy
#import numpy

import torch
from torch.utils.data.dataloader import DataLoader


#time tracking with timeit.timeit(lambda: func, number=1)
#or
# start_time = time.time()
# fn()
# elapsed = time.time() - start_time
from utils.config import LOGGER
from utils.general import check_requirements, set_logging
from utils.pytorch_utils import select_device
from utils.templates import allowed_fn, house_brackmann_lookup, house_brackmann_template
from utils.dataloader import LoadImages

PREFIX = "detect: "
LOGGING_STATE = logging.INFO #logging.DEBUG

@torch.no_grad()
def run(weights="models/model.pt", #pylint: disable=too-many-arguments, too-many-locals
        source="../data",
        imgsz=640,
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

    dataset = LoadImages(path=source, imgsz=imgsz, device=device, prefix_for_log=PREFIX)

    assert dataset, "No data in dataset given!"
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    #Calculating
    result_list = []
    for i_name, img_list, img_inv_list in dataloader:
        #TODO enable assert Path(weights).exists(), "File does not exists"
        assert weights.endswith('.pt'), "File has wrong ending"
        #TODO checkpoint = torch.load(weights)

        results = deepcopy(house_brackmann_template)
        results["symmetry"] = []
        results["eye"] = []
        results["mouth"] = []
        results["forehead"] = []
        for selected_function in fn_ptr:
            model=house_brackmann_lookup[selected_function]["model"]
            #TODO model.load_state_dict(checkpoint[selected_function]).to(device)
            if half:
                model.half()  # to FP16

            for img, img_inv in zip(img_list[selected_function], img_inv_list[selected_function]):
                img = (img.half() if half else img.float()) /255.0  # uint8 to fp16/32   0 - 255 to 0.0 - 1.0
                img = img[None] if len(img.shape) == 3 else img

                img_inv = (img_inv.half() if half else img_inv.float()) /255.0  # uint8 to fp16/32   0 - 255 to 0.0 - 1.0
                img_inv = img_inv[None] if len(img_inv.shape) == 3 else img_inv

                    #TODO mean of both prediction
                pred = model(img.to(device))
                pred_inv = model(img_inv.to(device))

                #TODO Lookup Prediction
                #print(pred.max(1))
                #print(pred.max(1)[1])
                #print(numpy.argmax(pred.detach().numpy()))

                #prediction = house_brackmann_lookup[selected_function]["lookup"][numpy.argmax(pred.detach().numpy())]

                #results.append(house_brackmann_lookup[selected_function]["lookup"][pred])
                results[selected_function].append(pred.max(1)[1])

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
    parser.add_argument("--weights", nargs="+", type=str, default="models/model.pt",
                        help="model path(s)")
    parser.add_argument("--source", type=str, default="../test_data",
                        help="file/dir")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640],
                        help="inference size h,w")
    parser.add_argument("--device", default="cpu",
                        help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--half", action="store_true",
                        help="use FP16 half-precision inference")
    parser.add_argument("--function_selector", type=str, default="all",
                        help="funchtions which an be executed or multiple of the list (all, symmetry, eye, mouth, forehead)")
    return parser.parse_args()

if __name__ == "__main__":
    opt_args = parse_opt()
    set_logging(LOGGING_STATE, PREFIX,opt_args)
    check_requirements()
    time = timeit.timeit(lambda: run(**vars(opt_args)), number=1) #pylint: disable=unnecessary-lambda
    LOGGER.info("Done with Detection. Finished in %s s", time)
