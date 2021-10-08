"""
TODO
Check internet connectivity
"""

import argparse
import sys
import logging
import time
import timeit
from copy import deepcopy
from pathlib import Path
import numpy

import torch
from torch.utils.data.dataloader import DataLoader


#time tracking with timeit.timeit(lambda: func, number=1)
#or
# start_time = time.time()
# fn()
# elapsed = time.time() - start_time

from utils.general import check_requirements, increment_path, set_logging
from utils.pytorch_utils import select_device
from utils.settings import allowed_fn, house_brackmann_lookup, house_brackmann_template
from utils.dataloader import LoadImages

LOGGER = logging.getLogger(__name__)
FILE = Path(__file__).resolve()
sys.path.append(FILE.parents[0].as_posix())

def detect(weights="models/model.pt",  # model.pt path(s)
           source="data/images",  # file/dir
           imgsz=640,  # inference size (pixels)
           device="cpu",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
           project="../results/detect",  # save results to project/name
           name="run",  # save results to project/name
           half=False,  # use FP16 half-precision
           function_selector="all"): #pylint: disable=too-many-arguments
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

    # Directories
    #TODO
    #save_dir = increment_path(Path(project) / name, exist_ok=False)  # increment run
    #save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    #Init
    device = select_device(device)
    half &= device.type != "cpu"  # half precision only supported on CUDA



    dataset = LoadImages(source, img_size=imgsz)
    dataset = DataLoader(dataset, batch_size=1, shuffle=False)



    #Calculating
    result_list = []
    for img in dataset:
        assert weights.endswith('.pt'), "File has wrong ending"
        assert Path(weights).exists(), "File does not exists"
        #TODO checkpoint = torch.load(weights)

        results = deepcopy(house_brackmann_template)
        for selected_function in fn_ptr:
            model=house_brackmann_lookup[selected_function]["model"]
            #TODO model.load_state_dict(checkpoint[selected_function]).to(device)
            img0 = img[selected_function]
            if half:
                model.half()  # to FP16

            img0 = (img0.half() if half else img0.float()) /255.0  # uint8 to fp16/32   0 - 255 to 0.0 - 1.0
            if len(img0.shape) == 3:
                img0 = img0[None]  # expand for batch dim


            pred = model(img0.to(device))

            #TODO Lookup Prediction
            print(pred.max(1))
            print(pred.max(1)[1])
            print(numpy.argmax(pred.detach().numpy()))

            #prediction = house_brackmann_lookup[selected_function]["lookup"][numpy.argmax(pred.detach().numpy())]

            #results.append(house_brackmann_lookup[selected_function]["lookup"][pred])
            results[selected_function] = pred.max(1)[1]

        print(results)

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
    parser.add_argument("--weights", nargs="+", type=str, default="models/model.pt", help="model path(s)")
    parser.add_argument("--source", type=str, default="data/images", help="file/dir")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--device", default="cpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--project", default="../results/detect", help="save results to project/name")
    parser.add_argument("--name", default="run", help="save results to project/name")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--function_selector", type=str, default="all", help="funchtions which an be executed or multiple of the list (all, symmetry, eye, mouth, forehead)")
    return parser.parse_args()

if __name__ == "__main__":
    #pylint: disable=pointless-string-statement
    """
    TODO
    Check internet connectivity
    """
    opt_args = parse_opt()
    set_logging(logging.DEBUG, "detect: ",opt_args)
    check_requirements()
    time = timeit.timeit(lambda: detect(**vars(opt_args)), number=1) #pylint: disable=unnecessary-lambda
    LOGGER.info("Done with Detection. Finished in %s s", time)