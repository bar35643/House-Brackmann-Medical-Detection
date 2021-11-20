#TODO Docstring
"""
TODO
"""

#import os
import argparse
import logging
import os
import time
import timeit
from pathlib import Path

import torch


#time tracking with timeit.timeit(lambda: func, number=1)
#or
# start_time = time.time()
# fn()
# elapsed = time.time() - start_time
from utils.config import LOGGER
from utils.general import check_requirements, set_logging, init_dict
from utils.pytorch_utils import select_device
from utils.templates import allowed_fn, house_brackmann_lookup, house_brackmann_template
from utils.dataloader import create_dataloader_only_images

PREFIX = "detect: "
LOGGING_STATE = logging.INFO #logging.DEBUG

@torch.no_grad()
def run(weights="models", #pylint: disable=too-many-arguments, too-many-locals
        source="../data",
        imgsz=640,
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

    dataloader= create_dataloader_only_images(path=source, imgsz=imgsz, device=device, batch_size=batch_size, prefix_for_log=PREFIX)
    #Calculating
    result_list = []
    for batch, item_struct in enumerate(dataloader):
        i_name, img_struct = item_struct

        results = init_dict(house_brackmann_template, [])
        for selected_function in fn_ptr:
            model_weights = os.path.join(Path(weights), selected_function+".pt")
            #assert model_weights.endswith('.pt'), f"File {model_weights} has wrong ending"
            assert Path(model_weights).exists(), f"Model-File {model_weights} does not exists"

            model=house_brackmann_lookup[selected_function]["model"].to(device)
            checkpoint = torch.load(model_weights)
            model.load_state_dict(checkpoint["model"])
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
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640],
                        help="inference size h,w")
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
    opt_args = parse_opt()
    set_logging(LOGGING_STATE, PREFIX,opt_args)
    check_requirements()
    time = timeit.timeit(lambda: run(**vars(opt_args)), number=1) #pylint: disable=unnecessary-lambda
    LOGGER.info("Done with Detection. Finished in %s s", time)
