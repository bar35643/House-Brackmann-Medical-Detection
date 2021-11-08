#TODO Docstring
"""
TODO
"""


import argparse
import logging
import timeit
from pathlib import Path

import torch

from utils.config import ROOT, ROOT_RELATIVE, LOCAL_RANK, RANK, WORLD_SIZE, LOGGER
from utils.general import check_requirements, increment_path, set_logging
from utils.pytorch_utils import select_device, select_data_parallel_mode, OptimizerClass, SchedulerClass, is_master_process, is_process_group
from utils.dataloader import create_dataloader
from utils.common import training_epochs
from utils.templates import allowed_fn, house_brackmann_lookup

PREFIX = "train: "
LOGGING_STATE = logging.INFO


#https://pytorch.org/tutorials/beginner/saving_loading_models.html



def run(weights="model/model.pt", #pylint: disable=too-many-arguments, too-many-locals
        source="../data",
        imgsz=640,
        cache=False,
        batch_size=16,
        # workers=8,
        device="cpu",
        optimizer="SGD",
        nosave=False,
        project="../results/train",
        name="run",
        epochs=100):
    """
    TODO
    Check internet connectivity
    """

    assert epochs, "Numper of Epochs is 0. Enter a valid Number that is greater than 0"

    # Directories
    #TODO
    save_dir = increment_path(Path(project) / name, exist_ok=False)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir


    # Device init
    device = select_device(device, batch_size=batch_size)
    cuda = device.type != "cpu"



    #TODO splitting up data
    # Setting up the Images
    val_path = train_path = source
    params = (device, cache, nosave, batch_size // WORLD_SIZE)
    train_loader = create_dataloader(path=train_path, imgsz=imgsz, params=params, rank=RANK, prefix_for_log="train: ")

    if is_master_process(RANK): # Validation Data only needed in Process 0 (GPU) or -1 (CPU)
        val_loader = create_dataloader(path=val_path, imgsz=imgsz, params=params, rank=-1, prefix_for_log="validation: ")


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-Training all Functions-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    for selected_function in allowed_fn:
        if weights.endswith('.pt') and Path(weights).exists():
            model = house_brackmann_lookup[selected_function]["model"].to(device)
            ckpt = torch.load(weights, map_location=device)  # load checkpoint
            csd = ckpt[selected_function].float().state_dict()  # checkpoint state_dict as FP32
            model.load_state_dict(csd, strict=False)  # load
            #LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # TODo report
        else:
            model = house_brackmann_lookup[selected_function]["model"].to(device)

        model = select_data_parallel_mode(model, cuda)

        for i_name, img_struct, img_inv_struct,label_struct in train_loader:
            for idx, item_list in enumerate(zip(img_struct[selected_function], img_inv_struct[selected_function], label_struct[selected_function])):
                img, img_inv, label = item_list
                print(idx, selected_function, i_name, img.shape, img_inv.shape, label.shape)

        # #Optimizer
        # optimizer = OptimizerClass(model).select(optimizer)
        #
        # #TODO Scheduler
        # scheduler = SchedulerClass(optimizer).select("tmp")


        # LOGGER.info("Training %s. Using %s workers and Logging results to %s \n \
        #             Starting training for %s epochs...", selected_function, train_loader.num_workers, save_dir, epochs)


        # training_epochs(path=save_dir,
        #                 model=model,
        #                 optimizer=optimizer,
        #                 scheduler=scheduler,
        #                 device=device,
        #                 train_loader=train_loader,
        #                 epochs=epochs)
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    if is_master_process(RANK): #Plotting only needed in Process 0 (GPU) or -1 (CPU)
        pass #TODO LOGGING and Plotting and validating


    torch.cuda.empty_cache()
#    return results

























def parse_opt():
    """
    TODO
    Check internet connectivity
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default="model/model.pt",
                        help="model path(s)")
    parser.add_argument("--source", type=str, default="../test_data",
                        help="file/dir")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640],
                        help="inference size h,w")
    parser.add_argument("--cache", action="store_true",
                        help="Caching Images to a SQLite File (can get really big)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="total batch size for all GPUs")
    # parser.add_argument("--workers", type=int, default=8,
    #                     help="maximum number of dataloader workers")
    parser.add_argument("--device", default="cpu",
                        help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--optimizer", default="SGD",
                        help="Select the Optimizer")
    parser.add_argument("--nosave", action="store_true",
                        help="do not save")
    parser.add_argument("--project", default="../results/train",
                        help="save results to project/name")
    parser.add_argument("--name", default="run",
                        help="save results to project/name")
    parser.add_argument("--epochs", type=int, default=100,
                        help="total epochs running")
    return parser.parse_args()

if __name__ == "__main__":
    opt_args = parse_opt()
    set_logging(LOGGING_STATE, PREFIX, opt_args)
    check_requirements()
    time = timeit.timeit(lambda: run(**vars(opt_args)), number=1) #pylint: disable=unnecessary-lambda
    LOGGER.info("Done with Training. Finished in %s s", time)
