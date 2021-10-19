"""
TODO
Check internet connectivity
"""


import argparse
import logging
import timeit
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn import DataParallel

from utils.config import ROOT, ROOT_RELATIVE, LOCAL_RANK, RANK, WORLD_SIZE, LOGGER
from utils.general import check_requirements, increment_path, set_logging
from utils.pytorch_utils import select_device, OptimizerClass, SchedulerClass, is_master_process, is_process_group
from utils.dataloader import create_dataloader
from utils.common import training_epochs
from utils.templates import allowed_fn, house_brackmann_lookup

PREFIX = "train: "
LOGGING_STATE = logging.INFO


#https://discuss.pytorch.org/t/what-is-the-difference-between-rank-and-local-rank/61940
#https://pytorch.org/docs/stable/elastic/run.html
#https://pytorch.org/docs/stable/distributed.html
#https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html
#https://discuss.pytorch.org/t/difference-between-torch-device-cuda-and-torch-device-cuda-0/46306/18
#https://discuss.pytorch.org/t/what-is-the-difference-between-rank-and-local-rank/61940/2
#https://discuss.pytorch.org/t/cuda-visible-device-is-of-no-use/10018/9
#https://pytorch.org/docs/1.9.0/generated/torch.cuda.set_device.html


#https://pytorch.org/tutorials/beginner/saving_loading_models.html



def run(weights="model/model.pt", #pylint: disable=too-many-arguments, too-many-locals
        source="../data",
        imgsz=640,
        batch_size=16,
        workers=8,
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
    if is_process_group(LOCAL_RANK): #Setting Devices to LOCAL_RANK if started as a Process Group
        assert torch.cuda.device_count() > LOCAL_RANK, "insufficient CUDA devices for DDP command"
        assert batch_size % WORLD_SIZE == 0, "--batch-size must be multiple of CUDA device count"
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device("cuda", LOCAL_RANK) # Sets for each Process group the GPU
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
    cuda = device.type != "cpu"



    #TODO splitting up data
    # Setting up the Images
    val_path = train_path = source

    train_loader = create_dataloader(path=train_path, imgsz=imgsz, device=device, batch_size=batch_size // WORLD_SIZE,
    rank=RANK, workers=workers, prefix_for_log="train: ")

    if is_master_process(RANK): # Validation Data only needed in Process 0 (GPU) or -1 (CPU)
        val_loader = create_dataloader(path=val_path, imgsz=imgsz, device=device, batch_size=batch_size // WORLD_SIZE,
        rank=-1, workers=workers, prefix_for_log="validation: ")


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

        # DP mode
        if cuda and not is_process_group(RANK) and torch.cuda.device_count() > 1: #Setting DataParrallel if Process Group not available but available devices more than 1
            LOGGER.info("DP not recommended! For better Multi-GPU performance with DistributedDataParallel \
                        use ---> torch.distributed.run --nproc_per_node <gpu count> <file.py> <parser options>")
            model = DataParallel(model)
        # DDP mode
        if cuda and is_process_group(RANK): #Setting to DistributedDataParralel if Process Group available
            model = DistributedDataParallel(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

        # #Optimizer
        # optimizer = OptimizerClass(model).select(optimizer)
        #
        # #TODO Scheduler
        # scheduler = SchedulerClass(optimizer).select("tmp")


        LOGGER.info("Training %s. Using %s workers and Logging results to %s \n \
                    Starting training for %s epochs...", selected_function, train_loader.num_workers, save_dir, epochs)


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
    parser.add_argument("--batch-size", type=int, default=16,
                        help="total batch size for all GPUs")
    parser.add_argument("--workers", type=int, default=8,
                        help="maximum number of dataloader workers")
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
    #pylint: disable=pointless-string-statement
    """
    TODO
    Check internet connectivity
    """
    opt_args = parse_opt()
    set_logging(LOGGING_STATE, PREFIX, opt_args)
    check_requirements()
    time = timeit.timeit(lambda: run(**vars(opt_args)), number=1) #pylint: disable=unnecessary-lambda
    LOGGER.info("Done with Training. Finished in %s s", time)
