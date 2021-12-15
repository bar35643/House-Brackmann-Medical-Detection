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

import os
import math
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import optim
from torch.optim import lr_scheduler

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn import DataParallel

from .templates import house_brackmann_lookup #pylint: disable=import-error
from .config import LOGGER,LOCAL_RANK, RANK




def is_process_group(rank: int):
    """
    Checking Process Group

    Info:
    https://pytorch.org/docs/stable/elastic/run.html
    https://discuss.pytorch.org/t/what-is-the-difference-between-rank-and-local-rank/61940

    rank == -1 (no Process Group available)
    rank != -1 (process group is available)

    :param rank:  Rank of the Current Process. Can be LOCAL_RANK for local Process Group or RANK for global Process Group (int)
    :returns: True or False (bool)
    """
    return bool(rank != -1)

def is_master_process(rank: int):
    """
    Checking if working on the master node

    Info:
    https://pytorch.org/docs/stable/elastic/run.html
    https://discuss.pytorch.org/t/what-is-the-difference-between-rank-and-local-rank/61940

    rank == -1 (no Process Group)
    rank == 0 (Process Group 0 -- Master Process)

    :param rank:  Rank of the Current Process. Can be LOCAL_RANK for local Process Group or RANK for global Process Group (int)
    :returns: True or False (bool)
    """
    return bool(rank in [-1, 0])


def select_data_parallel_mode(model, cuda: bool):
    """
    Selecting DataParallel or DistributedDataParallel for the model

    Info:
    https://pytorch.org/docs/stable/elastic/run.html
    https://pytorch.org/docs/stable/distributed.html
    https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html

    :param model:  model selected for changind the mode (Model)
    :param cuda:  cuda aailable (bool)
    :returns: model (Model)

    Source:
        Project: yolov5
        License: GNU GPL v3
        Author: Glenn Jocher
        Url: https://github.com/ultralytics/yolov5
        Date of Copy: 6. October 2021
    Modified Code
    """

    #DP mode
    #Setting DataParrallel if Process Group not available but available devices more than 1
    if cuda and not is_process_group(RANK) and torch.cuda.device_count() > 1:
        LOGGER.info("DP not recommended! For better Multi-GPU performance with DistributedDataParallel \
                    use ---> python -m torch.distributed.run --nproc_per_node <gpu count> <file.py> <parser options>")
        model = DataParallel(model)

    #DDP mode
    #Setting to DistributedDataParralel if Process Group available
    if cuda and is_process_group(RANK):
        model = DistributedDataParallel(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    return model

def load_model(pth_to_weights, func):
    """
    Selecting Cuda/Cpu devices

    :param pth_to_weights:  Path to weights (str)
    :param func: function name (str)
    :returns: model
    """

    pth = os.path.join(Path(pth_to_weights), func + ".pt")
    if pth.endswith('.pt') and Path(pth).exists():
        LOGGER.debug("Using Pretrained model at Path %s", pth)

        model = house_brackmann_lookup[func]["model"]
        ckpt = torch.load(pth)  # load checkpoint
        model.load_state_dict(ckpt["model"], strict=False)  # load
        model.float()
    else:
        LOGGER.debug("Using General Model")
        model = house_brackmann_lookup[func]["model"]
    return model

def select_device(device="", batch_size=None):
    """
    Selecting Cuda/Cpu devices

    :param device:  List of devices ("cpu" or 0 or 0,1,2,3)
    :param batch_size: Batch size for Training (int)
    :returns: cuda device or cpu

    Info:
    https://discuss.pytorch.org/t/os-environ-cuda-visible-devices-not-functioning/105545/3
    https://discuss.pytorch.org/t/cuda-visible-device-is-of-no-use/10018
    https://discuss.pytorch.org/t/difference-between-torch-device-cuda-and-torch-device-cuda-0/46306/18
    https://pytorch.org/docs/1.9.0/generated/torch.cuda.set_device.html

    Source/Idea:
        Project: yolov5
        License: GNU GPL v3
        Author: Glenn Jocher
        Url: https://github.com/ultralytics/yolov5
        Date of Copy: 6. October 2021
    Modified Code
    """

    # device = "cpu" or "0" or "0,1,2,3"
    torch_str = f"Torch Version: torch {torch.__version__} Selected Devices: "  # string
    device = str(device).strip().lower().replace("cuda:", "").replace(" ", "")  # to string, "cuda:0" to "0" and "CPU" to "cpu"
    cpu = (device == "cpu") # set cpu to True/False
    cuda = not cpu and torch.cuda.is_available()

    assert cpu or torch.cuda.is_available(), f"CUDA unavailable, invalid device {device} requested!"

    if cuda:
        devices = device.split(",") if device else "0"  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        length_devices = len(devices)

        if length_devices > 1 and batch_size:  # check batch_size is divisible by device_count
            assert batch_size % length_devices == 0, f"--batch-size {batch_size} not multiple of GPU count {length_devices}"

        for i, j in enumerate(devices):
            dev_properties = torch.cuda.get_device_properties(i)
            torch_str += f"CUDA:{j} ({dev_properties.name}, {dev_properties.total_memory / (math.pow(1024,3))}GB)\n"  # bytes to MB
    else:
        torch_str += "CPU\n"
    LOGGER.info(torch_str)

     #Setting Devices to LOCAL_RANK if started as a Process Group
    if is_process_group(LOCAL_RANK):
        assert torch.cuda.device_count() > LOCAL_RANK, "insufficient CUDA devices for DDP command"
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device("cuda", LOCAL_RANK) # Sets for each Process group the GPU
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    #Setting device to default if Process Group is not startet
    else:
        device = torch.device("cuda:0" if cuda else "cpu")

    return device

@contextmanager
def torch_distributed_zero_first():
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.

    Source:
        Project: yolov5
        License: GNU GPL v3
        Author: Glenn Jocher
        Url: https://github.com/ultralytics/yolov5
        Date of Copy: 6. October 2021
    """

    if not is_master_process(LOCAL_RANK):
        dist.barrier(device_ids=[LOCAL_RANK])
    yield
    if LOCAL_RANK == 0:
        dist.barrier(device_ids=[0])


def de_parallel(model):
    """
    De-parallelize a model: returns single-GPU model if model is of type DP or DDP

    :param model:  Model (Model)
    :returns: De-parallelized model (Model)

    Info:
    https://pytorch.org/tutorials/beginner/saving_loading_models.html

    """
    return model.module if type(model) in (DataParallel, DistributedDataParallel) else model




optimizer_list = {
"Adadelta":   optim.Adadelta,
"Adagrad":    optim.Adagrad,
"Adam":       optim.Adam,
"AdamW":      optim.AdamW,
"SparseAdam": optim.SparseAdam,
"Adamax":     optim.Adamax,
"ASGD":       optim.ASGD,
"LBFGS":      optim.LBFGS,
"NAdam":      optim.NAdam,
"RAdam":      optim.RAdam,
"RMSprop":    optim.RMSprop,
"Rprop":      optim.Rprop,
"SGD":        optim.SGD,
}

scheduler_list = {
"LambdaLR": lr_scheduler.LambdaLR,
"MultiplicativeLR": lr_scheduler.MultiplicativeLR,
"StepLR": lr_scheduler.StepLR,
"MultiStepLR": lr_scheduler.MultiStepLR,
"ConstantLR": lr_scheduler.ConstantLR,
"LinearLR": lr_scheduler.LinearLR,
"ExponentialLR": lr_scheduler.ExponentialLR,
"CosineAnnealingLR": lr_scheduler.CosineAnnealingLR,
#"ReduceLROnPlateau": lr_scheduler.ReduceLROnPlateau,
"CyclicLR": lr_scheduler.CyclicLR,
"OneCycleLR": lr_scheduler.OneCycleLR,
"CosineAnnealingWarmRestarts": lr_scheduler.CosineAnnealingWarmRestarts,
}

def select_optimizer_and_scheduler(yml_hyp, neural_net, epoch):
    """
    Database functions to convert np.array to entry
    :param yml_hyp: Loaded YAML Config (dict)
    :param neural_net: model (model)
    :param epoch: Epochs (int)
    :return: scheduler, optimizer
    """
    item, param = list(yml_hyp['optimizer'].keys())[0], list(yml_hyp['optimizer'].values())[0]
    optimizer = optimizer_list[item](neural_net.parameters(), **param)


    scheduler_aray = []
    for i in yml_hyp['scheduler']:
        item, param = list(i.keys())[0], list(i.values())[0]
        scheduler_aray.append(   scheduler_list[item](optimizer, **param)   )


    if len(scheduler_aray) == 1:
        return scheduler_aray[0], optimizer

    if yml_hyp['sequential_scheduler']:
        length = len(scheduler_aray)
        milestone_size = epoch/length
        scheduler = lr_scheduler.SequentialLR(optimizer,
                                              schedulers=scheduler_aray,
                                              milestones=[math.floor(milestone_size*i) for i in range(1, length)],
                                              last_epoch=- 1,
                                              verbose=False)
    else:
        scheduler = lr_scheduler.ChainedScheduler(scheduler_aray)

    return scheduler, optimizer
