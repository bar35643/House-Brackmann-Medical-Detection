#TODO Docstring
"""
TODO
"""

import math

import torch
from torch.optim import Adam, SGD

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn import DataParallel

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

    :param rank:  Rank of the global Process Group (int)
    :param rank:  Rank of the local Process Group (int)
    :param model:  model selected for changind the mode (Model)
    :param cuda:  cuda aailable (bool)
    :returns: model (Model)
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
    """

    # device = "cpu" or "0" or "0,1,2,3"
    torch_str = f"torch {torch.__version__} "  # string
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






#TODO  scheduler and optimizer to function
class OptimizerClass:
    """
    TODO
    Check internet connectivity
    """

    def __init__(self, neural_net):
        """
        TODO
        Check internet connectivity
        """
        self.neural_net = neural_net

        self.optimizer_list = {
        "SGD": SGD(self.neural_net.parameters(), lr=0.001, momentum=0.9, nesterov=True),
        "ADAM": Adam(self.neural_net.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False),
        #Adadelta(neural_net.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0),
        #Adagrad(neural_net.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0),
        #SparseAdam(neural_net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08),
        #Adamax(neural_net.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0),
        #ASGD(neural_net.parameters(), lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0),
        #LBFGS(neural_net.parameters(), lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09, history_size=100, line_search_fn=None),
        #RMSprop(neural_net.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False),
        #Rprop(neural_net.parameters(), lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
        }

    def select(self, argument):
        """
        TODO
        Check internet connectivity
        """
        ret_val = None
        if argument in self.optimizer_list:
            ret_val =  self.optimizer_list[argument]
        else:
            assert False, "given Optimizer is not in the list!"
        return ret_val

#TODO Scheduler
class SchedulerClass:
    """
    TODO
    Check internet connectivity
    """
    def __init__(self, optimizer):
        self.optimizer = optimizer

        self.scheduler_list = {
        }

    def select(self, argument):
        """
        TODO
        Check internet connectivity
        """
        ret_val = None
        if argument in self.scheduler_list:
            ret_val =  self.scheduler_list[argument]
        else:
            assert False, "given Scheduler is not in the list!"
        return ret_val
