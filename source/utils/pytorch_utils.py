"""
TODO
"""

import logging
import math

import torch
from torch.optim import Adam, SGD

LOGGER = logging.getLogger(__name__)



def is_process_group(rank: int):
    """
     Checking Process Group

     #https://pytorch.org/docs/stable/elastic/run.html
     rank == -1 (no Process Group available)
     rank != -1 (process group is available)

    :param rank:  Rank of the Current Process. Can be LOCAL_RANK for local Process Group or RANK for global Process Group (int)
    :returns: True or False (bool)
    """
    return bool(rank != -1)

def is_master_process(rank: int):
    """
    Checking if working on the master node

    #https://pytorch.org/docs/stable/elastic/run.html

    rank == -1 (no Process Group)
    rank == 0 (Process Group 0 -- Master Process)

    :param rank:  Rank of the Current Process. Can be LOCAL_RANK for local Process Group or RANK for global Process Group (int)
    :returns: True or False (bool)
    """
    return bool(rank in [-1, 0])

def select_device(device="", batch_size=None):
    """
     Selecting Cuda/Cpu devices

    :param device:  List of devices ("cpu" or 0 or 0,1,2,3)
    :param batch_size: Batch size for Training (int)
    :returns: cuda:0 or cpu

    Info:
    https://discuss.pytorch.org/t/os-environ-cuda-visible-devices-not-functioning/105545/3
    https://discuss.pytorch.org/t/cuda-visible-device-is-of-no-use/10018
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
            assert batch_size % length_devices == 0, f"batch-size {batch_size} not multiple of GPU count {length_devices}"

        for i, j in enumerate(devices):
            dev_properties = torch.cuda.get_device_properties(i)
            torch_str += f"CUDA:{j} ({dev_properties.name}, {dev_properties.total_memory / (math.pow(1024,3))}GB)\n"  # bytes to MB
    else:
        torch_str += "CPU\n"

    LOGGER.info(torch_str)
    return torch.device("cuda:0" if cuda else "cpu")






#TODO merge scheduler and optimizer
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
