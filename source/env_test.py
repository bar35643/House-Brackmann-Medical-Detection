"""
TODO
"""

#python env_test.py
#python -m torch.distributed.run --nproc_per_node 2 env_test.py



import  os
import logging
from argparse import Namespace

import torch

from utils.general import set_logging
from utils.pytorch_utils import select_device, is_process_group, is_master_process

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "-1"))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", "-1"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))

LOGGER = logging.getLogger(__name__)

if __name__ == "__main__":
    set_logging(logging.DEBUG, "env_test: ", Namespace())
    print("-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-")
    print("LOCAL_RANK: ", LOCAL_RANK)
    print("RANK: ", RANK)
    print("WORLD_SIZE: ", WORLD_SIZE)


    print("Cuda Avialable: ", torch.cuda.is_available())
    print("Cuda device count: ", torch.cuda.device_count())
    print("Selected device 1: ", torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    if is_process_group(LOCAL_RANK):
        print("Selected device 2", torch.device("cuda" if torch.cuda.is_available() else "cpu", LOCAL_RANK))
    print("Distributed is available: ", torch.distributed.is_available())

    if torch.cuda.is_available():
        print("Device Properties: ", torch.cuda.get_device_properties("cuda:0"))


    print(select_device('cpu'))
    print(select_device('0, 1, 2'))

    print(is_process_group(LOCAL_RANK))
    print(is_master_process(LOCAL_RANK))
    print("-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-")
