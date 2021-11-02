#TODO Docstring
"""
TODO
"""
#python env_test.py
#python -m torch.distributed.run --nproc_per_node 2 env_test.py

#pylint: skip-file
import logging
from argparse import Namespace
import torch

from utils.config import ROOT, ROOT_RELATIVE, LOCAL_RANK, RANK, WORLD_SIZE, LOGGER

from utils.general import set_logging
from utils.pytorch_utils import select_device, is_process_group, is_master_process
from utils.dataloader import CreateDataset, LoadImages


if __name__ == "__main__":
    set_logging(logging.INFO, "env_test: ", Namespace())
    print("-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-")
    print("ROOT: ", ROOT)
    print("ROOT_RELATIVE: ", ROOT_RELATIVE)
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
    #print(select_device('0, 1, 2'))

    print(is_process_group(LOCAL_RANK))
    print(is_master_process(LOCAL_RANK))

    print("\n\ntesting CreateDataset\n")
    tst = CreateDataset(path='../test_data', imgsz=640, prefix_for_log='')
    print("length: ", len(tst))
    print(tst[0])

    print("\n\ntesting LoadImages (All Categories, Single Category, Single Patient) \n")
    tst = LoadImages(path='../test_data', imgsz=640, prefix_for_log='')
    print("length: ", len(tst))
    tst = LoadImages(path='../test_data/Muskeltransplantation', imgsz=640, prefix_for_log='')
    print("length: ", len(tst))
    tst = LoadImages(path='../test_data/Muskeltransplantation/0001', imgsz=640, prefix_for_log='')
    print("length: ", len(tst))
    print("-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-")
