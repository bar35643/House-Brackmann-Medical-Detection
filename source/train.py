#TODO Docstring
"""
TODO
"""


import argparse
import logging
import os
import timeit
from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss
from torch.cuda import amp

from utils.argparse_utils import restricted_val_split
from utils.config import ROOT, ROOT_RELATIVE, RANK, WORLD_SIZE, LOGGER
from utils.general import check_requirements, increment_path, set_logging
from utils.pytorch_utils import select_device, select_data_parallel_mode, select_optimizer, select_scheduler, is_master_process, is_process_group, de_parallel
from utils.dataloader import create_dataloader
from utils.templates import allowed_fn, house_brackmann_lookup

PREFIX = "train: "
LOGGING_STATE = logging.INFO


#https://pytorch.org/tutorials/beginner/saving_loading_models.html
#https://pytorch.org/docs/stable/amp.html


def run(weights="model", #pylint: disable=too-many-arguments, too-many-locals
        source="../data",
        imgsz=640,
        cache=False,
        batch_size=16,
        val_split=None,
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
    model_save_dir = save_dir /"models"
    model_save_dir.mkdir(parents=True, exist_ok=True)  # make dir


    # Device init
    device = select_device(device, batch_size=batch_size)
    cuda = device.type != "cpu"

    # Setting up the Images
    train_loader, val_loader = create_dataloader(path=source, imgsz=imgsz, device=device, cache=cache,
                                                 nosave=nosave, batch_size=batch_size // WORLD_SIZE, val_split=val_split)
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-Training all Functions-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    for selected_function in allowed_fn:
        last, best = os.path.join(model_save_dir, selected_function+"_last.pt"), os.path.join(model_save_dir, selected_function+"_best.pt")
        weights_from_func = os.path.join(Path(weights), selected_function + ".pt")
        if weights_from_func.endswith('.pt') and Path(weights_from_func).exists():
            model = house_brackmann_lookup[selected_function]["model"].to(device)
            ckpt = torch.load(weights_from_func, map_location=device)  # load checkpoint
            csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
            model.load_state_dict(csd, strict=False)  # load
            #LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # TODo report
        else:
            model = house_brackmann_lookup[selected_function]["model"].to(device)

        # LOGGER.info("Training %s. Using %s workers and Logging results to %s \n \
        #             Starting training for %s epochs...", selected_function, train_loader.num_workers, save_dir, epochs)

        # #Optimizer & Scheduler
        _optimizer = select_optimizer(model, optimizer)
        _scheduler = select_scheduler(_optimizer, "StepLR")

        model = select_data_parallel_mode(model, cuda).to(device, non_blocking=True)
        compute_loss = CrossEntropyLoss()
        scaler = amp.GradScaler(enabled=cuda)
        _scheduler.last_epoch = epochs - 1  # do not move

        for epoch in range(epochs):
            model.train()
            if is_process_group(RANK):
                train_loader.sampler.set_epoch(epoch)
            _optimizer.zero_grad()

            for i_name, img_struct,label_struct in train_loader:
                for idx, item_list in enumerate(zip(img_struct[selected_function], label_struct[selected_function])):
                    img, label = item_list
                    print(epoch, idx, selected_function, i_name, img.shape, label.shape)

                    img = img.to(device, non_blocking=True).float() # uint8 to float32
                    with amp.autocast(enabled=cuda):
                        pred = model(img)  # forward
                        loss = compute_loss(pred, label)  # loss scaled by batch_size

                    #Backward & Optimize
                    scaler.scale(loss).backward() #loss.backward()
                    scaler.step(_optimizer)  #optimizer.step
                    scaler.update()

                #Scheduler
                _scheduler.step()
                if is_master_process(RANK):
                    #TODO validation

                    # Save model
                    if not nosave:  # if save
                        ckpt = {"epoch": epoch,
                                #"best_fitness": best_fitness,
                                "model": de_parallel(model).state_dict(),
                                "optimizer": _optimizer.state_dict(),}


                        # Save last, best and delete
                        torch.save(ckpt, last)
                        #if best_fitness == fi:
                        #    torch.save(ckpt, best)
                        del ckpt



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
    parser.add_argument("--weights", type=str, default="model",
                        help="model folder")
    parser.add_argument("--source", type=str, default="../test_data",
                        help="file/dir")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640],
                        help="inference size h,w")
    parser.add_argument("--cache", action="store_true",
                        help="Caching Images to a SQLite File (can get really big)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="total batch size for all GPUs")
    parser.add_argument("--val-split", type=restricted_val_split, default=None,
                        help="Factor for splitting Train and Validation for x=len(dataset):  \
                        None --> Train=Val=x, float between [0,1] --> Train=(1-fac)*x Val=fac*x, int --> Train=dataset-x Val=x")
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
