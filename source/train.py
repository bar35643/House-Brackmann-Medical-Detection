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
import argparse
import os
import timeit
import datetime
import gc as garbage_collector
from pathlib import Path

import yaml
from sklearn.metrics import accuracy_score

import torch
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from torch.cuda import amp

from torchinfo import summary

from hbmedicalprocessing.utils.argparse_utils import restricted_val_split, SmartFormatter
from hbmedicalprocessing.utils.config import RANK, WORLD_SIZE, LOGGER
from hbmedicalprocessing.utils.general import check_requirements, increment_path, set_logging, OptArgs
from hbmedicalprocessing.utils.pytorch_utils import (select_device,
                                                     select_data_parallel_mode,
                                                     is_master_process,
                                                     is_process_group,
                                                     de_parallel,
                                                     load_model,
                                                     select_optimizer_and_scheduler)
from hbmedicalprocessing.utils.dataloader import CreateDataloader, BatchSettings
from hbmedicalprocessing.utils.templates import house_brackmann_lookup
from hbmedicalprocessing.utils.plotting import Plotting
from hbmedicalprocessing.utils.specs import validate_file

PREFIX = "train: "
#https://pytorch.org/tutorials/beginner/saving_loading_models.html
#https://pytorch.org/docs/stable/amp.html
#https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
#https://towardsdatascience.com/7-tips-for-squeezing-maximum-performance-from-pytorch-ca4a40951259
#https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265


def run(weights="models", #pylint: disable=too-many-arguments, too-many-locals
        source="../data",
        config="./models/hyp.yaml",
        cache=False,
        batch_size=16,
        val_split=None,
        train_split=None,
        # workers=8,
        device="cpu",
        nosave=False,
        project="../results/train",
        name="run",
        epochs=100):
    """
    Training Routines for the Hosue-Brackmannn Score

    :param weights: path to Models (str)
    :param source: Data Source Directory (str)
    :param config: path to config File (str)
    :param cache: Enables external Cache (bool)
    :param batch_size: max size of the Batch (int)
    :param val_split: factor for splitting Dataset (int, float, None)
    :param train_split: factor for splitting Dataset (int, float, None)
    :param device: CPU or 0 or 0,1 (int)
    :param nosave: Disable saving (bool)
    :param project: Project main directory (str)
    :param name: Project name (str)
    :param epocs: Nomber of epochs (int)
    """
    LOGGER.info("%sStarting Training...",PREFIX)

    assert epochs, "Numper of Epochs is 0. Enter a valid Number that is greater than 0"

    # Directories for Saving the Results
    save_dir = increment_path(Path(project) / name, exist_ok=False)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir
    model_save_dir = save_dir /"models"
    model_save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    #Validates config file
    yml_hyp = validate_file(config)
    BatchSettings.instance().set_hyp(yml_hyp) #pylint: disable=no-member


    # Save Args and Config (Hyperparameters/Augmentation, Scheduler and Optimizer)
    with open(save_dir / 'opt.yaml', 'w', encoding="UTF-8") as file:
        yaml.safe_dump(OptArgs.instance().args, file, sort_keys=False) #pylint: disable=no-member
    with open(save_dir / 'hyp.yaml', 'w', encoding="UTF-8") as file:
        yaml.safe_dump(yml_hyp, file, sort_keys=False) #pylint: disable=no-member

    # Device Init
    device = select_device(device, batch_size=batch_size)
    cuda = device.type != "cpu"

    # Setting up the Dataloader
    dataloader = CreateDataloader(path=source, device=device, cache=cache,
                                  batch_size=batch_size // WORLD_SIZE,
                                  val_split=val_split, train_split=train_split)
    # Setting up the Plotter Classs
    plotter = Plotting(path=save_dir, nosave=nosave, prefix_for_log=PREFIX)

    LOGGER.info("\n")
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-Training all Functions-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    #Iterate over all modules
    for selected_module in house_brackmann_lookup:
        LOGGER.info("%sTraining %s. Using Batch-Size %s and Logging results to %s. Starting training for %s epochs...", PREFIX, selected_module, batch_size, save_dir, epochs)

        #Saving folder for the Models
        last = os.path.join(model_save_dir, selected_module+"_last.pt")
        best = os.path.join(model_save_dir, selected_module+"_best.pt")

        #Loading the Model and selecting mode
        model = load_model(weights, selected_module)
        model = select_data_parallel_mode(model, cuda).to(device, non_blocking=True)

        #Model infos
        input_size = (batch_size // WORLD_SIZE, 27) + tuple(BatchSettings.instance().hyp["imgsz"][selected_module]) #pylint: disable=no-member
        LOGGER.info("%sModel Infos --> Batch input_size=%s", PREFIX, input_size)
        LOGGER.info("%sModel Infos --> Layer View and Estimated Size\n %s\n", PREFIX, summary(model, input_size=input_size, verbose=0))

        #Get Dataloader specific from teh functeion because of Imbalance
        train_loader, val_loader = dataloader.get_dataloader_func(selected_module)

        #Optimizer & Scheduler & Loss function
        _scheduler, _optimizer = select_optimizer_and_scheduler(yml_hyp, model, epochs)
        criterion = CrossEntropyLoss() #https://pytorch.org/docs/stable/nn.html

        scaler = amp.GradScaler(enabled=cuda)

        best_score = 0
        for epoch in range(epochs):
            BatchSettings.instance().train() #pylint: disable=no-member
            model.train()
            if is_process_group(RANK):
                train_loader.sampler.set_epoch(epoch)

            #------------------------------BATCH------------------------------#
            LOGGER.info("Start train Epoch=%s", epoch)
            for idx, item in enumerate(train_loader):
                i_name, img_struct,label_struct = item
                img   = img_struct[selected_module]
                label = label_struct[selected_module]

                LOGGER.debug("train -> epoch=%s, minibatch-id=%s, names=%s, img-shape=%s, label-shape=%s",
                              epoch, idx, selected_module, i_name, img.shape, label.shape)

                _optimizer.zero_grad()
                img = img.to(device, non_blocking=True).float() # uint8 to float32

                #https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
                #https://pytorch.org/docs/stable/amp.html
                #https://pytorch.org/docs/stable/notes/amp_examples.html#amp-examples
                #with amp.autocast(enabled=cuda):
                pred = model(img)  # forward
                loss = criterion(pred, label.to(device))
                accurancy = accuracy_score(label.cpu(), pred.max(1)[1].cpu())

                LOGGER.info("pred=%s", pred.max(1)[1])
                LOGGER.info("real=%s", label)
                LOGGER.info("loss=%s, accurancy=%s", loss.item(), accurancy)

                #Backward & Optimize
                scaler.scale(loss).backward() #loss.backward()
                scaler.step(_optimizer)  #optimizer.step
                scaler.update()

                plotter.update("train", selected_module, label.cpu(), pred.cpu(), loss)
            LOGGER.info("End train Epoch=%s", epoch)
            #----------------------------END BATCH----------------------------#

            BatchSettings.instance().eval() #pylint: disable=no-member
            model.eval() #https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
        #---------------------------Validation----------------------------#
            if is_master_process(RANK): #Master Process 0 or -1
            #------------------------------BATCH------------------------------#
                LOGGER.info("Start val Epoch=%s", epoch)
                with torch.no_grad(): ##https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
                    for idx, item in enumerate(val_loader):
                        i_name, img_struct,label_struct = item
                        img   = img_struct[selected_module]
                        label = label_struct[selected_module]

                        LOGGER.debug("val -> epoch=%s, minibatch-id=%s, names=%s, img-shape=%s, label-shape=%s",
                                      epoch, idx, selected_module, i_name, img.shape, label.shape)

                        _optimizer.zero_grad()
                        img = img.to(device, non_blocking=True).float() # uint8 to float32

                        #https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
                        #https://pytorch.org/docs/stable/amp.html
                        #https://pytorch.org/docs/stable/notes/amp_examples.html#amp-examples
                        #with amp.autocast(enabled=cuda):
                        pred = model(img)
                        loss = criterion(pred, label.to(device))
                        accurancy = accuracy_score(label.cpu(), pred.max(1)[1].cpu())

                        LOGGER.info("pred=%s", pred.max(1)[1])
                        LOGGER.info("real=%s", label)
                        LOGGER.info("loss=%s, accurancy=%s", loss.item(), accurancy)

                        plotter.update("val", selected_module, label.cpu(), pred.cpu(), loss)
                LOGGER.info("End val Epoch=%s", epoch)
            #----------------------------END BATCH----------------------------#
                val_dict = plotter.update_epoch(selected_module)

                # Save model
                if not nosave:  #nosave is not enabled
                    ckpt = {"timestamp": datetime.datetime.now(),
                            "epoch": epoch,
                            "score": val_dict,
                            "model": de_parallel(model).state_dict(),
                            "optimizer": _optimizer.state_dict(),
                            "scheduler": _scheduler.state_dict(),}

                    # Save last, best and delete ckpt
                    torch.save(ckpt, last)
                    LOGGER.debug("Saved model for epoch %s", epoch)
                    if best_score <= val_dict["val_f1"]:
                        torch.save(ckpt, best)
                        best_score = val_dict["val_f1"]
                        LOGGER.debug("Saved model at epoch %s for best f1 score %s", epoch, best_score)
                    del ckpt
        #-------------------------End Validation--------------------------#
            #Scheduler Step
            _scheduler.step()
            #Collecting active garbage
            collected = garbage_collector.collect()
            LOGGER.debug('Collected Garbage: %s', collected)
            LOGGER.info('\n')
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    if is_master_process(RANK): #Plotting only needed in Process 0 (GPU) or -1 (CPU)
        plotter.plot(show=False)

    #Emptying cuda cache and destroying porcess group
    torch.cuda.empty_cache()
    if WORLD_SIZE > 1 and RANK == 0:
        LOGGER.info('Destroying process group... ')
        dist.destroy_process_group()

























def parse_opt():
    """
    Command line Parser Options see >> python train.py -h for more about
    """
    parser = argparse.ArgumentParser(formatter_class=SmartFormatter)
    parser.add_argument("--weights", type=str, default="models",
                        help="model folder")
    parser.add_argument("--source", type=str, default="../test_data",
                        help="file/dir")
    parser.add_argument("--config", "--cfg", type=str, default="./models/hyp.yaml",
                        help="path to hyperparamer file")
    parser.add_argument("--cache", action="store_true",
                        help="Caching Images to a SQLite File (can get really big)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="total batch size for all GPUs")
    parser.add_argument("--val-split", type=restricted_val_split, default=None,
                        help="R|Factor for splitting Validation for x=len(dataset) fac=Value:                           \n"
                        "None  (both --val-split and --train-split)                --> Val=Train=x,                     \n"
                        "float (if --train-split=None) between [0,1]               --> Val=fac*x      Train=(1-fac)*x,  \n"
                        "float sum of --train-split and --val-split between [0,1]  --> Val=fac*x,                       \n"
                        "int   (if --train-split=None)                             --> Val=fac        Train=x-fac       \n"
                        "int   (if --train-split=float or int)                     --> Val=fac                            ")
    parser.add_argument("--train-split", type=restricted_val_split, default=None,
                        help="R|Factor for splitting Train for x=len(dataset) fac=Value:                                \n"
                        "None  (both --val-split and --train-split)                --> Train=Val=x,                     \n"
                        "float (if --val-split=None) between [0,1]                 --> Train=fac*x      Val=(1-fac)*x,  \n"
                        "float sum of --train-split and --val-split between [0,1]  --> Train=fac*x,                     \n"
                        "int   (if --val-split=None)                               --> Train=fac        Val=x-fac       \n"
                        "int   (if --val-split=float or int)                       --> Train=fac                          ")
    # parser.add_argument("--workers", type=int, default=8,
    #                     help="maximum number of dataloader workers")
    parser.add_argument("--device", default="cpu",
                        help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--nosave", action="store_true",
                        help="do not save if activated")
    parser.add_argument("--project", default="../../results/train",
                        help="save results to project/name")
    parser.add_argument("--name", default="run",
                        help="save results to project/name")
    parser.add_argument("--epochs", type=int, default=100,
                        help="total epochs running")
    return parser.parse_args()

if __name__ == "__main__":
    opt_args = vars(parse_opt())
    OptArgs.instance()(opt_args)

    if is_master_process(RANK):  #Master Process 0 or -1
        set_logging(PREFIX)
        check_requirements()
    time = timeit.timeit(lambda: run(**opt_args), number=1) #pylint: disable=unnecessary-lambda
    LOGGER.info("Done with Training. Finished in %s s", time)
