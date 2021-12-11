#TODO Docstring
"""
TODO
"""


import argparse
import logging
import os
import timeit
import datetime
import gc as garbage_collector
from pathlib import Path

import yaml
from sklearn.metrics import accuracy_score

import torch
from torch.nn import CrossEntropyLoss
import torch.distributed as dist
from torch.cuda import amp

from utils.argparse_utils import restricted_val_split, SmartFormatter
from utils.config import RANK, WORLD_SIZE, LOGGER
from utils.general import check_requirements, increment_path, set_logging, OptArgs
from utils.pytorch_utils import select_device, select_data_parallel_mode, is_master_process, is_process_group, de_parallel, load_model, select_optimizer_and_scheduler
from utils.dataloader import create_dataloader, BatchSettings
from utils.templates import house_brackmann_lookup
from utils.plotting import Plotting
from utils.specs import validate_yaml_config

PREFIX = "train: "
LOGGING_STATE = logging.INFO
#https://pytorch.org/tutorials/beginner/saving_loading_models.html
#https://pytorch.org/docs/stable/amp.html
#https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
#https://towardsdatascience.com/7-tips-for-squeezing-maximum-performance-from-pytorch-ca4a40951259
#https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265


def run(weights="models", #pylint: disable=too-many-arguments, too-many-locals
        source="../data",
        hyp="./models/hyp.yaml",
        imgsz=640,
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

    pth = Path(hyp)
    assert hyp.endswith('.yaml') and pth.exists(), f"Error Path {hyp} has the wron ending or do not exist"
    with open(pth, 'r', encoding="UTF-8") as yaml_file:
        yml_hyp = yaml.safe_load(yaml_file)
        error, tru_fal = validate_yaml_config(yml_hyp)
        assert tru_fal, f"Error in YAML-Configuration (Path = {pth}): \n" + "\n".join(error)



    with open(save_dir / 'opt.yaml', 'w', encoding="UTF-8") as file:
        yaml.safe_dump(OptArgs.instance().args, file, sort_keys=False) #pylint: disable=no-member
    with open(save_dir / 'hyp.yaml', 'w', encoding="UTF-8") as file:
        yaml.safe_dump(yml_hyp, file, sort_keys=False) #pylint: disable=no-member

    # Device init
    device = select_device(device, batch_size=batch_size)
    cuda = device.type != "cpu"

    # Setting up the Images
    train_loader, val_loader = create_dataloader(path=source, imgsz=imgsz, device=device, cache=cache,
                                                 batch_size=batch_size // WORLD_SIZE, val_split=val_split, train_split=train_split)

    plotter = Plotting(path=save_dir, nosave=nosave, prefix_for_log=PREFIX)

    BatchSettings.instance().set_hyp(yml_hyp) #pylint: disable=no-member

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-Training all Functions-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    for selected_function in house_brackmann_lookup:
        last, best = os.path.join(model_save_dir, selected_function+"_last.pt"), os.path.join(model_save_dir, selected_function+"_best.pt")

        LOGGER.info("Training %s. Using Batch-Size %s and Logging results to %s. Starting training for %s epochs...\n", selected_function, batch_size, save_dir, epochs)

        model = load_model(weights, selected_function)
        model = select_data_parallel_mode(model, cuda).to(device, non_blocking=True)

        criterion = CrossEntropyLoss() #https://pytorch.org/docs/stable/nn.html

        #Optimizer & Scheduler
        _scheduler, _optimizer = select_optimizer_and_scheduler(yml_hyp, model, epochs)

        scaler = amp.GradScaler(enabled=cuda)

        best_score = 0
        for epoch in range(epochs):
            BatchSettings.instance().train() #pylint: disable=no-member
            model.train()
            if is_process_group(RANK):
                train_loader.sampler.set_epoch(epoch)

            #------------------------------BATCH------------------------------#
            LOGGER.info("train Epoch=%s", epoch)
            for i_name, img_struct,label_struct in train_loader:
                _optimizer.zero_grad()
                for idx, item_list in enumerate(zip(img_struct[selected_function], label_struct[selected_function])):
                    img, label = item_list
                    print(epoch, idx, selected_function, i_name, img.shape, label.shape)

                    img = img.to(device, non_blocking=True).float() # uint8 to float32

                    #https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
                    #https://pytorch.org/docs/stable/notes/amp_examples.html#amp-examples
                    #with amp.autocast(enabled=cuda):
                    pred = model(img)  # forward
                    loss = criterion(pred, label.to(device))  # loss scaled by batch_size
                    accurancy = accuracy_score(label.cpu(), pred.max(1)[1].cpu())

                    print("pred: ", pred.max(1)[1], "real: ", label, "loss: ", loss.item(), "accurancy: ", accurancy)

                    #Backward & Optimize
                    scaler.scale(loss).backward() #loss.backward()
                    scaler.step(_optimizer)  #optimizer.step
                    scaler.update()

                    plotter.update("train", selected_function, label.cpu(), pred.cpu(), loss)
            LOGGER.info("\n")
            #----------------------------END BATCH----------------------------#

            BatchSettings.instance().eval() #pylint: disable=no-member
            model.eval()
            if is_master_process(RANK): #Master Process 0 or -1
                #TODO validation
            #------------------------------BATCH------------------------------#
                LOGGER.info("val Epoch=%s", epoch)
                for i_name, img_struct,label_struct in val_loader:
                    _optimizer.zero_grad()
                    for idx, item_list in enumerate(zip(img_struct[selected_function], label_struct[selected_function])):
                        img, label = item_list
                        print(epoch, idx, selected_function, i_name, img.shape, label.shape)

                        img = img.to(device, non_blocking=True).float() # uint8 to float32

                        #https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
                        #https://pytorch.org/docs/stable/notes/amp_examples.html#amp-examples
                        #with amp.autocast(enabled=cuda):
                        pred = model(img)  # forward
                        loss = criterion(pred, label.to(device))  # loss scaled by batch_size
                        accurancy = accuracy_score(label.cpu(), pred.max(1)[1].cpu())

                        print("pred: ", pred.max(1)[1], "real: ", label, "loss: ", loss.item(), "accurancy: ", accurancy)

                        plotter.update("val", selected_function, label.cpu(), pred.cpu(), loss)
                LOGGER.info("\n")
            #----------------------------END BATCH----------------------------#
                val_dict = plotter.update_epoch(selected_function)

                # Save model
                if not nosave:  # if save
                    ckpt = {"timestamp": datetime.datetime.now(),
                            "epoch": epoch,
                            "score": val_dict,
                            "model": de_parallel(model).state_dict(),
                            "optimizer": _optimizer.state_dict(),
                            "scheduler": _scheduler.state_dict(),}

                    # Save last, best and delete ckpt
                    torch.save(ckpt, last)
                    if best_score <= val_dict["val_f1"]:
                        torch.save(ckpt, best)
                        best_score = val_dict["val_f1"]
                    del ckpt

            #Scheduler
            _scheduler.step()
            garbage_collector.collect()
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    if is_master_process(RANK): #Plotting only needed in Process 0 (GPU) or -1 (CPU)
        plotter.plot(show=False)


    torch.cuda.empty_cache()
    if WORLD_SIZE > 1 and RANK == 0:
        LOGGER.info('Destroying process group... ')
        dist.destroy_process_group()

























def parse_opt():
    """
    TODO
    Check internet connectivity
    """
    parser = argparse.ArgumentParser(formatter_class=SmartFormatter)
    parser.add_argument("--weights", type=str, default="models",
                        help="model folder")
    parser.add_argument("--source", type=str, default="../test_data",
                        help="file/dir")
    parser.add_argument("--hyp", "--hyperparameter", type=str, default="./models/hyp.yaml",
                        help="path to hyperparamer file")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640],
                        help="inference size h,w")
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
    parser.add_argument("--project", default="../results/train",
                        help="save results to project/name")
    parser.add_argument("--name", default="run",
                        help="save results to project/name")
    parser.add_argument("--epochs", type=int, default=100,
                        help="total epochs running")
    return parser.parse_args()

if __name__ == "__main__":
    opt_args = vars(parse_opt())
    OptArgs.instance()(opt_args)

    set_logging(LOGGING_STATE, PREFIX)

    if is_master_process(RANK):  #Master Process 0 or -1
        check_requirements()
    time = timeit.timeit(lambda: run(**opt_args), number=1) #pylint: disable=unnecessary-lambda
    LOGGER.info("Done with Training. Finished in %s s", time)
