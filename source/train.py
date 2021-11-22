#TODO Docstring
"""
TODO
"""


import argparse
import logging
import os
import timeit
from pathlib import Path

import yaml
from sklearn.metrics import accuracy_score

import torch
from torch.nn import CrossEntropyLoss
import torch.distributed as dist
from torch.cuda import amp

from utils.argparse_utils import restricted_val_split
from utils.config import ROOT, ROOT_RELATIVE, RANK, WORLD_SIZE, LOGGER
from utils.general import check_requirements, increment_path, set_logging
from utils.pytorch_utils import select_device, select_data_parallel_mode, select_optimizer, select_scheduler, is_master_process, is_process_group, de_parallel, AverageMeter, load_model
from utils.dataloader import create_dataloader, BoolAugmentation
from utils.templates import allowed_fn
from utils.plotting import Plotting

PREFIX = "train: "
LOGGING_STATE = logging.INFO
opt_args = None
#https://pytorch.org/tutorials/beginner/saving_loading_models.html
#https://pytorch.org/docs/stable/amp.html
#https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
#https://towardsdatascience.com/7-tips-for-squeezing-maximum-performance-from-pytorch-ca4a40951259
#https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265


def run(weights="model", #pylint: disable=too-many-arguments, too-many-locals
        source="../data",
        imgsz=640,
        cache=False,
        batch_size=16,
        val_split=None,
        # workers=8,
        device="cpu",
        optimizer="SGD",
        scheduler="StepLR",
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

    with open(save_dir / 'opt.yaml', 'w', encoding="UTF-8") as file:
        yaml.safe_dump(vars(opt_args), file, sort_keys=False)


    # Device init
    device = select_device(device, batch_size=batch_size)
    cuda = device.type != "cpu"

    # Setting up the Images
    train_loader, val_loader = create_dataloader(path=source, imgsz=imgsz, device=device, cache=cache,
                                                 nosave=nosave, batch_size=batch_size // WORLD_SIZE, val_split=val_split)
    plotter = Plotting(path=save_dir, nosave=nosave, prefix_for_log=PREFIX)
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-Training all Functions-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    for selected_function in allowed_fn:
        last, best = os.path.join(model_save_dir, selected_function+"_last.pt"), os.path.join(model_save_dir, selected_function+"_best.pt")

        # LOGGER.info("Training %s. Using %s workers and Logging results to %s \n \
        #             Starting training for %s epochs...", selected_function, train_loader.num_workers, save_dir, epochs)

        model = load_model(weights, selected_function)
        model = select_data_parallel_mode(model, cuda).to(device, non_blocking=True)

        criterion = CrossEntropyLoss() #https://pytorch.org/docs/stable/nn.html

        #Optimizer & Scheduler
        _optimizer = select_optimizer(model, optimizer)
        _scheduler = select_scheduler(_optimizer, scheduler)
        scaler = amp.GradScaler(enabled=cuda)

        for epoch in range(epochs):
            BoolAugmentation.instance().train() #pylint: disable=no-member
            model.train()
            if is_process_group(RANK):
                train_loader.sampler.set_epoch(epoch)

            #------------------------------BATCH------------------------------#
            loss_meter     = AverageMeter()
            accuracy_meter = AverageMeter()
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
                    loss = criterion(pred, label)  # loss scaled by batch_size
                    accurancy = accuracy_score(label, pred.max(1)[1].cpu())

                    loss_meter.update(loss.item())
                    accuracy_meter.update(accurancy)

                    print("pred: ", pred.max(1)[1], "real: ", label, "loss: ", loss.item(), "accurancy: ", accurancy)
                    print("loss_avg: ", loss_meter.avg, "accurancy_avg: ", accuracy_meter.avg)

                    #Backward & Optimize
                    scaler.scale(loss).backward() #loss.backward()
                    scaler.step(_optimizer)  #optimizer.step
                    scaler.update()

                    #https://en.wikipedia.org/wiki/Confusion_matrix
                    #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix
                    #https://deeplizard.com/learn/video/0LhiS6yu2qQ
                    plotter.confusion_matrix_update("train", selected_function, label, pred)
            #----------------------------END BATCH----------------------------#

            #Scheduler
            _scheduler.step()

            BoolAugmentation.instance().eval() #pylint: disable=no-member
            model.eval()
            if is_master_process(RANK): #Master Process 0 or -1
                #TODO validation
            #------------------------------BATCH------------------------------#
                loss_meter     = AverageMeter()
                accuracy_meter = AverageMeter()
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
                        loss = criterion(pred, label)  # loss scaled by batch_size
                        accurancy = accuracy_score(label, pred.max(1)[1].cpu())

                        loss_meter.update(loss.item())
                        accuracy_meter.update(accurancy)

                        print("pred: ", pred.max(1)[1], "real: ", label, "loss: ", loss.item(), "accurancy: ", accurancy)
                        print("loss_avg: ", loss_meter.avg, "accurancy_avg: ", accuracy_meter.avg)

                        #https://en.wikipedia.org/wiki/Confusion_matrix
                        #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix
                        #https://deeplizard.com/learn/video/0LhiS6yu2qQ
                        plotter.confusion_matrix_update("val", selected_function, label, pred)
            #----------------------------END BATCH----------------------------#



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
        plotter.plot(show=True)


    torch.cuda.empty_cache()
    if WORLD_SIZE > 1 and RANK == 0:
        LOGGER.info('Destroying process group... ')
        dist.destroy_process_group()

























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
    parser.add_argument("--scheduler", default="StepLR",
                        help="Select the Scheduler. If using more than one Scheduler, seperate with Comma")
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
    opt_args = parse_opt()
    set_logging(LOGGING_STATE, PREFIX, opt_args)
    if is_master_process(RANK):  #Master Process 0 or -1
        check_requirements()
    time = timeit.timeit(lambda: run(**vars(opt_args)), number=1) #pylint: disable=unnecessary-lambda
    LOGGER.info("Done with Training. Finished in %s s", time)
