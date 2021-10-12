"""
TODO
Check internet connectivity
"""
import logging
from copy import deepcopy

import torch


from .pytorch_utils import is_process_group #pylint: disable=import-error


LOGGER = logging.getLogger(__name__)

#TODO Epoch running function
def training_epochs(path, model, optimizer, scheduler, device="cpu", train_loader=None, start_epoch=0, epochs=0):
    last, best = path / "weights/last.pt", path / "weights/best.pt"

    cuda = device.type != "cpu"
    compute_loss = ComputeLoss(model)  # init loss class
    scaler = amp.GradScaler(enabled=cuda)
    scheduler.last_epoch = start_epoch - 1  # do not move

    for epoch in range(start_epoch, epochs):
        model.train()
        if is_process_group(RANK):
            train_loader.sampler.set_epoch(epoch)
        optimizer.zero_grad()

        for (imgs, targets, paths, _) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size


            # Backward
            scaler.scale(loss).backward()


            # Optimize
            scaler.step(optimizer)  # optimizer.step
            scaler.update()
            optimizer.zero_grad()

        #Scheduler
        scheduler.step()
        # Save model
        if (not nosave) or (final_epoch):  # if save
            ckpt = {"epoch": epoch,
                    "best_fitness": best_fitness,
                    "model": deepcopy(de_parallel(model)),
                    "optimizer": optimizer.state_dict(),}


            # Save last, best and delete
            torch.save(ckpt, last)
            if best_fitness == fi:
                torch.save(ckpt, best)
            del ckpt
