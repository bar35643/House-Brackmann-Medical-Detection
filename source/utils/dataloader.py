"""
TODO
Check internet connectivity
"""




import os
from copy import deepcopy
import torch
from torch.utils.data import Dataset
from PIL import Image

import torchvision.transforms as T

from .pytorch_utils import is_process_group #pylint: disable=import-error
from .settings import house_brackmann_template #pylint: disable=import-error



class LoadImages(Dataset):  # for inference
    """
    TODO
    """
    def __init__(self, path, img_size=640, stride=32, auto=True):
        super().__init__()
        self.path = path
        self.img_size = img_size
        self.stride = stride
        self.auto = auto


    def __getitem__(self, idx):
        img = Image.open("../images/index.jpg")

        #assert img0 is not None, 'Image Not Found ' + path
        #print(f'image {self.count}/{self.nf} {path}: ', end='')

        # Convert
        valid_transforms = T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])
        img = valid_transforms(img)

        template_img=deepcopy(house_brackmann_template)

        template_img["symmetry"] = deepcopy(img)
        template_img["eye"] = deepcopy(img)
        template_img["mouth"] = deepcopy(img)
        template_img["forehead"] = deepcopy(img)

        return template_img

    def __len__(self):
        return 1  # number of files




#TODO caching and LoadImagesAndLabels and augment
def create_dataloader(path, imgsz, batch_size,
                      rank=-1, workers=8, prefix_for_log=""):
    """
    TODO
    """
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    dataset = LoadImagesAndLabels(path=path,
                                  imgsz=imgsz,
                                  batch_size=batch_size,
                                  prefix_for_log=prefix_for_log)

    batch_size = min(batch_size, len(dataset))

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_process_group(rank) else None
    loader = torch.utils.data.DataLoader

    dataloader = loader(dataset,batch_size=batch_size,sampler=sampler,num_workers=num_workers,collate_fn=None, pin_memory=False)
    return dataloader
