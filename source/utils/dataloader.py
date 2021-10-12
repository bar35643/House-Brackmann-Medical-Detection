"""
TODO
Check internet connectivity
"""




import os
import csv
from copy import deepcopy
from PIL import Image


import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from .pytorch_utils import is_process_group #pylint: disable=import-error
from .templates import house_brackmann_template #pylint: disable=import-error


class LoadImages(Dataset):  # for inference
    """
    TODO
    """
    def __init__(self, path, imgsz=640, prefix_for_log=""):
        super().__init__()
        self.path = path
        self.imgsz = imgsz
        self.prefix_for_log = prefix_for_log


        #TODO loading single Patient
        #TODO Loading one Category            ready
        #TODO Loading all Categories
        self.listdir = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        self.listdir.sort()
        self.length = len(self.listdir)

        print("tst: ", self.length, self.listdir)

    def transform_image(self, img):
        #TODO Augmentation
        valid_transforms = T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])
        return valid_transforms(img)

    def __getitem__(self, idx):
        item_name = self.listdir[idx]
        #TODO Load and Process images
        #TODO Load Images, cut it and do Augmentation



        img = Image.open("../images/index.jpg")

        #assert img0 is not None, 'Image Not Found ' + path
        #print(f'image {self.count}/{self.nf} {path}: ', end='')

        #TODO cutting image in pieces

        img = self.transform_image(img)


        # #Build and return structure
        struct_images = deepcopy(house_brackmann_template)
        struct_images["symmetry"] = deepcopy(img)
        struct_images["eye"] = deepcopy(img)
        struct_images["mouth"] = deepcopy(img)
        struct_images["forehead"] = deepcopy(img)


        struct_images_inv = deepcopy(house_brackmann_template)
        for i in struct_images:
            struct_images_inv[i] = torch.fliplr(struct_images[i])

        return item_name, struct_images, struct_images_inv

    def __len__(self):
        return self.length






class LoadLabels(Dataset):
    """
    TODO
    """
    def __init__(self, path, name, prefix_for_log=""):
        super().__init__()
        self.path = path + '.csv'
        print(self.path)
        self.prefix_for_log = prefix_for_log


        #load CSV
        self.listdir = []
        with open(self.path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            next(reader, None)  # skip the headers
            for row in reader:
                self.listdir.append(row)
        self.listdir.sort()
        self.length = len(self.listdir)


    def __getitem__(self, idx):
        item_name = self.listdir[idx]

        #TODO Extract Data and lookup

        #Set and Return value
        struct_images = deepcopy(house_brackmann_template)
        struct_images["symmetry"] = None
        struct_images["eye"] = None
        struct_images["mouth"] = None
        struct_images["forehead"] = None

        return item_name, struct_images

    def __len__(self):
        return self.length  # number of files

class CreateDataset(Dataset):
    """
    TODO
    """
    def __init__(self, path='', imgsz=640, prefix_for_log=''):
        super().__init__()
        self.path = path
        self.prefix_for_log = prefix_for_log
        self.images = []
        self.labels = []

        self.listdir = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        for s_dir in self.listdir:
            self.images += LoadImages(path=os.path.join(path, s_dir), imgsz=imgsz, prefix_for_log=prefix_for_log)
            self.labels += LoadLabels(path=os.path.join(path, s_dir), name=dir, prefix_for_log=prefix_for_log)
        self.len_images = len(self.images)
        self.len_labels = len(self.labels)

        assert not self.len_images != self.len_labels, f"Length of the Images ({self.len_images}) do not match to length of Labels({self.len_labels}) ."

    def __getitem__(self, idx):
        #TODO return only right pair of Images on Label (checking if same Patient)
        #TODO augment left right,2 times training and detection

        i_name, img = self.images[idx]
        l_name, label = self.labels[idx]

        return img, label

    def __len__(self):
        return self.len_images

#TODO caching and LoadImagesAndLabels and augment
def create_dataloader(path, imgsz, batch_size,
                      rank=-1, workers=8, prefix_for_log=""):
    """
    TODO
    """
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    dataset = CreateDataset(path=path, imgsz=imgsz, prefix_for_log=prefix_for_log)


    batch_size = min(batch_size, len(dataset))

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_process_group(rank) else None
    loader = torch.utils.data.DataLoader

    dataloader = loader(dataset,batch_size=batch_size,sampler=sampler,num_workers=num_workers,collate_fn=None, pin_memory=False)
    return dataloader
