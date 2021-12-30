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




import os
import csv
from copy import deepcopy
from itertools import repeat
from functools import lru_cache
from multiprocessing.pool import ThreadPool
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

from .config import LOGGER, LRU_MAX_SIZE, RANK, LOCAL_RANK, THREADPOOL_NUM_THREADS
from .cutter import Cutter
from .database_utils import Database
from .pytorch_utils import is_process_group, is_master_process, torch_distributed_zero_first #pylint: disable=import-error
from .templates import house_brackmann_template, house_brackmann_lookup, house_brackmann_grading #pylint: disable=import-error
from .general import init_dict #pylint: disable=import-error
from .decorators import try_except #pylint: disable=import-error
from .singleton import Singleton #pylint: disable=import-error


@Singleton
class BatchSettings():
    """
    Class for setting the augmentation to true or false globally
    """
    def __init__(self):
        """
        Initializes the class
        :param augmentation: (bool)
        :param func: (str)
        """
        self.augmentation = False
        self.hyp = None

    def set_hyp(self, yml_hyp):
        """
        Setting the Hyperparameter
        :param yml_hyp: Hyperparameter Dictionary (dict)
        """
        self.hyp = yml_hyp["hyp"]
        #print(self.hyp["Normalize"])

    def train(self):
        """
        Set the augmentation to False
        """
        self.augmentation = True

    def eval(self):
        """
        Set the augmentation to True
        """
        self.augmentation = False
    def get_augmentation(self):
        """
        Return the augmentation value

        :returns augmentation (bool)
        """
        return self.augmentation

    def __call__(self):
        assert False, f"Select one of the functions from this class: {dir(self)}"

def get_list_patients(source_path: str):
    """
    Generating a list from the Patients

    :param source_path: path (str)
    :return List (arr)
    """
    list_patients=[]
    list_dir = [f for f in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, f))]
    for s_dir in list_dir:
        path = os.path.join(source_path, s_dir)
        list_patients += [os.path.join(path,f) for f in os.listdir(path) if os.path.isdir(os.path.join(path,f))]

    if not list_patients: #if list_patients is emtly then the folder includes list of Patients
        list_patients = [os.path.join(source_path, f) for f in list_dir]
    if not list_patients: #if everything is empty asumme that this is only a single Patient
        list_patients = [source_path]

    #TODO add Other Timestamst after Preop T000 for example T001,T002, T003 ...
    assert list_patients, "Failture no single Patient, Subcategory or all Categories with Patients included given!"
    list_patients.sort()
    return list_patients

class LoadImages(Dataset):
    """
    Loading Images from the Folders

    Single Patient:  /0001
    Set of Patients: /facial_palsy/0001
                     /facial_palsy/0002
                     /facial_palsy/0003
    Set of Classes:  /data/muscle_transplant/0001
                     /data/muscle_transplant/0002
                     /data/muscle_transplant/0003
    """
    def __init__(self, path, device="cpu", cache=False, prefix_for_log=""):
        """
        Initializes the LoadImages class


        :param path: one of List above (str/Path)
        :param device: cuda device (cpu or cuda:0)
        :param cache: Cache Enable(bool)
        :param prefix_for_log: logger output prefix (str)
        """
        super().__init__()
        self.path = path
        self.prefix_for_log = prefix_for_log

        self.database = None
        self.table = "dataloader_table"
        #-#-#-#-#-#-#-#-#-Generating List of Patients for Processing-#-#-#-#-#-#-#-#-#-#-#
        self.list_patients=get_list_patients(self.path)
        self.length = len(self.list_patients)

        LOGGER.info("%sFound %s Patients. List: %s", self.prefix_for_log, self.length, self.list_patients)
        #-#-#-#-#-#-#--#-#-#-#-#-#-#-#-#-#-#-#-#--#-#-#-#-#-#-#-#-#-#-#-#-#--#-#-#-#-#-#-#

        #-#-#-#-#-#-#-#-#-#-#-Initializing Cutter for the Images-#-#-#-#-#-#-#-#-#-#-#-#-#
        self.cutter_class = Cutter.instance() #pylint: disable=no-member
        self.cutter_class.set(device, self.prefix_for_log)

        #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-Caching Data-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
        if cache:
            self.database = Database.instance() #pylint: disable=no-member
            self.database.set("cache.db", self.prefix_for_log)
            if self.database.create_db_connection() is not None:
                self.database.create_db_table(f""" CREATE TABLE IF NOT EXISTS {self.table} (
                                                id integer PRIMARY KEY,
                                                struct_img dict
                                              ); """)

                if not self.database.db_table_entries_exists(self.table):
                    LOGGER.info("%sUsing SQLite3 Database to cache the Images for faster Access! Table: %s", self.prefix_for_log, self.table)

                    results = ThreadPool(THREADPOOL_NUM_THREADS).imap(self.get_structs, range(self.length))
                    pbar = tqdm(enumerate(results), total=self.length, desc=f'{self.prefix_for_log}Caching images')
                    for idx, item in pbar:
                        self.database.insert_db(self.table, (idx, item), "(?, ?)")
                    pbar.close()
                    LOGGER.info("%sDone Writing to Database.", self.prefix_for_log)
                else:
                    LOGGER.info("%sUsing Already Cached File.", self.prefix_for_log)
            else:
                LOGGER.info("%sError! cannot create the database connection. Using Native Image Access!", self.prefix_for_log)
        else:
            LOGGER.info("%sUsing Native Image Access!", self.prefix_for_log)
        #-#-#-#-#-#-#--#-#-#-#-#-#-#-#-#-#-#-#-#--#-#-#-#-#-#-#-#-#-#-#-#-#--#-#-#-#-#-#-#
    @try_except
    def __del__(self):
        """
        Destructor: remove database
        """
        if self.database:
            self.database.delete()

    def transform_resize_and_to_tensor(self, img, idx):
        """
        Resize images and Transform images to Tensor

        :param img: Image input (Image)
        :return Transformed Image as Tensor (Tensor)
        """
        if BatchSettings.instance().hyp is not None: #pylint: disable=no-member
            valid_transforms = T.Compose([  T.Resize(BatchSettings.instance().hyp["imgsz"][idx]), #pylint: disable=no-member
                               T.ToTensor()  ])
        else:
            imgsz = { #Failsave
                "symmetry": [640, 640],
                "eye": [420, 500],
                "mouth": [640, 420],
                "forehead": [640, 300],
            }
            valid_transforms = T.Compose([  T.Resize(imgsz[idx]),
                               T.ToTensor()  ])

        return valid_transforms(img)

    #TODO Augmentation
    def augmentation(self, img_tensor):
        """
        do Augmentation

        :param img: Tensor (Tensor)
        :return Transformed Tensor (Tensor)

        Info:
        https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
        """
        LOGGER.debug("%sAugmentation is %s", self.prefix_for_log, BatchSettings.instance().get_augmentation()) #pylint: disable=no-member
        if BatchSettings.instance().get_augmentation(): #pylint: disable=no-member
            valid_transforms = T.Compose([
                T.ToPILImage(),
                T.RandomRotation(degrees=  BatchSettings.instance().hyp["RandomRotation_Degree"]   ), #pylint: disable=no-member
                #T.ColorJitter(brightness=0.1, contrast=0, saturation=0.1, hue=0),
                #T.GaussianBlur(kernel_size=(15, 15), sigma=(0.5, 3)),
                T.RandomHorizontalFlip(p=  BatchSettings.instance().hyp["RandomHorizontalFlip"] ), #pylint: disable=no-member
                T.ToTensor(),
                T.Normalize(mean=  BatchSettings.instance().hyp["Normalize"]["mean"], #pylint: disable=no-member
                            std=   BatchSettings.instance().hyp["Normalize"]["std"]) #pylint: disable=no-member
                ])
        elif BatchSettings.instance().hyp is not None: #pylint: disable=no-member
            valid_transforms = T.Compose([
                T.Normalize(mean=  BatchSettings.instance().hyp["Normalize"]["mean"], #pylint: disable=no-member
                            std=   BatchSettings.instance().hyp["Normalize"]["std"])]) #pylint: disable=no-member
        else:
            valid_transforms = T.Compose([
                T.Normalize(mean=  [0.5, 0.5, 0.5],
                            std=   [0.5, 0.5, 0.5])])
        return valid_transforms(img_tensor)

    @lru_cache(LRU_MAX_SIZE)
    def get_structs(self, idx):
        """
        Get structures from index

        :param idx: Index (int)
        :return  struct_img, struct_img_inv  (struct, struct_inv)
        """
        path = self.list_patients[idx]


        #pics = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        #assert matching_folders, 'Image Not Available at Path ' + path

        func_list = self.cutter_class.cut_wrapper()

        image_input= {
            "1_rest":                 self.transform_resize_and_to_tensor(func_list["symmetry"](path, "01"), "symmetry"  ),
            "2_lift_eyebrow":         self.transform_resize_and_to_tensor(func_list["forehead"](path, "02"), "forehead"  ),
            "3_smile_closed":         self.transform_resize_and_to_tensor(   func_list["mouth"](path, "03"), "mouth"  ),
            "4_smile_open":           self.transform_resize_and_to_tensor(   func_list["mouth"](path, "04"), "mouth"  ),
            "5_Duckface":             self.transform_resize_and_to_tensor(   func_list["mouth"](path, "05"), "mouth"  ),
            "6_eye_closed_easy":      self.transform_resize_and_to_tensor(     func_list["eye"](path, "06"), "eye"  ),
            "7_eye_closed_forced":    self.transform_resize_and_to_tensor     (func_list["eye"](path, "07"), "eye"  ),
            "8_blow_nose":            self.transform_resize_and_to_tensor(   func_list["mouth"](path, "08"), "mouth"  ),
            "9_depression_lower_lip": self.transform_resize_and_to_tensor(   func_list["mouth"](path, "09"), "mouth"  ),
        }

        struct_img = deepcopy(house_brackmann_template)
        struct_img["symmetry"]  = [  image_input["1_rest"]   ]
        struct_img["eye"]       = [  image_input["6_eye_closed_easy"], image_input["7_eye_closed_forced"]   ]
        struct_img["mouth"]     = [  image_input["3_smile_closed"], image_input["4_smile_open"], image_input["5_Duckface"],
                                     image_input["8_blow_nose"], image_input["9_depression_lower_lip"]   ]
        struct_img["forehead"]  = [  image_input["2_lift_eyebrow"]   ]

        return struct_img

    #least recently used caching via @lru_cache(LRU_MAX_SIZE) restricted!
    #augmentation needs to calculated every epoch
    def __getitem__(self, idx):
        """
        Get item operator for retrive one item from the given set

        :param idx: Index (int)
        :return  path, struct_img, struct_img_inv  (str, struct, struct_inv)
        """
        path = self.list_patients[idx]

        if self.database:
            struct_img = self.database.get_db_one(self.table, idx)[1]
        else:
            struct_img = self.get_structs(idx)

        for i in struct_img:
            for number, j in enumerate(struct_img[i]):
                struct_img[i][number] = self.augmentation(j)

        return path, struct_img

    def __len__(self):
        """
        Length of the Dataset
        """
        return self.length


class CreateDataset(Dataset):
    """
    Loading Labels and Images and build it together
    """
    def __init__(self, path='', device="cpu", cache=False, prefix_for_log=''):
        """
        Initializes the CreateDataset class

        :param path: path to the dataset (str/Path)
        :param device: cuda device (cpu or cuda:0)
        :param prefix_for_log: logger output prefix (str)
        """
        super().__init__()
        self.path = path
        self.prefix_for_log = prefix_for_log
        self.images = LoadImages(path=self.path, device=device, cache=cache, prefix_for_log=prefix_for_log)
        self.len_images = len(self.images)

        self.labels = []
        listdir = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        #Errorfix Jupyterlab: listdir.pop(0)
        #Errorfix Jupyterlab: listdir.pop(-1)

        for s_dir in listdir:
            csv_path = os.path.join(self.path, s_dir) + '.csv'
            #load CSV
            with open(csv_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile, delimiter=';')
                next(reader, None)  # skip the headers
                for row in reader:
                    self.labels.append(row)
        self.labels.sort()
        self.len_labels = len(self.labels)

        assert self.len_images == self.len_labels, f"Length of the Images ({self.len_images}) do not match to length of Labels({self.len_labels}) ."

    #least recently used caching via @lru_cache(LRU_MAX_SIZE) restricted!
    #augmentation needs to calculated every epoch
    def __getitem__(self, idx):
        """
        Get item operator for retrive one item from the given set

        :param idx: Index (int)
        :return  struct_img, struct_label  (struct, struct)
        """

        #TODO return only right pair of Images on Label (checking if same Patient)

        tmp = ["I", "II", "III", "IV", "V", "VI"]
        tmp2 = tmp[int(self.labels[idx][1]) -1]

        grade_table = house_brackmann_grading[tmp2]
        #grade_table = house_brackmann_grading[self.labels[idx][1]]

        path, struct_img = self.images[idx]


        struct_label = init_dict(house_brackmann_template, [])
        for func in struct_label:
            hb_single = house_brackmann_lookup[func]["enum"]
            struct_label[func].extend(repeat(   hb_single[grade_table[func]]  , len(struct_img[func])  ))



        return path, struct_img, struct_label

    def __len__(self):
        """
        Length of the Dataset
        """
        return self.len_images


def create_dataloader_only_images(path, device, batch_size, prefix_for_log=""):
    """
    creates and returns the DataLoader
    checks the batch size

    :param path: path to the dataset (str/Path)
    :param device: cuda device (cpu or cuda:0)
    :param batch_size: Batch Size (int)
    :param prefix_for_log: logger output prefix (str)

    :returns dataloader
    """
    dataset = LoadImages(path=path, device=device, cache=False, prefix_for_log=prefix_for_log)
    assert dataset, "No data in dataset given!"

    return DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False)


def create_dataloader(path, device, cache, batch_size, val_split=None, train_split=None):
    """
    creates and returns the DataLoader
    checks the batch size

    :param path: path to the dataset (str/Path)
    :param device: cuda device (cpu or cuda:0)
    :param cache: True or False (bool)
    :param batch_size: Batch Size (int)
    :param val_split: Factor for splitting (float, int, None)
    :param train_split: Factor for splitting (float, int, None)

    :returns dataloader
    """
    prefix_for_log="Setup Train & Validation Data: "

    with torch_distributed_zero_first():
        dataset = CreateDataset(path=path, device=device, cache=cache, prefix_for_log=prefix_for_log)

    val_loader = train_loader = None

    if val_split or train_split:
        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split, train_size=train_split)

        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        LOGGER.debug("%strain-indices=%s, val-indices=%s",prefix_for_log, train_dataset.indices, val_dataset.indices)
    else:
        train_dataset = val_dataset = dataset

    LOGGER.info("%sLength of >> Training=%s >> Validation=%s", prefix_for_log, len(train_dataset), len(val_dataset))

    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if is_process_group(LOCAL_RANK) else None
    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), sampler=sampler, shuffle=True)

    if is_master_process(RANK): #Only Process 0
        val_loader = DataLoader(val_dataset, batch_size=min(batch_size, len(val_dataset)), sampler=None, shuffle=False)
    return train_loader, val_loader
