#TODO Docstring
"""
TODO
"""




import os
import csv
from copy import deepcopy
from itertools import repeat
from functools import lru_cache
from multiprocessing.pool import ThreadPool
from tqdm import tqdm

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from .config import LOGGER, LRU_MAX_SIZE
from .cutter import Cutter
from .pytorch_utils import is_process_group, torch_distributed_zero_first #pylint: disable=import-error
from .templates import house_brackmann_template, house_brackmann_lookup, house_brackmann_grading #pylint: disable=import-error
from .database_utils import Database #pylint: disable=import-error
from .general import init_dict, try_except #pylint: disable=import-error

THREADPOOL_NUM_THREADS = min(8, os.cpu_count())  # number of multiprocessing threads

#TODO Decide which pic is for what as array
path_list = deepcopy(house_brackmann_template)
path_list["symmetry"] = [0, 0, 0, 0]
path_list["eye"] =      [0, 0, 0]
path_list["mouth"] =    [0, 0]
path_list["forehead"] = [0]

settings = {
    "cache": False,
    "nosave": False,
    "imgsz": 640
}

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

def transform_totensor_and_normalize(img):
    """
    Transform images to Tensor and do Augmentation

    :param img: Image input (Image)
    :return Transformed Image (Tensor)
    """
    valid_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    return valid_transforms(img)





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
    def __init__(self, path, device="cpu", prefix_for_log=""):
        """
        Initializes the LoadImages class


        :param path: one of List above (str/Path)
        :param imgsz: crop images to the given size (int)
        :param device: cuda device (cpu or cuda:0)
        :param cache: Cache Enable(bool)
        :param prefix_for_log: logger output prefix (str)
        """
        super().__init__()
        self.path = path
        self.prefix_for_log = prefix_for_log

        self.database = None
        self.database_file = "pythonsqlite.db"
        self.table = "dataloader_table" #TODO Split train and val??
        #-#-#-#-#-#-#-#-#-Generating List of Patients for Processing-#-#-#-#-#-#-#-#-#-#-#
        self.list_patients=get_list_patients(self.path)
        self.length = len(self.list_patients)

        LOGGER.info("%sFound %s Patients. List: %s", self.prefix_for_log, self.length, self.list_patients)
        #-#-#-#-#-#-#--#-#-#-#-#-#-#-#-#-#-#-#-#--#-#-#-#-#-#-#-#-#-#-#-#-#--#-#-#-#-#-#-#

        #-#-#-#-#-#-#-#-#-#-#-Initializing Cutter for the Images-#-#-#-#-#-#-#-#-#-#-#-#-#
        self.cutter_class = Cutter.instance() #pylint: disable=no-member
        self.cutter_class.set(device, self.prefix_for_log)

        #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-Caching Data-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
        if settings["cache"]:
            self.database = Database.instance() #pylint: disable=no-member
            self.database.set(self.database_file, self.prefix_for_log)
            if self.database.create_db_connection() is not None:
                self.database.create_db_table(f""" CREATE TABLE IF NOT EXISTS {self.table} (
                                                id integer PRIMARY KEY,
                                                struct_img dict,
                                                struct_img_inv dict
                                              ); """)

                if not self.database.db_table_entries_exists(self.table):
                    LOGGER.info("%sUsing SQLite3 Database to cache the Images for faster Access! Table: %s", self.prefix_for_log, self.table)

                    results = ThreadPool(THREADPOOL_NUM_THREADS).imap(self.get_structs, range(self.length))
                    pbar = tqdm(enumerate(results), total=self.length, desc=f'{self.prefix_for_log}Caching images')
                    for idx, item in pbar:
                        self.database.insert_db(self.table, (idx, item[0], item[1]), "(?, ?, ?)")
                    pbar.close()
                    LOGGER.info("%sDone Writing to Database.", self.prefix_for_log)
                else:
                    LOGGER.info("%sUsing Already Cached File.", self.prefix_for_log)
            else:
                settings["cache"]=False
                LOGGER.info("%sError! cannot create the database connection. Using Native Image Access!", self.prefix_for_log)
        else:
            LOGGER.info("%sFound %s Images. Using Native Image Access!", self.prefix_for_log, self.length)
        #-#-#-#-#-#-#--#-#-#-#-#-#-#-#-#-#-#-#-#--#-#-#-#-#-#-#-#-#-#-#-#-#--#-#-#-#-#-#-#
    @try_except
    def __del__(self):
        """
        Destructor: remove database
        """
        if self.database:
            conn = self.database.get_conn()
            if conn is not None:
                conn.close()
        if os.path.exists(self.database_file) and settings["nosave"]:
            os.remove(self.database_file)
            LOGGER.info("%s Deleted Database File (Cache)!", self.prefix_for_log)

    @lru_cache(LRU_MAX_SIZE)
    def get_structs(self, idx):
        """
        Get structures from index

        :param idx: Index (int)
        :return  struct_img, struct_img_inv  (struct, struct_inv)
        """
        path = self.list_patients[idx]
        pics = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        assert pics, 'Image Not Available at Path ' + path

        struct_func_list = self.cutter_class.cut_wrapper()
        struct_img = init_dict(house_brackmann_template, [])
        struct_img_inv = init_dict(house_brackmann_template, [])
        for func in struct_img:
            pics = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
            for item in path_list[func]:
                struct_img[func].append(     transform_totensor_and_normalize(struct_func_list[func](path=os.path.join(path, pics[item]), inv=False))  )
                struct_img_inv[func].append( transform_totensor_and_normalize(struct_func_list[func](path=os.path.join(path, pics[item]), inv=True))  )

        return struct_img, struct_img_inv

    def __getitem__(self, idx):
        """
        Get item operator for retrive one item from the given set

        :param idx: Index (int)
        :return  path, struct_img, struct_img_inv  (str, struct, struct_inv)
        """
        path = self.list_patients[idx]

        if self.database:
            struct_img, struct_img_inv = self.database.get_db_one(self.table, idx)
        else:
            struct_img, struct_img_inv = self.get_structs(idx)

        return path, struct_img, struct_img_inv

    def __len__(self):
        """
        Length of the Dataset
        """
        return self.length


class CreateDataset(Dataset):
    """
    Loading Labels and Images and build it together
    """
    def __init__(self, path='',device="cpu", prefix_for_log=''):
        """
        Initializes the CreateDataset class

        :param path: path to the dataset (str/Path)
        :param imgsz: crop images to the given size (int)
        :param device: cuda device (cpu or cuda:0)
        :param prefix_for_log: logger output prefix (str)
        """
        super().__init__()
        self.path = path
        self.prefix_for_log = prefix_for_log

        self.images = LoadImages(path=self.path, device=device, prefix_for_log=prefix_for_log)
        self.len_images = len(self.images)

        self.labels = []
        listdir = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
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

    def __getitem__(self, idx):
        """
        Get item operator for retrive one item from the given set

        :param idx: Index (int)
        :return  struct_img, struct_label  (struct, struct)
        """

        #TODO return only right pair of Images on Label (checking if same Patient)

        #TODO Extract Data and lookup
        grade_table = house_brackmann_grading[self.labels[idx][1]]

        struct_label = init_dict(house_brackmann_template, [])
        for func in struct_label:
            hb_single = house_brackmann_lookup[func]["enum"]
            hb_single_all_tensors = torch.eye(len(hb_single))
            struct_label[func].extend(repeat(   hb_single_all_tensors[hb_single[grade_table[func]]]  , len(path_list[func])  ))



        path, struct_img, struct_img_inv = self.images[idx]

        return path, struct_img, struct_img_inv, struct_label

    def __len__(self):
        """
        Length of the Dataset
        """
        return self.len_images


def create_dataloader_only_images(path, imgsz, params, prefix_for_log=""):
    """
    creates and returns the DataLoader
    checks the batch size

    :param path: path to the dataset (str/Path)
    :param imgsz: crop images to the given size (int)
    :param device: cuda device (cpu or cuda:0)
    :param prefix_for_log: logger output prefix (str)

    :returns dataloader
    """
    device, batch_size = params
    settings["imgsz"] = imgsz

    dataset = LoadImages(path=path, device=device, prefix_for_log=prefix_for_log)
    assert dataset, "No data in dataset given!"

    batch_size = min(batch_size, len(dataset))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def create_dataloader(path, imgsz, params, rank=-1, prefix_for_log=""):
    """
    creates and returns the DataLoader
    checks the batch size

    :param path: path to the dataset (str/Path)
    :param imgsz: crop images to the given size (int)
    :param device: cuda device (cpu or cuda:0)
    :param cache: True or False (bool)
    :param nosave: True or Fale (bool)
    :param rank: Rank of the Cluster/Tread (int)
    :param prefix_for_log: logger output prefix (str)

    :returns dataloader
    """
    device, settings["cache"], settings["nosave"], batch_size = params
    settings["imgsz"] = imgsz

    with torch_distributed_zero_first(rank):
        dataset = CreateDataset(path=path, device=device, prefix_for_log=prefix_for_log)
    batch_size = min(batch_size, len(dataset))

    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_process_group(rank) else None
    loader = torch.utils.data.DataLoader
    return loader(dataset, batch_size=batch_size, sampler=sampler, shuffle=True)
