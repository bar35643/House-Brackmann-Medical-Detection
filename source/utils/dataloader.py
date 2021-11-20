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
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

from .config import LOGGER, LRU_MAX_SIZE, RANK, LOCAL_RANK, THREADPOOL_NUM_THREADS
from .cutter import Cutter
from .database_utils import Database
from .pytorch_utils import is_process_group, is_master_process, torch_distributed_zero_first #pylint: disable=import-error
from .templates import house_brackmann_template, house_brackmann_lookup, house_brackmann_grading #pylint: disable=import-error
from .general import init_dict #pylint: disable=import-error
from .decorators import try_except #pylint: disable=import-error

#TODO Decide which pic is for what as array
path_list = deepcopy(house_brackmann_template)
path_list["symmetry"] = [0, 0, 0, 0]
path_list["eye"] =      [0, 0, 0]
path_list["mouth"] =    [0, 0]
path_list["forehead"] = [0]

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
    def __init__(self, path, imgsz=640, device="cpu", cache=False, nosave=False, prefix_for_log=""):
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
        self.nosave = nosave
        self.imgsz = ((640, 640), #symmetry
                      (640, 640), #eye
                      (640, 640), #mouth
                      (640, 640)) #forehead

        self.database = None
        self.database_file = "pythonsqlite.db"
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
            self.database.set(self.database_file, self.prefix_for_log)
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
        if os.path.exists(self.database_file) and self.nosave:
            os.remove(self.database_file)
            LOGGER.info("%s Deleted Database File (Cache)!", self.prefix_for_log)

    def transform_resize_and_to_tensor(self, img, idx):
        """
        Resize images and Transform images to Tensor

        :param img: Image input (Image)
        :return Transformed Image as Tensor (Tensor)
        """
        valid_transforms = T.Compose([
            T.Resize(self.imgsz[idx]),
            T.ToTensor()
        ])
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
        valid_transforms = T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        return valid_transforms(img_tensor)

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
        for number, func in enumerate(struct_img):
            pics = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
            for item in path_list[func]:
                struct_img[func].append(     self.transform_resize_and_to_tensor(struct_func_list[func](path=os.path.join(path, pics[item])), number  )  )

        return struct_img

    @lru_cache(LRU_MAX_SIZE)
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
    def __init__(self, path='', imgsz=640, device="cpu", cache=False, nosave=False, prefix_for_log=''):
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
        self.images = LoadImages(path=self.path, imgsz=imgsz, device=device, cache=cache, nosave=nosave, prefix_for_log=prefix_for_log)
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

    @lru_cache(LRU_MAX_SIZE)
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
            struct_label[func].extend(repeat(   hb_single[grade_table[func]]  , len(path_list[func])  ))



        path, struct_img = self.images[idx]

        return path, struct_img, struct_label

    def __len__(self):
        """
        Length of the Dataset
        """
        return self.len_images


def create_dataloader_only_images(path, imgsz, device, batch_size, prefix_for_log=""):
    """
    creates and returns the DataLoader
    checks the batch size

    :param path: path to the dataset (str/Path)
    :param imgsz: crop images to the given size (int)
    :param device: cuda device (cpu or cuda:0)
    :param batch_size: Batch Size (int)
    :param prefix_for_log: logger output prefix (str)

    :returns dataloader
    """
    dataset = LoadImages(path=path, imgsz=imgsz, device=device, cache=False, nosave=False, prefix_for_log=prefix_for_log)
    assert dataset, "No data in dataset given!"

    return DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False)


def create_dataloader(path, imgsz, device, cache, nosave, batch_size, val_split):
    """
    creates and returns the DataLoader
    checks the batch size

    :param path: path to the dataset (str/Path)
    :param imgsz: crop images to the given size (int)
    :param device: cuda device (cpu or cuda:0)
    :param cache: True or False (bool)
    :param nosave: True or Fale (bool)
    :param batch_size: Batch Size (int)
    :param val_split: Factor for splitting (float, int, None)

    :returns dataloader
    """
    prefix_for_log="Setup Train & Validation Data: "

    with torch_distributed_zero_first():
        dataset = CreateDataset(path=path, imgsz=imgsz, device=device, cache=cache, nosave=nosave, prefix_for_log=prefix_for_log)

    val_loader = train_loader = None

    if val_split:
        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)

        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
    else:
        train_dataset = val_dataset = dataset

    LOGGER.info("%sLength of >> Training=%s >> Validation=%s >> Total=%s", prefix_for_log, len(train_dataset), len(val_dataset), len(dataset))

    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if is_process_group(LOCAL_RANK) else None
    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), sampler=sampler, shuffle=True)

    if is_master_process(RANK): #Only Process 0
        val_loader = DataLoader(val_dataset, batch_size=min(batch_size, len(val_dataset)), sampler=None, shuffle=False)
    return train_loader, val_loader
