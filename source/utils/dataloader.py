#TODO Docstring
"""
TODO
"""




import os
import csv
from copy import deepcopy
from itertools import repeat
from multiprocessing.pool import ThreadPool
from tqdm import tqdm

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from .config import LOGGER
from .cutter import Cutter
from .pytorch_utils import is_process_group #pylint: disable=import-error
from .templates import house_brackmann_template, house_brackmann_lookup, house_brackmann_grading #pylint: disable=import-error
from .database_utils import Database #pylint: disable=import-error
from .general import init_dict, try_except #pylint: disable=import-error

THREADPOOL_NUM_THREADS = min(8, os.cpu_count())  # number of multiprocessing threads


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

class LoadImagesAsStruct(Dataset):
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
    def __init__(self, path, imgsz=640, device="cpu", prefix_for_log=""):
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
        self.imgsz = imgsz
        self.device = device
        self.prefix_for_log = prefix_for_log

        #-#-#-#-#-#-#-#-#-Generating List of Patients for Processing-#-#-#-#-#-#-#-#-#-#-#
        self.list_patients=get_list_patients(self.path)
        self.length = len(self.list_patients)

        LOGGER.info("%sFound %s Patients. List: %s", self.prefix_for_log, self.length, self.list_patients)
        #-#-#-#-#-#-#--#-#-#-#-#-#-#-#-#-#-#-#-#--#-#-#-#-#-#-#-#-#-#-#-#-#--#-#-#-#-#-#-#

        #-#-#-#-#-#-#-#-#-#-#-Initializing Cutter for the Images-#-#-#-#-#-#-#-#-#-#-#-#-#
        self.cutter_class = Cutter.instance() #pylint: disable=no-member
        self.cutter_class.set(device, self.prefix_for_log)
        #-#-#-#-#-#-#--#-#-#-#-#-#-#-#-#-#-#-#-#--#-#-#-#-#-#-#-#-#-#-#-#-#--#-#-#-#-#-#-#

    def get_structs(self, idx):
        """
        Get structures from index

        :param idx: Index (int)
        :return  struct_img, struct_img_inv  (struct, struct_inv)
        """
        path = self.list_patients[idx]
        pics = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        assert pics, 'Image Not Available at Path ' + path
        #print(pics) #TODO Decide which pic is for what as array

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
        struct_img, struct_img_inv = self.get_structs(idx)

        return path, struct_img, struct_img_inv

    def __len__(self):
        """
        Length of the Dataset
        """
        return self.length







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
    def __init__(self, path, func='', imgsz=640, device="cpu", cache=False, nosave=False, prefix_for_log=""):
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
        self.imgsz = imgsz
        self.device = device
        self.cache = cache
        self.prefix_for_log = prefix_for_log
        self.func = func
        self.nosave = nosave

        self.database = None
        self.database_file = "pythonsqlite.db"
        self.table = "dataloader_table_"+func #TODO Split train and val??

        #-#-#-#-#-#-#-#-#-Generating List of Patients for Processing-#-#-#-#-#-#-#-#-#-#-#
        self.list_patients=get_list_patients(self.path)


        self.items = []
        for i in self.list_patients:
            pics = [f for f in os.listdir(i) if os.path.isfile(os.path.join(i, f))]
            for j in path_list[self.func]:
                self.items.append(os.path.join(i, pics[j]))
        self.length = len(self.items)

        #-#-#-#-#-#-#-#-#-#-#-Initializing Cutter for the Images-#-#-#-#-#-#-#-#-#-#-#-#-#
        self.cutter_class = Cutter.instance() #pylint: disable=no-member
        self.cutter_class.set(self.device, self.prefix_for_log)
        #-#-#-#-#-#-#--#-#-#-#-#-#-#-#-#-#-#-#-#--#-#-#-#-#-#-#-#-#-#-#-#-#--#-#-#-#-#-#-#

        #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-Caching Data-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
        if self.cache:
            self.database = Database.instance() #pylint: disable=no-member
            self.database.set(self.database_file, self.prefix_for_log)

            if self.database.create_db_connection() is not None:
                self.database.create_db_table(f""" CREATE TABLE IF NOT EXISTS {self.table} (
                                                id integer PRIMARY KEY,
                                                img np_array,
                                                img_inv np_array
                                              ); """)

                if not self.database.db_table_entries_exists(self.table):
                    LOGGER.info("%sUsing SQLite3 Database to cache the Images for faster Access! Table: %s", self.prefix_for_log, self.table)

                    results = ThreadPool(THREADPOOL_NUM_THREADS).imap(self.get_item, range(self.length))
                    pbar = tqdm(enumerate(results), total=self.length, desc=f'{self.prefix_for_log}Caching images for {func}')
                    for idx, item in pbar:
                        self.database.insert_db(self.table, (idx, item[0], item[1]), "(?, ?, ?)")
                    pbar.close()
                    LOGGER.info("%sDone Writing to Database.", self.prefix_for_log)
                else:
                    LOGGER.info("%sUsing Already Cached File. (func %s)", self.prefix_for_log, func)
            else:
                self.cache=False
                LOGGER.info("%sError! cannot create the database connection. Using Native Image Access!", self.prefix_for_log)
        else:
            LOGGER.info("%sFound %s Images for %s. Using Native Image Access!", self.prefix_for_log, self.length, func)
        #-#-#-#-#-#-#--#-#-#-#-#-#-#-#-#-#-#-#-#--#-#-#-#-#-#-#-#-#-#-#-#-#--#-#-#-#-#-#-#
    @try_except
    def __del__(self):
        """
        Destructor: remove database
        """
        conn = self.database.get_conn()
        if conn is not None:
            conn.close()
        if os.path.exists(self.database_file) and self.nosave:
            os.remove(self.database_file)
            LOGGER.info("%s Deleted Database File (Cache)!", self.prefix_for_log)

    def get_item(self, idx):
        """
        Get item operator for retrive one item native

        :param idx: Index (int)
        :return  img (np.array)
        """
        func = self.cutter_class.cut_wrapper()[self.func]
        path = self.items[idx]

        img = func(path=os.path.join(path), inv=False)
        img_inv = func(path=os.path.join(path), inv=True)
        return img, img_inv

    def __getitem__(self, idx):
        """
        Get item operator for retrive one item

        :param idx: Index (int)
        :return  Tensor (torch.tensor)
        """
        if self.database:
            img, img_inv = self.database.get_db_one(self.table, idx)
        else:
            img, img_inv = self.get_item(idx)
        return transform_totensor_and_normalize(img), transform_totensor_and_normalize(img_inv)

    def __len__(self):
        """
        Length of the Dataset
        """
        return self.length




class LoadLabels(Dataset):
    """
    Loading Labels from the .csv File
    """
    def __init__(self, path, func='', prefix_for_log=""):
        """
        Initializes the LoadLabels class


        :param path: Path to the .csv (str/Path)
        :param prefix_for_log: logger output prefix (str)
        """
        super().__init__()
        self.path = path + '.csv'
        self.prefix_for_log = prefix_for_log
        self.func = func


        #load CSV
        self.list = []
        with open(self.path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            next(reader, None)  # skip the headers
            for row in reader:
                self.list.extend(repeat(row, len(path_list[self.func])))
        self.list.sort()
        self.length = len(self.list)


    def __getitem__(self, idx):
        """
        Get item operator for retrive one item from the given set

        :param idx: Index (int)
        :return  item_name, struct_label  (str, struct)
        """
        item_name = self.list[idx]

        #TODO Extract Data and lookup
        grade_table = house_brackmann_grading[item_name[1]]

        hb_single = house_brackmann_lookup[self.func]["enum"]
        hb_single_all_tensors = torch.eye(len(hb_single))

        return hb_single_all_tensors[hb_single[grade_table[self.func]]]

    def __len__(self):
        """
        Length of the Dataset
        """
        return self.length  # number of items




class CreateDataset(Dataset):
    """
    Loading Labels and Images and build it together
    """
    def __init__(self, path='', func='', imgsz=640, device="cpu", cache=False, nosave=False, prefix_for_log=''):
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

        self.images = LoadImages(path=self.path,func=func, imgsz=imgsz, device=device, cache=cache, nosave=nosave, prefix_for_log=prefix_for_log)
        self.len_images = len(self.images)

        self.labels = []
        self.listdir = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        for s_dir in self.listdir:
            self.labels += LoadLabels(path=os.path.join(path, s_dir),func=func, prefix_for_log=prefix_for_log)

        self.len_labels = len(self.labels)
    def __getitem__(self, idx):
        """
        Get item operator for retrive one item from the given set

        :param idx: Index (int)
        :return  struct_img, struct_label  (struct, struct)
        """

        #TODO return only right pair of Images on Label (checking if same Patient)

        img, img_inv = self.images[idx]
        label = self.labels[idx]

        return img, img_inv, label

    def __len__(self):
        """
        Length of the Dataset
        """
        return self.len_images


def create_dataloader_only_images(path, imgsz, device, batch_size, prefix_for_log=""):
    """
    creates and returns the DataLoader
    checks the batch size
    checks the num workers

    :param path: path to the dataset (str/Path)
    :param imgsz: crop images to the given size (int)
    :param device: cuda device (cpu or cuda:0)
    :param worker: num worker for loading the dataset (int)
    :param prefix_for_log: logger output prefix (str)

    :returns dataloader
    """

    dataset = LoadImagesAsStruct(path=path, imgsz=imgsz, device=device, prefix_for_log=prefix_for_log)
    assert dataset, "No data in dataset given!"

    batch_size = min(batch_size, len(dataset))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def create_dataloader(path, imgsz, device, cache, nosave, batch_size,
                      rank=-1, workers=8, prefix_for_log=""):
    """
    creates and returns the DataLoader
    checks the batch size
    checks the num workers

    :param path: path to the dataset (str/Path)
    :param imgsz: crop images to the given size (int)
    :param device: cuda device (cpu or cuda:0)
    :param rank: Rank of the Cluster/Tread (int)
    :param worker: num worker for loading the dataset (int)
    :param prefix_for_log: logger output prefix (str)

    :returns dataloader
    """
    #TODO ?? Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    #dataset = CreateDataset(path=path, imgsz=imgsz, device=device, cache=cache, prefix_for_log=prefix_for_log)
    struct_dataloader = deepcopy(house_brackmann_template)
    #for idx in struct_dataloader:
    for item  in struct_dataloader:
        dataset = CreateDataset(path=path, func=item, imgsz=imgsz, device=device, cache=cache, nosave=nosave, prefix_for_log=prefix_for_log)
        batch_size = min(batch_size, len(dataset))
        #num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 1, workers])  # number of worker

        #TODO sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_process_group(rank) else None
        #TODO num_worker
        loader = torch.utils.data.DataLoader
        struct_dataloader[item] = loader(dataset, batch_size=batch_size)
    return struct_dataloader
