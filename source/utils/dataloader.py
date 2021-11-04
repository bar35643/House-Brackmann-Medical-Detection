#TODO Docstring
"""
TODO
"""




import os
import csv
from copy import deepcopy


import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from .config import LOGGER
from .cutter import Cutter
from .pytorch_utils import is_process_group #pylint: disable=import-error
from .templates import house_brackmann_template, house_brackmann_lookup, house_brackmann_grading, allowed_fn #pylint: disable=import-error


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
    def __init__(self, path, imgsz=640, device="cpu", prefix_for_log=""):
        """
        Initializes the LoadImages class


        :param path: one of List above (str/Path)
        :param imgsz: crop images to the given size (int)
        :param device: cuda device (cpu or cuda:0)
        :param prefix_for_log: logger output prefix (str)
        """
        super().__init__()
        self.path = path
        self.imgsz = imgsz
        self.device = device
        self.prefix_for_log = prefix_for_log

        #-#-#-#-#-#-#-#-#-Generating List of Patients for Processing-#-#-#-#-#-#-#-#-#-#-#
        self.list_patients=[]
        list_dir = [f for f in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, f))]
        for s_dir in list_dir:
            path = os.path.join(self.path, s_dir)
            self.list_patients += [os.path.join(path,f) for f in os.listdir(path) if os.path.isdir(os.path.join(path,f))]

        if not self.list_patients: #if list_patients is emtly then the folder includes list of Patients
            self.list_patients = [os.path.join(self.path, f) for f in list_dir]
        if not self.list_patients: #if everything is empty asumme that this is only a single Patient
            self.list_patients = [self.path]

        #TODO add Other Timestamst after Preop T000 for example T001,T002, T003 ...

        assert self.list_patients, "Failture no single Patient, Subcategory or all Categories with Patients included given!"
        self.list_patients.sort()
        self.length = len(self.list_patients)

        LOGGER.info("%sFound %s Patients. List: %s", self.prefix_for_log, self.length, self.list_patients)
        #-#-#-#-#-#-#--#-#-#-#-#-#-#-#-#-#-#-#-#--#-#-#-#-#-#-#-#-#-#-#-#-#--#-#-#-#-#-#-#

        #-#-#-#-#-#-#-#-#-#-#-Initializing Cutter for the Images-#-#-#-#-#-#-#-#-#-#-#-#-#
        self.cutter_class = Cutter(device=device, prefix_for_log=prefix_for_log)
        #-#-#-#-#-#-#--#-#-#-#-#-#-#-#-#-#-#-#-#--#-#-#-#-#-#-#-#-#-#-#-#-#--#-#-#-#-#-#-#

    def transform_image(self, img):
        """
        Transform images to Tensor and do Augmentation

        :param img: Image input (Image)
        :return Transformed Image (Tensor)
        """
        #TODO Augmentation
        valid_transforms = T.Compose([
            T.Resize(self.imgsz),
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        return valid_transforms(img)

    def __getitem__(self, idx):
        """
        Get item operator for retrive one item from the given set

        :param idx: Index (int)
        :return  path, struct_img, struct_img_inv  (str, struct, struct_inv)
        """
        path = self.list_patients[idx]

        pics = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        assert pics, 'Image Not Available at Path ' + path
        #print(pics) #TODO Decide which pic is for what as array

        struct_func_list = self.cutter_class.cut_wrapper()

        struct_img = deepcopy(house_brackmann_template)
        struct_img["symmetry"] = [struct_func_list["symmetry"](path=os.path.join(path, pics[0])),
                                  struct_func_list["symmetry"](path=os.path.join(path, pics[0])),
                                  struct_func_list["symmetry"](path=os.path.join(path, pics[0]))]
        struct_img["eye"] =      [struct_func_list["eye"](path=os.path.join(path, pics[0]))]
        struct_img["mouth"] =    [struct_func_list["mouth"](path=os.path.join(path, pics[0]))]
        struct_img["forehead"] = [struct_func_list["forehead"](path=os.path.join(path, pics[0]))]

        #TODO Fix
        struct_img_inv = deepcopy(house_brackmann_template)
        struct_img_inv["symmetry"] = [struct_func_list["symmetry"](path=os.path.join(path, pics[0]), inv=True),
                                      struct_func_list["symmetry"](path=os.path.join(path, pics[0]), inv=True),
                                      struct_func_list["symmetry"](path=os.path.join(path, pics[0]), inv=True)]
        struct_img_inv["eye"] =      [struct_func_list["eye"](path=os.path.join(path, pics[0]), inv=True)]
        struct_img_inv["mouth"] =    [struct_func_list["mouth"](path=os.path.join(path, pics[0]), inv=True)]
        struct_img_inv["forehead"] = [struct_func_list["forehead"](path=os.path.join(path, pics[0]), inv=True)]


        #assert img0 is not None, 'Image Not Found ' + path
        #print(f'image {self.count}/{self.nf} {path}: ', end='')

        for i in struct_img:
            for j, img in enumerate(struct_img[i]):
                struct_img[i][j] = self.transform_image(img)
            for j, img_inv in enumerate(struct_img_inv[i]):
                struct_img_inv[i][j] = self.transform_image(img_inv)

        return path, struct_img, struct_img_inv

    def __len__(self):
        """
        Length of the Dataset
        """
        return self.length






class LoadLabels(Dataset):
    """
    Loading Labels from the .csv File
    """
    def __init__(self, path, prefix_for_log=""):
        """
        Initializes the LoadLabels class


        :param path: Path to the .csv (str/Path)
        :param prefix_for_log: logger output prefix (str)
        """
        super().__init__()
        self.path = path + '.csv'
        print(self.path)
        self.prefix_for_log = prefix_for_log


        #load CSV
        self.list = []
        with open(self.path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            next(reader, None)  # skip the headers
            for row in reader:
                self.list.append(row)
        self.list.sort()
        self.length = len(self.list)


    def __getitem__(self, idx):
        """
        Get item operator for retrive one item from the given set

        :param idx: Index (int)
        :return  item_name, struct_label  (str, struct)
        """
        item_name = self.list[idx]
        #print(item_name)

        #TODO Extract Data and lookup
        grade_table = house_brackmann_grading[item_name[1]]

        struct_label = deepcopy(house_brackmann_template)
        for i in allowed_fn:
            #Create Tensor with the point e.g x=3 with Value 1 for the right label [0, 0, 0, 1, 0, 0]
            hb_single = house_brackmann_lookup[i]["enum"]
            hb_single_all_tensors = torch.eye(len(hb_single))
            struct_label[i] = hb_single_all_tensors[hb_single[grade_table[i]]]

        return item_name, struct_label

    def __len__(self):
        """
        Length of the Dataset
        """
        return self.length  # number of items




class CreateDataset(Dataset):
    """
    Loading Labels and Images and build it together
    """
    def __init__(self, path='', imgsz=640, device="cpu", prefix_for_log=''):
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
        self.images = []
        self.labels = []

        self.images = LoadImages(path=self.path, imgsz=imgsz, device=device, prefix_for_log=prefix_for_log)
        self.len_images = len(self.images)

        self.listdir = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        for s_dir in self.listdir:
            self.labels += LoadLabels(path=os.path.join(path, s_dir), prefix_for_log=prefix_for_log)
        self.len_labels = len(self.labels)

        assert not self.len_images != self.len_labels, f"Length of the Images ({self.len_images}) do not match to length of Labels({self.len_labels}) ."

    def __getitem__(self, idx):
        """
        Get item operator for retrive one item from the given set

        :param idx: Index (int)
        :return  struct_img, struct_label  (struct, struct)
        """

        #TODO return only right pair of Images on Label (checking if same Patient)

        i_name, struct_img, struct_img_inv = self.images[idx]
        l_name, struct_label = self.labels[idx]

        return struct_img, struct_img_inv, struct_label

    def __len__(self):
        """
        Length of the Dataset
        """
        return self.len_images

def create_dataloader(path, imgsz, device, batch_size,
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
    dataset = CreateDataset(path=path, imgsz=imgsz, device=device, prefix_for_log=prefix_for_log)


    batch_size = min(batch_size, len(dataset))

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_process_group(rank) else None
    loader = torch.utils.data.DataLoader

    dataloader = loader(dataset,batch_size=batch_size,sampler=sampler,num_workers=num_workers,collate_fn=None, pin_memory=False)
    return dataloader
