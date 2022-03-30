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
# - 2022-03-12 Final Version 1.0.0 (~Raphael Baumann)
"""

#https://docs.python.org/3/library/unittest.html
#https://ongspxm.gitlab.io/blog/2016/11/assertraises-testing-for-errors-in-unittest/

#pylint: disable=invalid-name, no-member, too-few-public-methods, no-self-use, line-too-long, unused-variable

import argparse
import unittest
import logging
import threading
import time
from copy import deepcopy
import os
import sys
from pathlib import Path
from contextlib import contextmanager

from PIL import Image

import torch
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch import nn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.config import LOGGER #pylint: disable=import-error
from utils.general import check_python, check_requirements, init_dict, merge_two_dicts, check_online, check_version, get_key_from_dict, create_workspace, delete_folder_content, increment_path #pylint: disable=import-error
from utils.pytorch_utils import select_device, is_process_group, is_master_process, select_optimizer_and_scheduler #pylint: disable=import-error
from utils.decorators import try_except_none, try_except, thread_safe #pylint: disable=import-error
from utils.singleton import Singleton #pylint: disable=import-error
from utils.dataloader import ImbalancedDatasetSampler, CreateDataloader, create_dataloader_only_images, LoadImages, CreateDataset, transform_resize_and_to_tensor, get_list_patients #pylint: disable=import-error
from utils.errorimports import read_heif #pylint: disable=import-error
from utils.database_utils import Database #pylint: disable=import-error
from utils.automata import one, two, three, four, five, six, hb_automata #pylint: disable=import-error
from utils.specs import validate_yaml_config, validate_file  #pylint: disable=import-error
from utils.argparse_utils import restricted_val_split  #pylint: disable=import-error
from utils.plotting import AverageMeter, Plotting #pylint: disable=import-error
from utils.cutter import load_image, Cutter #pylint: disable=import-error

@contextmanager
def assertNotRaises(exc_type):
    """
    function for testing if no Assertion  Raises
    """
    try:
        yield None
    except exc_type:
        raise unittest.TestCase.failureException('{} raised'.format(exc_type.__name__)) from exc_type #pylint: disable=consider-using-f-string

class TestCase(unittest.TestCase):
    """
    Test Everything Else
    """
    def test_it_does_not_raise_key_error(self):
        """
        test_it_does_not_raise_key_error
        Test if AssertNotRaises works
        """
        data = {}
        with self.assertRaises(AssertionError):
            with assertNotRaises(KeyError):
                data['missing_key'] #pylint: disable=pointless-statement

        data['missing_key'] = 123
        with assertNotRaises(KeyError):
            data['missing_key'] #pylint: disable=pointless-statement

    def test_restricted_val_split(self):
        """
        restricted_val_split
        """

        x = restricted_val_split("0.99")
        self.assertEqual(isinstance(x, float), True)
        self.assertEqual(x, 0.99)

        x = restricted_val_split("55")
        self.assertEqual(isinstance(x, int), True)
        self.assertEqual(x, 55)

        x = restricted_val_split("None")
        self.assertEqual(isinstance(x, type(None)), True)
        self.assertEqual(x, None)

        with self.assertRaises(argparse.ArgumentTypeError):
            x = restricted_val_split("HAHA")

        with self.assertRaises(argparse.ArgumentTypeError):
            x = restricted_val_split("1.1")

class TestCaseGeneral(unittest.TestCase):
    """
    Test Casees of the general.py file
    """
    def test_merge_two_dicts(self):
        """
        test_merge_two_dicts
        Testing Merging two dicts
        """
        a = {"test1": None,
             "test2": None,}

        b = {"test3": [],
             "test4": [],}

        c = {"test1": None,
             "test2": None,
             "test3": [],
             "test4": [],}

        self.assertEqual(merge_two_dicts(a,b), c)

    def test_get_key_from_dict(self):
        """
        test_get_key_from_dict
        Testing if you get the key from dict
        """
        a = {"test1": 1,
             "test2": 2,}

        self.assertEqual(get_key_from_dict(a, 1), "test1")
        self.assertEqual(get_key_from_dict(a, 2), "test2")
        self.assertNotEqual(get_key_from_dict(a, 0), "test1")
        self.assertNotEqual(get_key_from_dict(a, 0), "test2")
    def test_init_dict(self):
        """
        test_init_dict
        Testing if initializing dictionary correct
        """
        a = {"test1": None,
             "test2": None,}

        b = {"test1": [],
             "test2": [],}

        c = {"test1": None,
             "test2": None,}

        self.assertEqual(init_dict(a, []), b)
        self.assertNotEqual(init_dict(a, []), c)

    def test_check_online(self):
        """
        test_check_onlines
        Testing Check online if True
        """

        self.assertEqual(check_online(), True)

    def test_check_version(self):
        """
        test_check_version
        Testing Check Version
        Expects: No Assertion raises and Assertion Raises
        """

        check_version(current="2.0.0", minimum="1.0.0")

        with self.assertRaises(AssertionError):
            check_version(current="1.0.0", minimum="2.0.0")


    def test_check_python(self):
        """
        test_check_python
        Testing if no exeption raises

        Enters: a Version that is smaller than the Version install on the Computer
        Expects: No Assertion raises and Assertion Raises
        """
        check_python(minimum="3.7.0")

        with self.assertRaises(AssertionError):
            check_python(minimum="3.9.0")

    def test_create_delete_increment_workspace(self):
        """
        test_create_delete_increment_workspace
        Testing if no exeption raises

        Enters: a Version that is smaller than the Version install on the Computer
        Expects: No Assertion raises and Assertion Raises
        """
        ws = create_workspace()

        with assertNotRaises(AssertionError):
            ws = create_workspace()

        dir1 = increment_path(Path("./tmp/test"), exist_ok=False)
        dir1.mkdir(parents=True, exist_ok=True)  # make dir

        dir2 = increment_path(Path("./tmp/test"), exist_ok=False)
        dir2.mkdir(parents=True, exist_ok=True)  # make dir

        self.assertNotEqual(dir1, dir2)
        lst1 = os.listdir("./tmp")
        delete_folder_content("./tmp")
        lst2 = os.listdir("./tmp")
        self.assertNotEqual(lst1, lst2)

        #os.rmdir(str(ws))
        os.rmdir(Path("./tmp"))

        with self.assertRaises(FileNotFoundError):
            os.listdir(Path("./tmp"))






class TestCasePytorchUtils(unittest.TestCase):
    """
    Test Casees of the pytorch.utils.py file
    """

    def test_is_process_group(self):
        """
        test_is_process_group
        Testing process group for right return value
        """
        self.assertEqual(is_process_group(0), True)
        self.assertEqual(is_process_group(1), True)
        self.assertEqual(is_process_group(2), True)

        self.assertEqual(is_process_group(-1), False)

    def test_is_master_process(self):
        """
        test_is_master_process
        Testing process group for right return value
        """
        self.assertEqual(is_master_process(-1), True)
        self.assertEqual(is_master_process(0), True)

        self.assertEqual(is_master_process(1), False)
        self.assertEqual(is_master_process(2), False)
        self.assertEqual(is_master_process(3), False)


    def test_select_device_cpu_general(self):
        """
        test_select_device_cpu_general
        Testing if the device sets correctly part 1

        Enters: select device "cpu"
        Expects: set to "cpu"
        """
        self.assertEqual(select_device("cpu", 16), torch.device("cpu"))

    def test_select_device_gpu_cuda_not_available(self):
        """
        test_select_device_gpu_cuda_not_available
        Testing if the device sets correctly part 2

        Enters: select device "0" if cuda is not available
        Expects: Assertion Raises
        """
        if not torch.cuda.is_available():
            with self.assertRaises(AssertionError):
                select_device("0", 16)

    def test_select_device_gpu_cuda_available(self):
        """
        test_select_device_gpu_cuda_available
        Testing if the device sets correctly part 3

        Enters: select device "0" if cuda is available
        Expects: set device to cuda:0
        """
        if torch.cuda.is_available():
            self.assertEqual(select_device("0", 16), torch.device("cuda:0"))

    def test_select_device_more_gpu_cuda_available(self):
        """
        test_select_device_more_gpu_cuda_available
        Testing if the device sets correctly part 4

        Enters: select more than one device
        Expects: set device to cuda:0
        """
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.assertEqual(select_device("0, 1", 16), torch.device("cuda:0"))


    def test_select_device_cuda_available_more_batchsize_wrong(self):
        """
        test_select_device_cuda_available_more_batchsize_wrong
        Testing if the device sets correctly part 5

        Enters: select more than one device and batch size is not correct
        Expects: Assertion Raises
        """
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            with self.assertRaises(AssertionError):
                select_device("0, 1", 1)


    def test_select_optimizer_and_scheduler(self):
        """
        test_select_optimizer_and_scheduler
        Test optimizer and scheduler for function and type
        """
        with assertNotRaises(Exception):
            yml_hyp = validate_file("./models/hyp.yaml")
            neural_net = resnet18(pretrained=False)
            epoch = 25
            sh, opt = select_optimizer_and_scheduler(yml_hyp, neural_net, epoch)

            self.assertEqual(isinstance(opt, torch.optim.SGD), True)
            self.assertEqual(isinstance(sh, torch.optim.lr_scheduler.ChainedScheduler), True)




class TestCasePatterns(unittest.TestCase):
    """
    Test Casees of the singleton.py and decorators.py file
    """
    def test_singleton(self):
        """
        Testing signleton Pattern
        Explicit return type check and instance check
        """

        @Singleton
        class Test1():
            """
            test Class
            """
            def __init__(self):
                pass
            def __call__(self):
                return True


        class Test2():
            """
            test Class
            """
            def __init__(self):
                pass
            def __call__(self):
                return True



        with self.assertRaises(TypeError):
            Test1()()

        f = Test1.instance()
        g = Test1.instance() # Returns already created instance
        h = Test2()

        self.assertEqual(f is g, True)
        self.assertEqual(f is not h, True)

    def test_try_expect(self):
        """
        test_try_expect
        Explicit return type check
        """
        @try_except
        def func_test(val:bool ):
            assert val, "Test"
            return "abc"

        self.assertEqual(func_test(True), None)
        self.assertEqual(func_test(False), None)

    def test_try_except_none(self):
        """
        test_try_except_none
        Explicit return type check
        """
        @try_except_none
        def func_test(val:bool ):
            assert val, "Test"
            return "abc"

        self.assertEqual(func_test(True), "abc")
        self.assertEqual(func_test(False), None)

    def test_thread_safe(self):
        """
        test_thread_safe
        Test thread safe function
        """
        threads_name = []
        @thread_safe
        def func_test1():
            time.sleep(1)
            threads_name.append(threading.currentThread().getName())

        threads = []
        for _ in range(9):
            t = threading.Thread(target=func_test1)
            threads.append(t)
            t.start()

        time.sleep(15)
        tmp = deepcopy(threads_name)
        tmp.sort()
        self.assertEqual(threads_name, tmp)

    def test_thread_not_safe(self):
        """
        test_thread without thread_safe
        Test thread safe function
        """
        threads_name = []
        def func_test2():
            time.sleep(1)
            threads_name.append(threading.currentThread().getName())

        threads = []
        for _ in range(9):
            t = threading.Thread(target=func_test2)
            threads.append(t)
            t.start()

        time.sleep(15)
        tmp = deepcopy(threads_name)
        tmp.sort()
        self.assertNotEqual(threads_name, tmp)


class TestCase_Errorimport(unittest.TestCase):
    """
    Test Casees of errorimports.py
    """
    def test_read_heif(self):
        """
        test_read_heif
        Tests Errorimport helper
        """
        with self.assertRaises(AssertionError):
            read_heif(1, 2, 3, 4, 5)

class TestCase_Automata(unittest.TestCase):
    """
    Test Casees of config.py
    """
    def test_one(self):
        """
        test_one
        Tests Automata for correct function
        """
        x1, x2 = one(symmetry=0, eye=0, mouth=0, forehead=0)
        self.assertEqual(x1, 0)
        self.assertEqual(x2, False)

        x1, x2 = one(symmetry=1, eye=0, mouth=0, forehead=0)
        self.assertEqual(x1, 4)
        self.assertEqual(x2, True)

        x1, x2 = one(symmetry=0, eye=1, mouth=0, forehead=0)
        self.assertEqual(x1, 3)
        self.assertEqual(x2, True)

        x1, x2 = one(symmetry=0, eye=0, mouth=3, forehead=0)
        self.assertEqual(x1, 5)
        self.assertEqual(x2, True)

        x1, x2 = one(symmetry=0, eye=0, mouth=1, forehead=0)
        self.assertEqual(x1, 1)
        self.assertEqual(x2, True)

        x1, x2 = one(symmetry=0, eye=0, mouth=0, forehead=1)
        self.assertEqual(x1, 2)
        self.assertEqual(x2, True)

    def test_two(self):
        """
        test_two
        Tests Automata for correct function
        """
        x1, x2 = two(symmetry=0, eye=0, mouth=0, forehead=0)
        self.assertEqual(x1, 1)
        self.assertEqual(x2, False)

        x1, x2 = two(symmetry=0, eye=0, mouth=2, forehead=0)
        self.assertEqual(x1, 3)
        self.assertEqual(x2, True)

    def test_three(self):
        """
        test_three
        Tests Automata for correct function
        """
        x1, x2 = three(symmetry=0, eye=0, mouth=0, forehead=0)
        self.assertEqual(x1, 2)
        self.assertEqual(x2, False)

        x1, x2 = three(symmetry=0, eye=0, mouth=0, forehead=2)
        self.assertEqual(x1, 3)
        self.assertEqual(x2, True)

        x1, x2 = three(symmetry=0, eye=0, mouth=2, forehead=0)
        self.assertEqual(x1, 3)
        self.assertEqual(x2, True)

    def test_four(self):
        """
        test_four
        Tests Automata for correct function
        """
        x1, x2 = four(symmetry=0, eye=0, mouth=0, forehead=0)
        self.assertEqual(x1, 3)
        self.assertEqual(x2, False)

        x1, x2 = four(symmetry=0, eye=1, mouth=0, forehead=0)
        self.assertEqual(x1, 3)
        self.assertEqual(x2, False)

    def test_five(self):
        """
        test_five
        Tests Automata for correct function
        """
        x1, x2 = five(symmetry=0, eye=0, mouth=0, forehead=0)
        self.assertEqual(x1, 4)
        self.assertEqual(x2, False)

        x1, x2 = five(symmetry=2, eye=0, mouth=0, forehead=0)
        self.assertEqual(x1, 5)
        self.assertEqual(x2, True)

    def test_six(self):
        """
        test_one
        Tests Automata for correct function
        """
        x1, x2 = six(symmetry=0, eye=0, mouth=0, forehead=0)
        self.assertEqual(x1, 5)
        self.assertEqual(x2, False)

        x1, x2 = six(symmetry=0, eye=1, mouth=0, forehead=0)
        self.assertEqual(x1, 5)
        self.assertEqual(x2, False)


    def test_hb_automata(self):
        """
        test_hb_automata
        Tests Automata for correct function
        """
        state = []
        to_compare = [0, 2, 3, 2, 2, 1, 2, 3, 2, 2, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 1, 2, 3, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 3, 3, 3, 3, 3, 0, 2, 3, 2, 2, 1, 2, 3, 2, 2, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 1, 2, 3, 2, 2, 0, 2, 3, 2, 2, 1, 2, 3, 2, 2, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 1, 2, 3, 2, 2, 0, 2, 3, 2, 2, 1, 2, 3, 2, 2, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 1, 2, 3, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4]
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    for l in range(5):
                        state.append(hb_automata(i, j, k, l))
        self.assertEqual(state, to_compare)


class TestCaseDataloader(unittest.TestCase):
    """
    Test Casees of dataloader.py
    """
    def test_transform_resize_and_to_tensor(self):
        """
        test_transform_resize_and_to_tensor

        Tests if the shape is correct after resize and to tensor
        """
        img = Image.new('RGB', (100, 100))
        tensor0 = transform_resize_and_to_tensor(img, "symmetry")
        tensor1 = torch.zeros(3, 640, 640)

        self.assertEqual(list(tensor0.shape), list(tensor1.shape))

    def test_get_list_patients(self):
        """
        test_get_list_patients

        Tests if the returned list is the same
        """
        lst = get_list_patients('../../test_data')
        lst1 = ['../../test_data\\Faziale_Reanimation\\0001', '../../test_data\\Faziale_Reanimation\\0002', '../../test_data\\Faziale_Reanimation\\0003', '../../test_data\\Faziale_Reanimation\\0004', '../../test_data\\Faziale_Reanimation\\0005', '../../test_data\\Faziale_Reanimation\\0006', '../../test_data\\Muskeltransplantation\\0001', '../../test_data\\Muskeltransplantation\\0002', '../../test_data\\Muskeltransplantation\\0003', '../../test_data\\Muskeltransplantation\\0004', '../../test_data\\Muskeltransplantation\\0005', '../../test_data\\Muskeltransplantation\\0006']
        self.assertEqual(lst, lst1)

    def test_load_images(self):
        """
        test_load_images
        Test if the LoadImages Dataset is loaded with the correct length and the items has the correst tpye
        """
        tst = LoadImages(path='../../test_data', prefix_for_log='')
        self.assertEqual(len(tst), 12)

        tst = LoadImages(path='../../test_data/Muskeltransplantation', prefix_for_log='')
        self.assertEqual(len(tst), 6)


        tst = LoadImages(path='../../test_data/Muskeltransplantation/0001', prefix_for_log='')
        self.assertEqual(len(tst), 1)


        with assertNotRaises(AssertionError):
            for i, j in tst:
                self.assertEqual(isinstance(i,str), True)
                self.assertEqual(isinstance(j,dict), True)

    def test_create_dataset(self):
        """
        test_create_dataset
        Test if the Dataset is loaded with the correct length and the items has the correst tpye
        """
        tst = CreateDataset(path='../../test_data', prefix_for_log='')
        self.assertEqual(len(tst), 12)

        with assertNotRaises(AssertionError):
            for i, j, k in tst:
                self.assertEqual(isinstance(i,str), True)
                self.assertEqual(isinstance(j,dict), True)
                self.assertEqual(isinstance(k,dict), True)

    def test_create_dataloader_only_images(self):
        """
        test_create_dataloader_only_images
        Tests the Dataloader for correct type
        """

        with assertNotRaises(AssertionError):
            tst = create_dataloader_only_images(path='../../test_data',
                                                device="cpu",
                                                batch_size=4,
                                                prefix_for_log='')
            self.assertEqual(isinstance(tst,DataLoader), True)

    def test_CreateDataloader(self):
        """
        test_CreateDataloader
        Tests the Dataloader for correct type and the Sampler for correct type
        """

        with assertNotRaises(AssertionError):
            cls = CreateDataloader(path='../../test_data',
                                   device="cpu",
                                   cache=False,
                                   batch_size=4,
                                   val_split=None,
                                   train_split=None,
                                   oversampling=False)
            tst_train, tst_val = cls.get_dataloader_func("symmetry")
            self.assertEqual(isinstance(tst_train,DataLoader), True)
            self.assertEqual(isinstance(tst_val,DataLoader), True)
            self.assertEqual(isinstance(tst_train.sampler, torch.utils.data.sampler.RandomSampler), True)

            cls = CreateDataloader(path='../../test_data',
                                   device="cpu",
                                   cache=False,
                                   batch_size=4,
                                   val_split=None,
                                   train_split=None,
                                   oversampling=True)
            tst_train, tst_val = cls.get_dataloader_func("symmetry")
            self.assertEqual(isinstance(tst_train,DataLoader), True)
            self.assertEqual(isinstance(tst_val,DataLoader), True)
            self.assertEqual(isinstance(tst_train.sampler, ImbalancedDatasetSampler), True)

class TestCaseDatabaseUtils(unittest.TestCase):
    """
    Test Casees of database_utils.py
    """
    def test_Database(self):
        """
        test_Database
        Tests the Database working properly
        """
        with assertNotRaises(AssertionError):
            x1 = Database.instance()
            x2 = Database.instance()
            self.assertEqual(x1, x2)

            self.assertEqual(x1.db_file, "cache.db")
            x1.set("haha.db", "")
            self.assertEqual(x1.db_file, "haha.db")

            tmp1 = x1.create_db_connection()
            tmp2 = x1.get_conn()
            self.assertEqual(tmp1, tmp2)

            table = "dataloader_table"
            x1.create_db_table(f""" CREATE TABLE IF NOT EXISTS {table} (
                                    id integer PRIMARY KEY,
                                    struct_img dict); """)


            tmp_dict = {"test1": 1,
                        "test2": 2,
                        "test3": None,
                        "test4": "aha"}
            self.assertEqual(x1.db_table_entries_exists(table), False)
            x1.insert_db(table, (1, tmp_dict), "(?, ?)")
            x1.insert_db(table, (2, tmp_dict), "(?, ?)")
            self.assertEqual(x1.db_table_entries_exists(table), True)
            self.assertEqual(x1.get_db_one(table, 1), (1, {'test1': 1, 'test2': 2, 'test3': None, 'test4': 'aha'}))
            self.assertEqual(x1.get_db_all(table), [(1, {'test1': 1, 'test2': 2, 'test3': None, 'test4': 'aha'}),
                                                       (2, {'test1': 1, 'test2': 2, 'test3': None, 'test4': 'aha'})])

            self.assertEqual(x1.get_db_all("test"), None)
            self.assertEqual(x1.get_db_one("test", 1), None)
            self.assertEqual(x1.db_table_entries_exists("test"), None)

            x1.delete()
            x1.conn = None

            #Creates error log statement
            x1.insert_db(table, (2, tmp_dict), "(?, ?)")

            x1.create_db_table(f""" CREATE TABLE IF NOT EXISTS {table} (
                                    id integer PRIMARY KEY,
                                    struct_img dict); """)

            x1.delete()

class TestCaseSpecs(unittest.TestCase):
    """
    Test Casees of specs.py
    """
    def test_validate_yaml_config(self):
        """
        test_validate_yaml_config
        Tests the validation of a yaml dictionary
        """

        tmp_dict = {"test1": 1,
                    "test2": 2,
                    "test3": None,
                    "test4": "aha"}
        a, b = validate_yaml_config(tmp_dict)
        self.assertEqual(a, ["missing 'optimizer', not in .", "missing 'sequential_scheduler', not in .", "missing 'scheduler', not in .", "missing 'hyp', not in .", "unknown 'test1' in .", "unknown 'test2' in .", "unknown 'test3' in .", "unknown 'test4' in ."])
        self.assertEqual(b, False)

    def test_validate_file(self):
        """
        test_validate_file
        Tests the validation of a yaml dictionary from a file
        """
        with self.assertRaises(AssertionError):
            yml_hyp = validate_file("./models/hyp.a")
            yml_hyp = validate_file("./models/hyp2.yaml")

        with assertNotRaises(AssertionError):
            yml_hyp = validate_file("./models/hyp.yaml")

class TestPlotting(unittest.TestCase):
    """
    Test Casees of plotting.py
    """
    def test_AverageMeter(self):
        """
        test_AverageMeter
        Tests the Average Meter if function is properly
        """

        x = AverageMeter()
        self.assertEqual(x.avg, 0)
        self.assertEqual(x.sum, 0)
        self.assertEqual(x.count, 0)

        x.update(1)
        x.update(2)
        x.update(3)
        x.update(4)
        x.update(1)

        self.assertEqual(x.avg, 2.2)
        self.assertEqual(x.sum, 11)
        self.assertEqual(x.count, 5)
        self.assertEqual(list(x.vallist), [1., 4., 3., 2., 1., 0.])

        x.reset()
        self.assertEqual(x.avg, 0)
        self.assertEqual(x.sum, 0)
        self.assertEqual(x.count, 0)

    def test_Plotting(self):
        """
        test_Plotting
        Tests Plotting if functions properly
        """
        with assertNotRaises(AssertionError):
            neural_net = resnet18(pretrained=False)
            neural_net.fc = nn.Linear(neural_net.fc.in_features, 3)

            pred = neural_net(torch.rand(8, 3, 100, 100))
            label = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0])
            criterion = CrossEntropyLoss()
            loss = criterion(pred, label)

            p = Plotting(nosave=True)
            p.update("train", "symmetry", label, pred, loss)
            p.update("train", "symmetry", label, pred, loss)
            p.update("train", "symmetry", label, pred, loss)
            p.update("train", "symmetry", label, pred, loss)
            p.update("train", "symmetry", label, pred, loss)
            val_dict = p.update_epoch("symmetry")


class TestCutter(unittest.TestCase):
    """
    Test Casees of cutter.py
    """
    def test_load_image(self):
        """
        test_load_image
        Tests loadimage and wrapper
        """
        with assertNotRaises(AssertionError):
            path = '../../test_data/Muskeltransplantation/0001'
            x = load_image(path, "01")

            cut = Cutter.instance()
            wrap = cut.cut_wrapper()

            wrap["symmetry"](path, "01")
            wrap["eye"](path, "01")
            wrap["mouth"](path, "01")
            wrap["forehead"](path, "01")
            wrap["hb_direct"](path, "01")

            self.assertEqual(load_image("../../test_data/Faziale_Reanimation/001", "aa"), None)


if __name__ == "__main__":
    LOGGER.setLevel(logging.CRITICAL)
    check_requirements()
    unittest.main()
