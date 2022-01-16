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

#https://docs.python.org/3/library/unittest.html
#https://ongspxm.gitlab.io/blog/2016/11/assertraises-testing-for-errors-in-unittest/

import argparse
import unittest
import logging
import threading
import time
from copy import deepcopy

import torch

from utils.config import LOGGER
from utils.general import check_python, check_requirements, set_logging, OptArgs, init_dict, merge_two_dicts, check_online, check_version
from utils.pytorch_utils import select_device, is_process_group, is_master_process
from utils.decorators import try_except_none, try_except, thread_safe
from utils.singleton import Singleton
from utils.dataloader import LoadImages, CreateDataset

#pylint: disable=invalid-name, no-member, too-few-public-methods, no-self-use

class TestCaseGeneral(unittest.TestCase):
    """
    Test Casees of the general.py file
    """
    def test_init_dict(self):
        """
        Testing if initializing dictionary correct
        """
        a = {"test1": None,
             "test2": None,}

        b = {"test1": [],
             "test2": [],}

        self.assertEqual(init_dict(a, []), b)

    def test_merge_two_dicts(self):
        """
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

    def test_check_online(self):
        """
        Testing Check online if True
        """

        self.assertEqual(check_online(), True)

    def test_check_version_no_exeption(self):
        """
        Testing Check Version if Assertion raises
        """

        check_version(current="2.0.0", minimum="1.0.0")

    def test_check_version_exeption(self):
        """
        Testing Check Version if Assertion raises
        """

        with self.assertRaises(AssertionError):
            check_version(current="1.0.0", minimum="2.0.0")

    def test_check_python_no_exception(self):
        """
        Testing if no exeption raises

        Enters: a Version that is smaller than the Version install on the Computer
        Expects: No Assertion raises
        """
        check_python(minimum="3.7.0")

    def test_check_python_exception(self):
        """
        Testing if no exeption raises

        Enters: a Version that is greater than the Version install on the Computer
        Expects: Assertion Raises
        """
        with self.assertRaises(AssertionError):
            check_python(minimum="3.9.0")






class TestCasePytorchUtils(unittest.TestCase):
    """
    Test Casees of the pytorch.utils.py file
    """

    def test_is_process_group_test1(self):
        """
        Testing is_process_group
        """
        self.assertEqual(is_process_group(0), True)
        self.assertEqual(is_process_group(1), True)
        self.assertEqual(is_process_group(2), True)

    def test_is_process_group_test2(self):
        """
        Testing is_process_group
        """
        self.assertEqual(is_process_group(-1), False)

    def test_is_master_process_test1(self):
        """
        Testing is_master_process
        """
        self.assertEqual(is_master_process(-1), True)
        self.assertEqual(is_master_process(0), True)

    def test_is_master_process_test2(self):
        """
        Testing is_master_process
        """
        self.assertEqual(is_master_process(1), False)
        self.assertEqual(is_master_process(2), False)
        self.assertEqual(is_master_process(3), False)


    def test_select_device_cpu_general(self):
        """
        Testing if the device sets correctly part 1

        Enters: select device "cpu"
        Expects: set to "cpu"
        """
        self.assertEqual(select_device("cpu", 16), torch.device("cpu"))

    def test_select_device_gpu_cuda_not_available(self):
        """
        Testing if the device sets correctly part 2

        Enters: select device "0" if cuda is not available
        Expects: Assertion Raises
        """
        if not torch.cuda.is_available():
            with self.assertRaises(AssertionError):
                select_device("0", 16)

    def test_select_device_gpu_cuda_available(self):
        """
        Testing if the device sets correctly part 3

        Enters: select device "0" if cuda is available
        Expects: set device to cuda:0
        """
        if torch.cuda.is_available():
            self.assertEqual(select_device("0", 16), torch.device("cuda:0"))

    def test_select_device_more_gpu_cuda_available(self):
        """
        Testing if the device sets correctly part 4

        Enters: select more than one device
        Expects: set device to cuda:0
        """
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.assertEqual(select_device("0, 1", 16), torch.device("cuda:0"))

    def test_select_device_cuda_available_more_batchsize_wrong(self):
        """
        Testing if the device sets correctly part 5

        Enters: select more than one device and batch size is not correct
        Expects: Assertion Raises
        """
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            with self.assertRaises(AssertionError):
                select_device("0, 1", 1)





class TestCasePatterns(unittest.TestCase):
    """
    Test Casees of the singleton.py and decorators.py file
    """
    def test_singleton(self):
        """
        Testing signleton Pattern
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

        Explicit return type
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


class TestCaseDataloader(unittest.TestCase):
    """
    Test Casees of dataloader.py
    """
    def test_load_images(self):
        """
        test_try_expect
        """
        tst = LoadImages(path='../test_data', prefix_for_log='')
        self.assertEqual(len(tst), 12)

        tst = LoadImages(path='../test_data/Muskeltransplantation', prefix_for_log='')
        self.assertEqual(len(tst), 6)

        tst = LoadImages(path='../test_data/Muskeltransplantation/0001', prefix_for_log='')
        self.assertEqual(len(tst), 1)

    def test_create_dataset(self):
        """
        test_try_expect
        """
        tst = CreateDataset(path='../test_data', prefix_for_log='')
        self.assertEqual(len(tst), 12)






def parse_opt():
    """
    Parser options
    """
    parser = argparse.ArgumentParser()
    return parser.parse_args()


if __name__ == "__main__":
    opt_args = vars(parse_opt())
    OptArgs.instance()(opt_args)
    set_logging("unittest: ")
    LOGGER.setLevel(logging.CRITICAL)
    check_requirements()
    unittest.main()
