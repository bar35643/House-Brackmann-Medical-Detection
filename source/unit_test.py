"""
TODO
"""

#https://docs.python.org/3/library/unittest.html
#https://ongspxm.gitlab.io/blog/2016/11/assertraises-testing-for-errors-in-unittest/

import argparse
import unittest
import logging

import torch

from utils.general import check_python, check_requirements, set_logging
from utils.pytorch_utils import select_device

LOGGER = logging.getLogger(__name__)

class TestCaseGeneral(unittest.TestCase):
    """
    Test Casees of the general.py file
    """

    def test_check_python_no_exception(self): #pylint: disable=no-self-use
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
















def parse_opt():
    """
    Parser options
    """
    parser = argparse.ArgumentParser()
    return parser.parse_args()


if __name__ == "__main__":
    #pylint: disable=pointless-string-statement
    """
    Main Function for starting the unittest
    """
    opt_args = parse_opt()
    set_logging(logging.ERROR, "unittest: starting...", opt_args)
    check_requirements()
    unittest.main()
