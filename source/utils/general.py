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

#⚠️ 🚀

import logging
import glob
import platform
import re
import socket
from copy import deepcopy
from pathlib import Path
from subprocess import check_output
import pkg_resources as pkg

import torch

from .config import LOGGER, LOCAL_RANK, RANK, WORLD_SIZE
from .pytorch_utils import is_process_group #pylint: disable=import-error
from .decorators import try_except #pylint: disable=import-error
from .singleton import Singleton #pylint: disable=import-error


@Singleton
class OptArgs():
    """
    Class for setting the augmentation to true or false globally
    """
    def __init__(self):
        """
        Initializes the class
        :param val: value (bool)
        """
        self.args = None
    def get_arg_from_key(self, key):
        """
        Setting the value
        :param item: kewword in dict (dict)
        """
        return self.args[key]

    def __call__(self, args):
        """
        Setting the value
        :param args: args (dict)
        """
        self.args = args

def set_logging(level, main_inp_func):
    """
    Setting up the logger

    :param level: one of (logging.DEBUG, logging.INFO)
    """
    if is_process_group(RANK):
        format = f"%(asctime)s Process_{RANK}:%(filename)s:%(funcName)s():%(lineno)d [%(levelname)s] --- %(message)s"  #pylint: disable=redefined-builtin
    else:
        format = "%(asctime)s %(filename)s:%(funcName)s():%(lineno)d [%(levelname)s] --- %(message)s"

    logging.basicConfig(
        level=level,
        format=format,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            #logging.FileHandler("debug.log"), #TODO enable
        ])
    log_str = main_inp_func + ", ".join(f"{k}={v}" for k, v in OptArgs.instance().args.items()) #pylint: disable=no-member
    if level == logging.WARN:
        LOGGER.warning(log_str)
        LOGGER.warning("%sEnvironment: Local_Rank=%s Rank=%s World-Size=%s",
                       main_inp_func, LOCAL_RANK, RANK, WORLD_SIZE)
        LOGGER.warning("%sEnvironment: Cuda-Available=%s Device-Count=%s Distributed-Available=%s",
                       main_inp_func, torch.cuda.is_available(), torch.cuda.device_count(), torch.distributed.is_available())
    else:
        LOGGER.info(log_str)
        LOGGER.info("%sEnvironment: Local_Rank=%s Rank=%s World-Size=%s",
                       main_inp_func, LOCAL_RANK, RANK, WORLD_SIZE)
        LOGGER.info("%sEnvironment: Cuda-Available=%s Device-Count=%s Distributed-Available=%s",
                       main_inp_func, torch.cuda.is_available(), torch.cuda.device_count(), torch.distributed.is_available())







def merge_two_dicts(dict1, dict2):
    """
    merges two dictionary
    :param dict1: Dictionary 1 (dict)
    :param dict2: Dictionary 2 (dict)
    :return dict
    """
    res_dict = dict1.copy()   # start with keys and values of dict1
    res_dict.update(dict2)    # modifies res_dict with keys and values of dict2
    return res_dict

def init_dict(inp_dict: dict, val):
    """
    Creates a new Dictionary with a Preset Value

    :param inp_dict: Input dictionary which should be Initialized (dict)
    :param val: Value which the Dictionary should be initialized (string, int, float, list, ...)

    :return dict
    """
    return dict((k, deepcopy(val)) for k in inp_dict)







def check_online():
    """
    Check if the outside world is accsessible

    :return True (World is accsessible) or False (World is not accsessible only localhost)
    """

    try:
        socket_instance = socket.create_connection(("8.8.8.8", 443), 5)  # check host accessibility
        socket_instance.close()
        return True
    except OSError:
        return False

def check_version(current="0.0.0", minimum="0.0.0", name="version "):
    """
    Check versions of packages

    :param current: Current version of the Package (String)
    :param minimum: Minimum version of the Package required (String)
    :param name: name of the Package (String)

    Source:
        Project: yolov5
        License: GNU GPL v3
        Author: Glenn Jocher
        Url: https://github.com/ultralytics/yolov5
        Date of Copy: 6. October 2021
    """

    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    assert current >= minimum, f"{name}{minimum} is required, but {name}{current} is currently installed"

def check_python(minimum="3.8.0"):
    """
    Check versions of Python

    :param minimum: Minimum version of the Package required (String)

    Source:
        Project: yolov5
        License: GNU GPL v3
        Author: Glenn Jocher
        Url: https://github.com/ultralytics/yolov5
        Date of Copy: 6. October 2021
    """

    check_version(platform.python_version(), minimum, name="Python ")

@try_except
def check_requirements(requirements="requirements.txt", exclude=(), install=True):
    """
     Check installed dependencies meet requirements

    :param requirements:  List of all Requirements needed (parse *.txt file or list of packages)
    :param exclude: List of all Requirements which will be excuded from the checking (list of packages)
    :param install: True for attempting auto update or False for manual use (True or False)

    Source:
        Project: yolov5
        License: GNU GPL v3
        Author: Glenn Jocher
        Url: https://github.com/ultralytics/yolov5
        Date of Copy: 6. October 2021
    Modified Code
    """

    check_python()  # check python version
    if isinstance(requirements, (str, Path)):  # requirements.txt file
        file = Path(requirements)
        assert file.exists(), f"requirements: {file.resolve()} not found, check failed."
        requirements = [f"{x.name}{x.specifier}" for x in pkg.parse_requirements(file.open(encoding="UTF-8")) if x.name not in exclude]
    else:  # list or tuple of packages
        requirements = [x for x in requirements if x not in exclude]

    for req in requirements:
        try:
            pkg.require(req)
        except Exception as err:  # DistributionNotFound or VersionConflict if requirements not met
            if install:
                LOGGER.info("requirements: %s not found or has the wrong version and is required by this Package, attempting auto-update...", req)
                try:
                    assert check_online(), f"pip install {req} skipped (offline no internet connection)"

                    LOGGER.info(check_output(f"pip install {req}").decode("utf-8").strip())
                    LOGGER.info("requirements: Package %s updated!", req)
                except Exception as err:
                    LOGGER.error("requirements:  %s", err)
            else:
                LOGGER.error("requirements: %s not found and is required by this Package. Please install it manually and rerun your command.", req)







def increment_path(path, exist_ok=False, sep="", mkdir=False):
    """
    TODO
     Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.

    Source:
        Project: yolov5
        License: GNU GPL v3
        Author: Glenn Jocher
        Url: https://github.com/ultralytics/yolov5
        Date of Copy: 6. October 2021
    """

    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix("")
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        path = Path(f"{path}{sep}{max(i) + 1 if i else 2}{suffix}")  # update path
    dir_0 = path if path.suffix == "" else path.parent  # directory
    if not dir_0.exists() and mkdir:
        dir_0.mkdir(parents=True, exist_ok=True)  # make directory
    return path
