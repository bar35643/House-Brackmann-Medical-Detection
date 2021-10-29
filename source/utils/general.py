#⚠️ 🚀
#TODO Docstring
"""
TODO
"""

import logging
import glob
import platform
import re
import socket
from pathlib import Path
from subprocess import check_output
import pkg_resources as pkg

from .config import LOGGER

def set_logging(level, main_inp_func, opt):
    """
    Setting up the logger

    :param level: one of (logging.DEBUG, logging.INFO)
    """

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(filename)s:%(funcName)s():%(lineno)d [%(levelname)s] --- %(message)s",
        #datefmt="%Y-%m-%d %H:%M:%S", #TODO enable
        handlers=[
            logging.StreamHandler(),
            #logging.FileHandler("debug.log"), #TODO enable
        ])
    log_str = main_inp_func + ", ".join(f"{k}={v}" for k, v in vars(opt).items())
    if level == logging.WARN:
        LOGGER.warning(log_str)
    else:
        LOGGER.info(log_str)

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

    #TODO pinned delete
    """

    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    assert current >= minimum, f"{name}{minimum} is required, but {name}{current} is currently installed"

def check_python(minimum="3.8.0"):
    """
    Check versions of Python

    :param minimum: Minimum version of the Package required (String)
    """

    check_version(platform.python_version(), minimum, name="Python ")

def try_except(func):
    """
    try-except function. Usage: @try_except decorator

    :param func:  Function which should be decorated (function)
    """

    def handler(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except RuntimeError as err:
            LOGGER.error(err)

    return handler

@try_except
def check_requirements(requirements="requirements.txt", exclude=(), install=True):
    """
     Check installed dependencies meet requirements

    :param requirements:  List of all Requirements needed (parse *.txt file or list of packages)
    :param exclude: List of all Requirements which will be excuded from the checking (list of packages)
    :param install: True for attempting auto update or False for manual use (True or False)
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
