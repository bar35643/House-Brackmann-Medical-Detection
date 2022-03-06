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

import threading

from .config import LOGGER #pylint: disable=import-error

def try_except(func):
    """
    try-except function. Usage: @try_except decorator.
    Decorated Function returns function Falue or terminates and Print error

    :param func:  Function which should be decorated (function)
    """

    def handler(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as err: #pylint: disable=broad-except
            LOGGER.error(err)

    return handler

def try_except_none(func):
    """
    try-except_none function. Usage: @try_except_none decorator.
    Decorated Function returns function Falue or None

    :param func:  Function which should be decorated (function)
    :returns value of func or None
    """

    def handler(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as err: #pylint: disable=broad-except
            LOGGER.error(err)
            return None
    return handler


def thread_safe(func):
    """
    thread_safe function. Usage: @thread_safe decorator.
    Decorated Function can be executed Thread Safe

    :param func:  Function which should be decorated (function)
    :returns value of func
    """
    lock = threading.Lock()

    def handler(*args, **kwargs):
        #lock.acquire()
        with lock:
            #curr_thread = threading.currentThread().getName()
            ret = func(*args, **kwargs)

        #lock.release()
        return ret
    return handler
