
import threading

from .config import LOGGER

def try_except(func):
    """
    try-except function. Usage: @try_except decorator

    :param func:  Function which should be decorated (function)
    """

    def handler(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as err:
            LOGGER.error(err)

    return handler

def try_except_none(func):
    """
    try-except_none function. Usage: @try_except_none decorator

    :param func:  Function which should be decorated (function)
    :returns value of func or None
    """

    def handler(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as err:
            LOGGER.error(err)
            return None
    return handler


def thread_safe(func):
    """
    thread_safe function. Usage: @thread_safe decorator

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
