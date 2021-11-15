#TODO Docstring
"""
TODO
"""

import os
import sys
from pathlib import Path
import logging

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT_RELATIVE = ROOT.relative_to(Path.cwd())  # relative


LOCAL_RANK = int(os.getenv("LOCAL_RANK", "-1"))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", "-1"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))

LOGGER = logging.getLogger(__name__)

LRU_MAX_SIZE=100
THREADPOOL_NUM_THREADS = min(8, os.cpu_count())  # number of multiprocessing threads
