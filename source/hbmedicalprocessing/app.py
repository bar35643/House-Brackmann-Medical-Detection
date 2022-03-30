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

import os
import sys
#import asyncio
#import datetime
import argparse
import threading
from pathlib import Path
from typing import List
import uvicorn

#from  apscheduler.schedulers.asyncio import AsyncIOScheduler

from fastapi import Request, File, UploadFile, FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


import detect as dt #pylint: disable=import-error
from utils.argparse_utils import SmartFormatter #pylint: disable=import-error
from utils.config import LOGGER #pylint: disable=import-error
from utils.general import set_logging, OptArgs, delete_folder_content, create_workspace #pylint: disable=import-error



PREFIX = "app: "

templates = Jinja2Templates(directory="static/templates")
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


scheduler_lock = threading.Lock()
# def work_dir_cleaner():
#     with scheduler_lock:
#         delete_folder_content(WORKDIR)
#         print("Scheduler Run: ", datetime.datetime.now())
#
# # https://apscheduler.readthedocs.io/en/stable/modules/triggers/interval.html
# scheduler = AsyncIOScheduler()
# scheduler.add_job(work_dir_cleaner, 'interval', minutes=10)
# scheduler.start()


@app.get("/", response_class=HTMLResponse)
def get_http(request: Request):
    """
    Return Template HTML PAGE
    """
    return templates.TemplateResponse('base.html', context={'request': request})


@app.post("/api/upload/")
async def run_detect(files: List[UploadFile] = File(...)):
    """
    Start Detection for len(files)

    :param files: List of Files (File)
    :return ret_val dictionary with Results
    """
    with scheduler_lock:
        workspace = create_workspace()
        LOGGER.debug("%sCreate Workspace %s", PREFIX, workspace)

        #write all files into the workspace folder
        for file in files:
            # print(file.filename)
            file_name = Path(file.filename)
            space = Path(os.path.join(str(workspace), str(file.filename)))
            space.parent.mkdir(exist_ok=True, parents=True)  # make dir

            img_full_path = workspace / file_name
            with open(str(img_full_path), 'wb') as myfile:
                contents = await file.read()
                myfile.write(contents)


        # Do Operation Detection
        try:
            LOGGER.debug("%sRun Detection...", PREFIX)
            ret_val = dt.run(weights="models",
                             source=workspace,
                             batch_size=4,
                             device="cpu",
                             half=False,
                             function_selector="all",
                             convert=True)
            LOGGER.debug("%sFinished Detection: %s", PREFIX, ret_val)
        except Exception as err: #pylint: disable=broad-except
            ret_val = {"errorcode": str(err)}
            LOGGER.debug("%sError: %s", PREFIX, ret_val)

        #Delete Workspace
        delete_folder_content(workspace)
        os.rmdir(workspace)
        LOGGER.debug("%sDeleted Workspace %s", PREFIX, workspace)
    return ret_val









def parse_opt():
    """
    Command line Parser Options see >> python detect.py -h for more about
    """
    parser = argparse.ArgumentParser(formatter_class=SmartFormatter)
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="corresponding IP")
    parser.add_argument("--port", type=int, default=8000     , help="corresponding Port")
    parser.add_argument("--device", default="cpu"            , help="Device for Processing gpu or cpu")
    parser.add_argument("--reload", action="store_true"      , help="Reload if Files Changes")
    parser.add_argument("--workers", type=int, default=4     , help="Number of worker processes")
    parser.add_argument("--log", action="store_true"         , help="activates log")
    parser.add_argument("--debug", action="store_true"       , help="activates debug logging")
    return parser.parse_args()

if __name__ == "__main__":
    opt_args = vars(parse_opt())
    OptArgs.instance()(opt_args)

    set_logging(PREFIX)

    uvicorn.run("app:app",
                host=OptArgs.instance().get_arg_from_key("ip"),
                port=OptArgs.instance().get_arg_from_key("port"),
                reload=OptArgs.instance().get_arg_from_key("reload"),
                workers=OptArgs.instance().get_arg_from_key("workers"))
