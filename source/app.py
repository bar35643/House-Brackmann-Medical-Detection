import os
import os.path
import uuid
import logging
from pathlib import Path
from typing import List
import uvicorn
import asyncio
import datetime
import argparse


import threading
from  apscheduler.schedulers.asyncio import AsyncIOScheduler

from fastapi import Request, File, UploadFile, FastAPI
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


import detect
from utils.argparse_utils import restricted_val_split, SmartFormatter
from utils.general import check_requirements, set_logging, OptArgs


PREFIX = "app: "
LOGGING_STATE = logging.INFO #logging.DEBUG

templates = Jinja2Templates(directory="static/templates")
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


def delete_folder_content(dir):
    for root, dirs, files in os.walk(dir, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))

work_dir = Path('static/upload_temporary/')
scheduler_lock = threading.Lock()
def work_dir_cleaner():
    with scheduler_lock:
        delete_folder_content(work_dir)
        print("Scheduler Run: ", datetime.datetime.now())

# https://apscheduler.readthedocs.io/en/stable/modules/triggers/interval.html
scheduler = AsyncIOScheduler()
scheduler.add_job(work_dir_cleaner, 'interval', minutes=10)
scheduler.start()


def create_workspace():
    """
    Return workspace path
    """
    # UUID to prevent file overwrite
    request_id = Path(str(uuid.uuid4())[:32])
    # path concat instead of work_dir + '/' + request_id
    workspace = work_dir / request_id
    if not os.path.exists(workspace):
        os.makedirs(workspace)
    return workspace


@app.get("/", response_class=HTMLResponse)
def get_upload(request: Request):
    return templates.TemplateResponse('base.html', context={'request': request})






@app.post("/api/upload/")
async def post_upload(files: List[UploadFile] = File(...)):
    with scheduler_lock:
        workspace = create_workspace()


        for file in files:
            print(file.filename)
            file_name = Path(file.filename)
            space = Path(os.path.join(str(workspace), str(file.filename)))
            space.parent.mkdir(exist_ok=True, parents=True)  # make dir

            img_full_path = workspace / file_name
            with open(str(img_full_path), 'wb') as myfile:
                contents = await file.read()
                myfile.write(contents)

        #Do Operation
        print(workspace)


        #try:
        detect.run(weights="models",
                    source=workspace,
                    imgsz=640,
                    batch_size=4,
                    device="cpu",
                    half=False,
                    function_selector="all")
        #except:
        #    ret_val = None


        data = {
            "name": "name",
            "result": "TODO list Result",
        }
        await asyncio.sleep(20)
        delete_folder_content(workspace)
        os.rmdir(workspace)
    return data








def parse_opt():
    """
    TODO
    Check internet connectivity
    """
    parser = argparse.ArgumentParser(formatter_class=SmartFormatter)
    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="corresponding IP")
    parser.add_argument("--port", type=int, default=8000,
                        help="corresponding Port")

    parser.add_argument("--device", default="cpu",
                        help="Device for Processing gpu or cpu")
    parser.add_argument("--reload", action="store_true",
                        help="Reload if Files Changes")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of worker processes")
    return parser.parse_args()

if __name__ == "__main__":
    opt_args = vars(parse_opt())
    OptArgs.instance()(opt_args)

    set_logging(LOGGING_STATE, PREFIX)

    uvicorn.run("app:app",
                host=OptArgs.instance().get_arg_from_key("ip"),
                port=OptArgs.instance().get_arg_from_key("port"),
                reload=OptArgs.instance().get_arg_from_key("reload"),
                workers=OptArgs.instance().get_arg_from_key("workers"))
