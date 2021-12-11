import os
import os.path
import uuid
from pathlib import Path
from typing import List
import uvicorn
import asyncio
import datetime


import threading
from  apscheduler.schedulers.asyncio import AsyncIOScheduler

from fastapi import Request, File, UploadFile, FastAPI
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from utils.general import check_requirements

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
        data = {
            "name": "name",
            "result": "TODO list Result",
        }
        await asyncio.sleep(20)
        delete_folder_content(workspace)
        os.rmdir(workspace)
    return data


if __name__ == "__main__":
    check_requirements()
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True, workers=4)
