import os
import os.path
import uuid
from pathlib import Path
from typing import List
import uvicorn

from fastapi import Request, File, UploadFile, FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from utils.general import check_requirements

templates = Jinja2Templates(directory="static/templates")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")







def create_workspace():
    """
    Return workspace path
    """
    # base directory
    work_dir = Path('static/upload_temporary/')
    # UUID to prevent file overwrite
    request_id = Path(str(uuid.uuid4())[:32])
    # path concat instead of work_dir + '/' + request_id
    workspace = work_dir / request_id
    if not os.path.exists(workspace):
        # recursively create workdir/unique_id
        os.makedirs(workspace)

    return workspace





@app.get("/", response_class=HTMLResponse)
def get_upload(request: Request):
    result = "Hello from upload.py"
    return templates.TemplateResponse('base.html', context={'request': request, 'result': result})


@app.post("/upload/new/")
async def post_upload(files: List[UploadFile] = File(...)):
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

    print(workspace)
    data = {
        "name": "name",
        "result": "TODO list Result",
    }
    return data


    """
    data_dict = eval(imgdata[0])
    #winWidth, imgWidth, imgHeight = data_dict["winWidth"], data_dict["imgWidth"], data_dict["imgHeight"]

    # create the directory path
    workspace = create_workspace()
    # filename
    file_name = Path(file.filename)
    print(file.filename)
    print(type(file.filename))
    print(file_name)
    print(type(file_name))
    # image full path
    #img_full_path = workspace / file_name
    #with open(str(img_full_path), 'wb') as myfile:
    #    contents = await file.read()
    #    myfile.write(contents)
    # create a thumb image and save it
    #thumb(img_full_path, winWidth, imgWidth, imgHeight)
    # create the thumb path
    # ext is like .png or .jpg
   # filepath, ext = os.path.splitext(img_full_path)
    #thumb_path = filepath + ".thumbnail"+ext

    data = {
        "img_path": img_full_path,
        "thumb_path": thumb_path
    }
    """


if __name__ == "__main__":
    check_requirements()
    uvicorn.run(app, host="127.0.0.1", port=8000)
