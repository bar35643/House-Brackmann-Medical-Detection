FROM python:3.8

RUN apt-get update
RUN pip install --upgrade pip

RUN apt-get install -y python3-opencv

COPY source/hbmedicalprocessing/requirements.txt /
RUN pip install opencv-python
RUN pip intsall pyheif
RUN pip install -r /requirements.txt

COPY ./source/hbmedicalprocessing /
ENV PYTHONPATH /
