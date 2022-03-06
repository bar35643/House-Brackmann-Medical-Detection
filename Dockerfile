FROM python:3.8

COPY source/hbmedicalprocessing/requirements.txt /
RUN pip install -r /requirements.txt


COPY ./source/hbmedicalprocessing /hbmedicalprocessing
ENV PYTHONPATH /hbmedicalprocessing
