FROM continuumio/miniconda3

RUN conda create -n detect_mask python=3.6
RUN echo "source activate detect_mask" > ~/.bashrc
ADD . /repo
WORKDIR /repo
RUN pip install -r requirements.txt