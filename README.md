FROM continuumio/miniconda3

RUN conda create -n detect_mask python=3.6
RUN echo "source activate env" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH
# create docker image
>> docker build -t condatest .
<!-- improve dataset and fix error case -->

