#ARG PYTORCH="1.6.0"
#ARG CUDA="10.1"
#ARG CUDNN="7"

FROM nvidia/cuda:10.1-devel-ubuntu18.04
#pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 python3 python3-virtualenv virtualenv libpython3.6-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV
RUN virtualenv -p python3.6 /envs/mmdet
RUN . /envs/mmdet/bin/activate && pip install cython numpy openmim
RUN . /envs/mmdet/bin/activate && pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html 
RUN . /envs/mmdet/bin/activate && pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html

# Install MMDetection
#RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection
WORKDIR /mmdetection
ENV FORCE_CUDA="1"
RUN . /envs/mmdet/bin/activate && pip install --no-cache-dir -e .
RUN 'export LC_ALL=C.UTF-8' >> ${HOME}/.bashrc
RUN 'export LANG=C.UTF-8' >> ${HOME}/.bashrc
RUN '. /envs/mmdet/bin/activate' >> ${HOME}/.bashrc

