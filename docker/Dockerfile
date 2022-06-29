ARG PYTORCH="1.10.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

#ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
#ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

ADD https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub .
RUN apt-key add 3bf863cc.pub

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y apt-utils && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 curl wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV (note that cuda and torch versions on mmcv-full download link are hardcoded)
RUN pip install --no-cache-dir --upgrade pip wheel setuptools
RUN pip install --no-cache-dir mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html

# Install MMDetection
RUN conda clean --all
RUN git clone --depth 1 --branch v2.24.1 https://github.com/open-mmlab/mmdetection.git /mmdetection
WORKDIR /mmdetection
RUN echo 'host_data/' >> .gitignore
ENV FORCE_CUDA="1"
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir openmim rich modelindex
RUN pip install --no-cache-dir -e .
