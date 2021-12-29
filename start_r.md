# MMDetection Hackathon
## getting started tutorial

__NOTE: This instructions are written to be used with CUDA-10.2 and Ubuntu 18.04__

### Prerequisites:
* Sign-in (and sign-up if required) to github
* Fork your mmdetection copy from github website
* Clone your local copy of mmdetection (NOTE: Replace `<user>` with your own user-name)
```
git clone git@github.com:<user>/mmdetection.git
```
* Install CUDA as explained in https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation
```
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get update
sudo apt-get install cuda-10-2
```
* Install python, python-dev, virtualenv
```
sudo apt install python3 python3-virtualenv virtualenv libpython3.6-dev
```

Install mmdetection
---
* create venv, load it, and install mmdetection in __editable__ mode:
```
virtualenv -p python3 venv_mmdet
. venv_mmdet/bin/activate
pip install torch torchvision torchaudio openmim
mim install -e .
```
Test:
---
* Verifiy mmdetection installation:
```
python -c 'import mmdet; print(mmdet.__version__)'
```
* Verifiy CUDA is used:
```
python -c 'import torch; print("IS CUDA SUPPORTED: %s"%torch.cuda.is_available())'
```
* Download the model
```
mim download mmdet --config faster_rcnn_r50_fpn_1x_coco --dest .
mkdir checkpoints
mv faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth checkpoints/
```
* Run an inference example:
```
python -c "from mmdet.apis import init_detector, inference_detector;config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py';checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth';device = 'cuda:0';model = init_detector(config_file, checkpoint_file, device=device);inference_detector(model, 'demo/demo.jpg');"
```