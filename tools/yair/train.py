from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
import os.path as osp
import mmcv
from mmcv import Config, mkdir_or_exist
from mmdet.apis import init_detector, inference_detector, show_result_pyplot



cfg = Config.fromfile('/home/yairshe/projectes/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')


from mmdet.apis import set_random_seed

# Modify dataset type and path
cfg.dataset_type = 'CocoCarDamage'
cfg.data_root = '/home/yairshe/datasets/coco_car_damage/'

cfg.data.test.type = 'CocoCarDamage'
cfg.data.test.data_root = '/home/yairshe/datasets/coco_car_damage/'
cfg.data.test.ann_file = 'COCO_val_annos.json'
cfg.data.test.img_prefix = 'val'

cfg.data.train.type = 'CocoCarDamage'
cfg.data.train.data_root = '/home/yairshe/datasets/coco_car_damage/'
cfg.data.train.ann_file = 'COCO_train_annos.json'
cfg.data.train.img_prefix = 'train'

cfg.data.val.type = 'CocoCarDamage'
cfg.data.val.data_root = '/home/yairshe/datasets/coco_car_damage/'
cfg.data.val.ann_file = 'COCO_val_annos.json'
cfg.data.val.img_prefix = 'val'

# modify num classes of the model in box head
cfg.model.roi_head.bbox_head.num_classes = 1

cfg.load_from = '/home/yairshe/projectes/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# Set up working dir to save files and logs.
cfg.work_dir = '/home/yairshe/results/mmdetection_hackathon/coco_car_damage'

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

# cfg.dump('/home/yairshe/projectes/mmdetection/configs/yair/config_complete/faster_rcnn_r50_fpn_1x_coco_car_damage.py')
# Build dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_detector(
    cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
# Add an attribute for visualization convenience
# model.CLASSES =
model.CLASSES = ['damage']

# Create work_dir
mkdir_or_exist(osp.abspath(cfg.work_dir))
train_detector(model, datasets, cfg, distributed=False, validate=True)

# img = mmcv.imread('/home/yairshe/datasets/coco_car_damage/test/11.jpg')
# config='/home/yairshe/projectes/mmdetection/configs/yair/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_car_damage.py'
# checkpoint = '/home/yairshe/results/mmdetection_hackathon/coco_car_damage/epoch_12.pth'
# # initialize the detector
# model = init_detector(config, checkpoint, device='cuda:0')
# model.cfg = cfg
# result = inference_detector(model, img)
# show_result_pyplot(model, img, result, score_thr=0.9, wait_time=10)
#
