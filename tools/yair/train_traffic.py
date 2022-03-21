from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
import os.path as osp
import mmcv
from mmcv import Config, mkdir_or_exist
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import os


cfg = Config.fromfile('/home/yairshe/projectes/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')


from mmdet.apis import set_random_seed

# Modify dataset type and path
cfg.dataset_type = 'Traffic'
cfg.data_root = '/home/yairshe/data/traffic/'

cfg.data.test.type = 'Traffic'
cfg.data.test.data_root = '/home/yairshe/data/traffic/'
cfg.data.test.ann_file = 'set_500_data.json'
cfg.data.test.img_prefix = 'set_500'

cfg.data.train.type = 'Traffic'
cfg.data.train.data_root = '/home/yairshe/data/traffic/'
cfg.data.train.ann_file = 'set_500_data.json'
cfg.data.train.img_prefix = 'set_500'

cfg.data.val.type = 'Traffic'
cfg.data.val.data_root = '/home/yairshe/data/traffic/'
cfg.data.val.ann_file = 'set_500_data.json'
cfg.data.val.img_prefix = 'set_500'

# modify num classes of the model in box head
cfg.model.roi_head.bbox_head.num_classes = 12

cfg.load_from = '/home/yairshe/projectes/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# Set up working dir to save files and logs.
cfg.work_dir = '/home/yairshe/results/mmdetection_hackathon/traffic'

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
cfg.data.workers_per_gpu = 0  # Allows debugging
cfg.dump(os.path.join(cfg.work_dir, 'config.py'))
# cfg.dump('/home/yairshe/projectes/mmdetection/configs/yair/config_complete/faster_rcnn_r50_fpn_1x_coco_car_damage.py')
# Build dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_detector(
    cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
# Add an attribute for visualization convenience
# model.CLASSES =
model.CLASSES = ['obstacles', 'biker', 'car', 'pedestrian', 'trafficLight', 'trafficLight-Green', 'trafficLight-GreenLeft', 'trafficLight-Red', 'trafficLight-RedLeft', 'trafficLight-Yellow', 'trafficLight-YellowLeft', 'truck']

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
