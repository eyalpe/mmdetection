from .builder import DATASETS
from mmdet.datasets.coco import CocoDataset
from .api_wrappers import COCO, COCOeval


@DATASETS.register_module()
class Traffic(CocoDataset):
    CLASSES = ('obstacles', 'biker', 'car', 'pedestrian', 'trafficLight', 'trafficLight-Green', 'trafficLight-GreenLeft', 'trafficLight-Red', 'trafficLight-RedLeft', 'trafficLight-Yellow', 'trafficLight-YellowLeft', 'truck')

    # def get_cat_ids(self, idx):
    #     return super(self).get_cat_ids(idx+1)

