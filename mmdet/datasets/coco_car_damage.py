from .builder import DATASETS
from mmdet.datasets.coco import CocoDataset
from .api_wrappers import COCO, COCOeval


@DATASETS.register_module()
class CocoCarDamage(CocoDataset):
    CLASSES = ('damage',)

    # def get_cat_ids(self, idx):
    #     return super(self).get_cat_ids(idx+1)

