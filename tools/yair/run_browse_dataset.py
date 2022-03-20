from tools.misc.browse_dataset import main as mmdet_browse_dataset
import sys

def run_browse_dataset():
    sys.argv.clear()
    config = '/home/yairshe/projectes/mmdetection/configs/yair/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_car_damage.py'
    sys.argv.append('config')
    sys.argv.append(config)
    mmdet_browse_dataset()
    return

if __name__ == '__main__':
    run_browse_dataset()