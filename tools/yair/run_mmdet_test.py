from tools.test import main as mmdet_test
import sys

def run_mmdet_test():
    sys.argv.clear()
    config = '/home/yairshe/projectes/mmdetection/configs/yair/config_complete/faster_rcnn_r50_fpn_1x_coco_car_damage.py'
    checkpoint = '/home/yairshe/results/mmdetection_hackathon/coco_car_damage/epoch_12.pth'
    sys.argv.append('config')
    sys.argv.append(config)
    sys.argv.append(checkpoint)
    sys.argv.append('--show-dir')
    sys.argv.append('/home/yairshe/results/mmdetection_hackathon/coco_car_damage/results')
    sys.argv.append('--out')
    sys.argv.append('/home/yairshe/results/mmdetection_hackathon/coco_car_damage/test.pkl')


    mmdet_test()
    return

if __name__ == '__main__':
    run_mmdet_test()