from tools.analysis_tools.confusion_matrix import main as confusion_matrix
import sys

def run_confusion_matrix_coco_car_damage():
    sys.argv.clear()
    config = '/home/yairshe/projectes/mmdetection/configs/yair/config_complete/faster_rcnn_r50_fpn_1x_coco_car_damage.py'
    prediction_path = '/home/yairshe/results/mmdetection_hackathon/coco_car_damage/test.pkl'
    out_dir = '/home/yairshe/results/mmdetection_hackathon/coco_car_damage/results_coco_error_analysis'
    sys.argv.append('config')
    sys.argv.append(config)
    sys.argv.append(prediction_path)
    sys.argv.append(out_dir)
    sys.argv.append('--show')
    confusion_matrix()
    return

def run_confusion_matrix_traffic():
    sys.argv.clear()
    config = '/home/yairshe/results/mmdetection_hackathon/traffic/config.py'
    prediction_path = '/home/yairshe/results/mmdetection_hackathon/traffic/test.pkl'
    out_dir = '/home/yairshe/results/mmdetection_hackathon/traffic/results_coco_error_analysis'
    sys.argv.append('config')
    sys.argv.append(config)
    sys.argv.append(prediction_path)
    sys.argv.append(out_dir)
    sys.argv.append('--show')
    confusion_matrix()
    return

if __name__ == '__main__':
    # run_confusion_matrix_coco_car_damage()
    run_confusion_matrix_traffic()