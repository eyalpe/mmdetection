from tools.analysis_tools.analyze_results import main as analyze_results
import sys

def run_analyze_results():
    sys.argv.clear()
    config = '/home/yairshe/projectes/mmdetection/configs/yair/config_complete/faster_rcnn_r50_fpn_1x_coco_car_damage.py'
    prediction_path = '/home/yairshe/results/mmdetection_hackathon/coco_car_damage/test.pkl'
    show_dir = '/home/yairshe/results/mmdetection_hackathon/coco_car_damage/results2'
    sys.argv.append('config')
    sys.argv.append(config)
    sys.argv.append(prediction_path)
    sys.argv.append(show_dir)
    analyze_results()
    return

if __name__ == '__main__':
    run_analyze_results()