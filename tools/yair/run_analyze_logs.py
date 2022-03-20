from tools.analysis_tools.analyze_logs import main as analyze_logs
import sys

def run_analyze_logs():
    sys.argv.clear()
    json_logs = '/home/yairshe/results/mmdetection_hackathon/coco_car_damage/None.log.json'
    sys.argv.append('json_logs')
    # sys.argv.append(json_logs)
    sys.argv.append('plot_curve')

    sys.argv.append(json_logs)



    analyze_logs()
    return

if __name__ == '__main__':
    run_analyze_logs()