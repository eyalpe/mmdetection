from tools.deployment.pytorch2onnx import main as mmdet_pytorch2onnx
import sys

def run_pytorch2onnx():
    #sys.path.append()
    sys.argv.clear()
    config = '/home/yairshe/projectes/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint = '/home/yairshe/projectes/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    sys.argv.append('config')
    sys.argv.append(config)
    #sys.argv.append('checkpoint')
    sys.argv.append(checkpoint)
    sys.argv.append('--output-file')
    sys.argv.append('/home/yairshe/results/mmdetection_hackathon/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.onnx')
    sys.argv.append('--input-img')
    sys.argv.append('/home/yairshe/projectes/mmdetection/demo/demo.jpg')
    mmdet_pytorch2onnx()
    return

if __name__ == '__main__':
    run_pytorch2onnx()