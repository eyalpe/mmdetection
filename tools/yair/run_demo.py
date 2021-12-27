from mmdet.apis import init_detector, inference_detector
from plot_utils import plot_tensor
import sys
import torch
from tools.yair.FeatureExtractor import FeatureExtractor
sys.path.append('/home/yairshe/projectes/mmdetection')

def get_layers_dict(model):
    layers_dict = dict([*model.named_modules()])
    return layers_dict

def get_layers_names(model):
    layers_dict = get_layers_dict(model)
    return list(layers_dict.keys())

def plot_layer_filter(model, layer_name):
    layers_dict = get_layers_dict(model)
    layer_parameters = [*layers_dict[layer_name].parameters()]
    assert len(layer_parameters) == 1
    plot_tensor(layer_parameters[0])


config_file = '/home/yairshe/projectes/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = '/home/yairshe/projectes/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
device = 'cuda:0'
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)

layers_names = get_layers_names(model)

#plot_layer_filter(model, 'backbone.conv1')
plot_layer_filter(model, 'backbone.layer1.0.conv2')



#resnet_features = FeatureExtractor(model, layers=["backbone.conv1"])
#dummy_input = torch.ones(10, 3, 224, 224)
#features = resnet_features(dummy_input)


with open('/home/yairshe/results/mmdetection_hackathon/onnx/model_print.txt', 'w') as f:
    f.write(str(model))

with open('/home/yairshe/results/mmdetection_hackathon/onnx/model_layers.txt', 'w') as f:
    for name, param in model.named_parameters():
        print(name)
        print(param.shape)
        f.write(f'{name}\n')
        f.write(f'{str(param.shape)}\n\n')

# inference the demo image
results = inference_detector(model, '/home/yairshe/projectes/mmdetection/demo/demo.jpg')