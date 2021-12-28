import numpy as np

from mmdet.apis import init_detector, inference_detector
from plot_utils import plot_tensor
from tools.yair.save_output import SaveOutput

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
layers_dict = get_layers_dict(model)
#plot_layer_filter(model, 'backbone.conv1')
plot_layer_filter(model, 'backbone.layer1.0.conv2')

save_output = SaveOutput()
layers_dict['rpn_head.rpn_cls'].register_forward_hook(save_output)
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
# plot_tensor(save_output.outputs)
# plt.imshow(features['backbone.conv1'][0, 0, ...].cpu().numpy().astype(np.uint8))
x = save_output.outputs[0][0, ...].cpu().numpy()
x = np.transpose(x, (1,2,0))
x = -x
x[..., 0] = (x[..., 0] + x[..., 1] + x[..., 2 ]) / 3
x[..., 1] = x[..., 2 ] = np.zeros(x[..., 0].shape)
x = x.astype(np.uint8)
# plt.imshow(x)
# plt.show()
from PIL import Image
rpn_cls_output = Image.fromarray(x)
l = rpn_cls_output.resize(image.size)
image = Image.open('/home/yairshe/projectes/mmdetection/demo/demo.jpg')
# l = l.convert('L')
image_with_rpn_cls = Image.blend(image, l, 0.5)
image_with_rpn_cls.show()

l.paste(image)
image.size
Image.Image.paste(image, l)

pass