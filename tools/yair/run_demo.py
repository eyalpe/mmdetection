import numpy as np

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from tools.yair.save_output import SaveOutput
from tools.yair.plot_utils import plot_tensor_ontop_image,plot_tensor
from tools.yair.visualization_utils import get_layers_dict, plot_layer_filter, print_model
import sys

sys.path.append('/home/yairshe/projectes/mmdetection')

# Define input:
config_file = '/home/yairshe/projectes/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = '/home/yairshe/projectes/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
device = 'cuda:0'
input_image_path = '/home/yairshe/projectes/mmdetection/demo/demo.jpg'
output_dir_path = '/home/yairshe/results/mmdetection_hackathon/demo'

# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
print_model(model, output_dir_path)

#plot_layer_filter(model, 'backbone.conv1')
plot_layer_filter(model, 'backbone.layer1.0.conv2', show=True)


layers_dict = get_layers_dict(model)
layers_names = list(layers_dict.keys())

# Add an hook for saving output:
save_output = SaveOutput()
layers_dict['rpn_head.rpn_cls'].register_forward_hook(save_output)

# inference the demo image
results = inference_detector(model, input_image_path)


# plot_tensor(save_output.outputs)
# plt.imshow(features['backbone.conv1'][0, 0, ...].cpu().numpy().astype(np.uint8))


rpn_cls_fpn_4 = -save_output.outputs[4]
plot_tensor_ontop_image(tensor=rpn_cls_fpn_4, image_path=input_image_path)


# Let's plot the result
show_result_pyplot(model, input_image_path, results, score_thr=0.3)