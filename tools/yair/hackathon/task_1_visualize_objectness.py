import shutil

import PIL.ImageShow
import numpy as np
import torch

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from tools.yair.save_output import SaveOutput
from tools.yair.plot_utils import plot_tensor_ontop_image
from tools.yair.visualization_utils import get_layers_dict
import tools.yair.hackathon.hackathon_utils as hku
import os


# Show filters before loading the trained net:
device = 'cuda:0'
MMDET_DIR = hku.get_mmdet_root()
RESULTS_ROOT_DIR = MMDET_DIR + '/../../results/'

config_file_path = MMDET_DIR + '/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
input_image_path = MMDET_DIR + '/demo/demo.jpg'
output_dir_root_path = RESULTS_ROOT_DIR + '/task_1_visualize_objectness'

# Show filters before loading the trained net:
device = 'cuda:0'
config_file_path = MMDET_DIR + '/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
input_image_path = MMDET_DIR + '/demo/demo.jpg'
checkpoint_file_path = MMDET_DIR + '/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

model = init_detector(config_file_path, checkpoint_file_path, device=device)

layers_dict = get_layers_dict(model)
layers_names = list(layers_dict.keys())

os.makedirs(output_dir_root_path, exist_ok=True)
# Add an hook for saving output:
save_output = SaveOutput()
layers_dict['rpn_head.rpn_cls'].register_forward_hook(save_output)

# inference the demo image
results = inference_detector(model, input_image_path)


# plot_tensor(save_output.outputs)
# plt.imshow(features['backbone.conv1'][0, 0, ...].cpu().numpy().astype(np.uint8))


#rpn_cls_fpn_4 = -save_output.outputs[4]

plot_tensor_ontop_image(tensor=torch.zeros_like(-save_output.outputs[0]), image_path=input_image_path,
                        output_path=os.path.join(output_dir_root_path, f'fpn_0.png'), ontop_black=False)

for i in range(0, 5):
    bg_im = os.path.join(output_dir_root_path, f'fpn_{i}.png')
    output_path = os.path.join(output_dir_root_path, f'fpn_{i+1}.png')
    plot_tensor_ontop_image(tensor=-save_output.outputs[i], image_path=bg_im, output_path=output_path, alpha=0.15)

from PIL import Image
heatmap = Image.open(os.path.join(output_dir_root_path, f'fpn_5.png'))
orig_image = Image.open(input_image_path)
image_with_heatmap = Image.blend(orig_image, heatmap, ontop_black=True, alpha=0)
image_with_heatmap.save(os.path.join(output_dir_root_path, f'image_w_heatmap.png'))




