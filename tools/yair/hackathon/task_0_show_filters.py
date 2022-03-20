from mmdet.apis import init_detector
from tools.yair.visualization_utils import get_layers_dict, plot_layer_filter
import tools.yair.hackathon.hackathon_utils as hku
import os


# Show filters before loading the trained net:
device = 'cuda:0'
MMDET_DIR = hku.get_mmdet_root()
HOME_DIR = hku.get_user_home_dir()
config_file_path = MMDET_DIR + '/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
input_image_path = MMDET_DIR + '/demo/demo.jpg'
output_dir_root_path = HOME_DIR + '/results/mmdetection_hackathon/task_0_show_filters'

model = init_detector(config_file_path, checkpoint=None, device=device)

layer_name = 'backbone.layer1.0.conv1.weight'
layer_name = 'backbone.layer1.0.conv2'
layer_name = 'backbone.layer1.2.conv3'
layer_name = 'backbone.layer1.2.conv2'

layers_dict = get_layers_dict(model)
layers_names = list(layers_dict.keys())


output_dir_path = os.path.join(output_dir_root_path, 'before_init')
for layer_name in layers_names[1:]:
    if '.bn' in layer_name or 'downsample' in layer_name:
        continue
    output_file_path = os.path.join(output_dir_path, f'{layer_name}.png')
    plot_layer_filter(model, layer_name, show=False, output_file_path=output_file_path)


# Show filters after loading the trained net:

checkpoint_file_path = MMDET_DIR + '/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
model = init_detector(config_file_path, checkpoint=checkpoint_file_path, device=device)

output_dir_path = os.path.join(output_dir_root_path, 'after_init')
for layer_name in layers_names[1:]:
    if '.bn' in layer_name or 'downsample' in layer_name:
        continue
    output_file_path = os.path.join(output_dir_path, f'{layer_name}.png')
    plot_layer_filter(model, layer_name, show=False, output_file_path=output_file_path)




