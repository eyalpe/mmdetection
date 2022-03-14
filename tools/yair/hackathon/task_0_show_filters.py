from mmdet.apis import init_detector
from tools.yair.visualization_utils import get_layers_dict, plot_layer_filter
import os


# Show filters before loading the trained net:
device = 'cuda:0'
config_file_path = '/home/yairshe/projectes/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
input_image_path = '/home/yairshe/projectes/mmdetection/demo/demo.jpg'
output_dir_root_path = '/home/yairshe/results/mmdetection_hackathon/task_0_show_filters'

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

checkpoint_file_path = '/home/yairshe/projectes/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
model = init_detector(config_file_path, checkpoint=checkpoint_file_path, device=device)

output_dir_path = os.path.join(output_dir_root_path, 'after_init')
for layer_name in layers_names[1:]:
    if '.bn' in layer_name or 'downsample' in layer_name:
        continue
    output_file_path = os.path.join(output_dir_path, f'{layer_name}.png')
    plot_layer_filter(model, layer_name, show=False, output_file_path=output_file_path)




