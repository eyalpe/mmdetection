from mmdet.apis import init_detector
from tools.yair.visualization_utils import get_layers_dict, plot_layer_filter
import tools.yair.hackathon.hackathon_utils as hku
import os


# Show filters before loading the trained net:
device = 'cuda:0'
MMDET_DIR = hku.get_mmdet_root()
RESULTS_ROOT_DIR = MMDET_DIR + '/../../results/'

config_file_path = MMDET_DIR + '/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
input_image_path = MMDET_DIR + '/demo/demo.jpg'
output_dir_root_path = RESULTS_ROOT_DIR + '/task_0_show_filters'

model = init_detector(config_file_path, checkpoint=None, device=device)

layers_dict = get_layers_dict(model)
layers_names = list(layers_dict.keys())
output_dir_path = os.path.join(output_dir_root_path, 'before_init')
for layer_name in [l for l in layers_names if '.bn' not in l and 'downsample' not in l and l != '']:
    output_file_path = os.path.join(output_dir_path, f'{layer_name}.png')
    plot_layer_filter(model, layer_name, show=False, output_file_path=output_file_path)


# Show filters after loading the trained net:

checkpoint_file_path = MMDET_DIR + '/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
model = init_detector(config_file_path, checkpoint=checkpoint_file_path, device=device)

output_dir_path = os.path.join(output_dir_root_path, 'after_init')
for layer_name in [l for l in layers_names if '.bn' not in l and 'downsample' not in l and l != '']:
    output_file_path = os.path.join(output_dir_path, f'{layer_name}.png')
    plot_layer_filter(model, layer_name, show=False, output_file_path=output_file_path)




