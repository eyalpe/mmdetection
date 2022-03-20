from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from tools.yair.save_output import SaveOutput
from tools.yair.plot_utils import plot_tensor_ontop_image
from tools.yair.visualization_utils import get_layers_dict
import sys
import os


# Show filters before loading the trained net:
device = 'cuda:0'
config_file_path = '/home/yairshe/projectes/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
input_image_path = '/home/yairshe/projectes/mmdetection/demo/demo.jpg'
checkpoint_file_path = '/home/yairshe/projectes/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

model = init_detector(config_file_path, checkpoint_file_path, device=device)

layer_name = 'backbone.layer1.0.conv1.weight'
layer_name = 'backbone.layer1.0.conv2'
layer_name = 'backbone.layer1.2.conv3'
layer_name = 'backbone.layer1.2.conv2'

layers_dict = get_layers_dict(model)
layers_names = list(layers_dict.keys())

output_dir_root_path = '/home/yairshe/results/mmdetection_hackathon/task_1_visualize_objectness'
os.makedirs(output_dir_root_path, exist_ok=True)
# Add an hook for saving output:
save_output = SaveOutput()
layers_dict['rpn_head.rpn_cls'].register_forward_hook(save_output)

# inference the demo image
results = inference_detector(model, input_image_path)


# plot_tensor(save_output.outputs)
# plt.imshow(features['backbone.conv1'][0, 0, ...].cpu().numpy().astype(np.uint8))


rpn_cls_fpn_4 = -save_output.outputs[4]
plot_tensor_ontop_image(tensor=rpn_cls_fpn_4, image_path=input_image_path)

for i in range(5):
    output_path = os.path.join(output_dir_root_path, f'fpn_{i}.png')
    plot_tensor_ontop_image(tensor=-save_output.outputs[i], image_path=input_image_path, output_path=output_path)



