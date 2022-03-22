import shutil
import PIL.ImageShow
import numpy as np
import torch
import clip
from PIL import Image
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
input_image_path = '/home/eyalper/hackathon2022/data/depositphotos_93215318-stock-photo-brown-bear-and-other-asian.jpg'
output_dir_root_path = RESULTS_ROOT_DIR + '/task_CLIP'

checkpoint_file_path = MMDET_DIR + '/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
model = init_detector(config_file_path, checkpoint_file_path, device=device)
results = inference_detector(model, input_image_path)
show_result_pyplot(model, input_image_path, results, patches_out_dir=os.path.join(output_dir_root_path, 'patches'))
###
"""
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
"""