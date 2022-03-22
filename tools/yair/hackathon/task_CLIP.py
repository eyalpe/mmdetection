import glob
import shutil
import PIL.ImageShow
from PIL import ImageDraw
import numpy as np
import torch
import clip
from PIL import Image
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from tools.yair.save_output import SaveOutput
from tools.yair.plot_utils import plot_tensor_ontop_image
from tools.yair.visualization_utils import get_layers_dict
import tools.yair.hackathon.hackathon_utils as hku
import animal_names
import os



# Show filters before loading the trained net:
device = 'cuda:0'
MMDET_DIR = hku.get_mmdet_root()
RESULTS_ROOT_DIR = MMDET_DIR + '/../../results/'

config_file_path = MMDET_DIR + '/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
output_dir_root_path = RESULTS_ROOT_DIR + '/task_CLIP'
patches_out_dir=os.path.join(output_dir_root_path, 'patches')

def generate_patches():
    input_image_path = '/home/eyalper/hackathon2022/data/depositphotos_93215318-stock-photo-brown-bear-and-other-asian.jpg'
    checkpoint_file_path = MMDET_DIR + '/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    model = init_detector(config_file_path, checkpoint_file_path, device=device)
    results = inference_detector(model, input_image_path)
    show_result_pyplot(model, input_image_path, results,  patches_out_dir=patches_out_dir)


def run_clip_on_patches_dir():
    classes_for_clip = animal_names.animals_list
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    for input_image_path in glob.glob(patches_out_dir + '/*.png'):
        img = Image.open(input_image_path)
        image = preprocess(img).unsqueeze(0).to(device)
        text = clip.tokenize(classes_for_clip).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        max_prob_index = np.argmax(probs)
        class_name = classes_for_clip[max_prob_index]
        class_prob = probs[0, max_prob_index]
        class_prob_txt = f'Class: {class_name}\nProbability: {class_prob}'
        patch_name = os.path.basename(input_image_path)
        print(f'Image: {patch_name}, {class_prob_txt}' )
        ImageDraw.Draw(img).text((0,0), class_prob_txt, (50,50,50))
        os.makedirs(patches_out_dir + '/../after_CLIP/', exist_ok=True)
        img.save(patches_out_dir + '/../after_CLIP/' + patch_name)




if __name__ == '__main__':
    run_clip_on_patches_dir()