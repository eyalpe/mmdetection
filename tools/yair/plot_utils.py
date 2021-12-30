import os

import matplotlib.pyplot as plt
from pyforest import *
from torchvision import utils

# Based on stackoverflow #55594969
def plot_tensor(tensor, channle_idx=0, allkernels=False, nrow=8, padding=1, show=False, output_dir_path=None, title=None):
    tensor = tensor.cpu()
    n, c, w, h = tensor.shape
    if allkernels:
        tensor = tensor.view(n * c, -1, w, h)
    elif c != 3:
        tensor = tensor[:, channle_idx, :, :].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure(figsize=(nrow, rows))
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    if title is not None:
        plt.title(title)

    if show:
        plt.show()
    if output_dir_path is not None:
        os.makedirs(output_dir_path, exist_ok=True)
        plt.imsave(os.path.join(output_dir_path, ))


def rgb_2_red(tensor):
    tensor[..., 0] = (tensor[..., 0] + tensor[..., 1] + tensor[..., 2]) / 3
    tensor[..., 1] = tensor[..., 2] = np.zeros(tensor[..., 0].shape)
    return tensor

from PIL import Image
def plot_tensor_ontop_image(tensor, image_path):
    tensor = tensor[0, ...].cpu().numpy()
    tensor = np.transpose(tensor, (1, 2, 0))
    tensor = rgb_2_red(tensor)
    tensor = tensor.astype(np.uint8)
    rpn_cls_output = Image.fromarray(tensor)
    image = Image.open(image_path)
    l = rpn_cls_output.resize(image.size)
    image_with_rpn_cls = Image.blend(image, l, 0.3)
    image_with_rpn_cls.show()