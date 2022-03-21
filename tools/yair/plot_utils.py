import os

import matplotlib.pyplot as plt
import torch
from pyforest import *
from torchvision import utils
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import copy

# Based on stackoverflow #55594969
def plot_tensor(tensor, channle_idx=0, allkernels=False, ncols=8, padding=1, show=False, output_file_path=None, title=None):
    tensor = tensor.cpu()
    if len(tensor.shape) != 4:
        print(f'len(tensor.shap) = {len(tensor.shape)} != 4')
        return

    n, c, w, h = tensor.shape
    if allkernels:
        tensor = tensor.view(n * c, -1, w, h)
    elif c > 3:
        tensor = reduce_tensor_dim(tensor)
    elif c == 2:
        tensor = tensor[:, channle_idx, :, :].unsqueeze(dim=1)

    cols = np.min((tensor.shape[0] // ncols + 1, 64))
    grid = utils.make_grid(tensor, nrow=ncols, normalize=True, padding=padding)
    plt.figure(figsize=(ncols, cols))
    plt.imshow(grid.detach().numpy().transpose((1, 2, 0)))

    if title is not None:
        plt.title(title)

    if show:
        plt.show()
    if output_file_path is not None:
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        plt.savefig(output_file_path)


def rgb_2_red(tensor):
    tensor[..., 0] = (tensor[..., 0] + tensor[..., 1] + tensor[..., 2]) / 3
    tensor[..., 1] = tensor[..., 2] = np.zeros(tensor[..., 0].shape)
    return tensor

from PIL import Image
def plot_tensor_ontop_image(tensor, image_path, output_path=None):
    tensor = tensor[0, ...].cpu().numpy()
    tensor = np.transpose(tensor, (1, 2, 0))
    tensor = rgb_2_red(tensor)
    tensor = tensor.astype(np.uint8)
    rpn_cls_output = Image.fromarray(tensor)
    image = Image.open(image_path)
    l = rpn_cls_output.resize(image.size)
    image_with_rpn_cls = Image.blend(image, l, 0.3)
    image_with_rpn_cls.show()
    if output_path is not None:
        image_with_rpn_cls.save(output_path)

def reduce_tensor_dim(tensor):
    orig_shape = tensor.shape
    new_tensor = torch.zeros((orig_shape[0], 3, orig_shape[2], orig_shape[3]))
    for i in range(tensor.shape[0]): # range(5):  #
        array_ = torch.reshape(tensor[i, ...], (orig_shape[1],orig_shape[2]* orig_shape[3]))
        array_ = array_.transpose(1,0).detach()
        print(array_[0,0])
        # tsne = TSNE(n_components=3, perplexity=10, early_exaggeration=6, random_state=9)
        # array = tsne.fit_transform(copy.deepcopy(array_))
        pca = PCA(n_components=3)
        array = pca.fit_transform(array_)
        print(array[0,0])
        array = torch.from_numpy(array)
        array.transpose(0, 1)
        new_tensor[i, ...] = torch.reshape(array, (3, orig_shape[2], orig_shape[3])).clone()
    return new_tensor

