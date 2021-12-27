import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils

# Based on stackoverflow #55594969
def plot_tensor(tensor, ch=0, allkernels=False, nrow=8, padding=1):
    tensor = tensor.cpu()
    n, c, w, h = tensor.shape
    if allkernels:
        tensor = tensor.view(n * c, -1, w, h)
    elif c != 3:
        tensor = tensor[:, ch, :, :].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure(figsize=(nrow, rows))
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.show()