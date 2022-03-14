import os

from pyforest import *
from tools.yair.plot_utils import plot_tensor

def get_layers_dict(model):
    layers_dict = dict([*model.named_modules()])
    return layers_dict


def plot_layer_filter(model, layer_name, show=False, output_file_path=None):
    layers_dict = get_layers_dict(model)
    layer_parameters = [*layers_dict[layer_name].parameters()]
    try:
        if len(layer_parameters) >= 1 and layer_parameters[0].shape[2]>1:
            plot_tensor(layer_parameters[0], show=show, title=f'{model._get_name()} - {layer_name}', output_file_path=output_file_path)
        # if output_dir_path is not None:
        #     output_dir_path = os.path.join(output_dir_path, 'filters_visualization')
    except:
        print(f'Layer {layer_name} failed to be plotted')

def print_model(model, output_dir_path):
    os.makedirs(output_dir_path, exist_ok=True)

    with open(os.path.join(output_dir_path, 'model_architecture.txt'), 'w') as f:
        f.write(str(model))

    with open(os.path.join(output_dir_path, 'model_layers.txt'), 'w') as f:
        for name, param in model.named_parameters():
            print(name)
            print(param.shape)
            f.write(f'{name}\n')
            f.write(f'{str(param.shape)}\n\n')


