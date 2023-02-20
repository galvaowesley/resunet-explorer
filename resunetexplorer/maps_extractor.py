from torch._C import LiteScriptModule
from resunetexplorer.utils import ActivationSampler
import torch
import matplotlib.pyplot as plt


class ExtractResUNetMaps:
    # TODO remover a entrada do dataset. A imagem já basta.
    def __init__(
            self,
            model,
            dataset=None,
            image=None,
            device='cpu'
    ):

        assert (dataset == None and image is not None) or (
                dataset is not None and image == None), f'dataset and image cannot both be None or have content. Only use dataset or image.'
        if image is not None:
            assert type(image) is torch.Tensor, f'Image is not a torch.Tensor type'

        self.model = model
        self.dataset = dataset
        self.image = image
        self.device = device

    def get_feature_maps(self, layer, img_idx=None):
        """
    Function that receives an image index and a ResUNet layer, send the 
    feature maps of its respective image and layer to CPU and returns the 
    feature maps.  
    """

        sampler = ActivationSampler(layer)

        if self.image is not None:
            with torch.no_grad():
                img = self.image.to(self.device)[None]
                self.model(img);
        else:
            img, label = self.dataset[img_idx]
            with torch.no_grad():
                img = img.to(self.device)[None]
                self.model(img);

        layer_feature_maps = sampler().to('cpu')[0]

        return layer_feature_maps

    # TODO: Descrever a função
    def get_multiple_feature_maps(self, layers, img_idx=None):
        layers_fm_list = []

        if self.image is not None:
            for i in range(len(layers)):
                layers_fm_list.append(self.get_feature_maps(layers[i], None))
        else:
            for i in range(len(layers)):
                layers_fm_list.append(self.get_feature_maps(layers[i], img_idx))

        return layers_fm_list

    def get_kernels(self, layer):
        kernels = layer.weight
        kernels_to_cpu = kernels.detach().to('cpu')
        return kernels_to_cpu

    def get_multiple_kernels(self, layers):
        kernels_list = []
        for i in range(len(layers)):
            kernels_list.append(self.get_kernels(layers[i]))

        return kernels_list

    # TODO: Descrever a função
    # TODO: Caso img_idx = None, então dar a possbilidade do usuário escolher a quantidade de primeiras feature maps a serem impressas.
    # FIX: Quando img_idx = None, por algum motivo o mapa 56 não está sendo impresso.
    # TODO: Parâmetro opcional para salvar a figura em determinada extensão em determinado diretório.
    def show_feature_maps(self, layers, layers_fm_list, img_idx=None, maps_idx=None, scalar_data=False,
                          fig_size=(20, 75), ncols=4):

        n_layers = len(layers_fm_list)
        nrows = 64 // ncols

        if maps_idx != None:

            qty_maps = len(maps_idx)

            for layer_idx in range(n_layers):

                plt.figure(figsize=fig_size)

                if scalar_data == True:
                    # Get min and max from set of feature maps
                    v_min = layers_fm_list[layer_idx].min()
                    v_max = layers_fm_list[layer_idx].max()
                else:
                    v_min = None
                    v_max = None

                for idx in range(qty_maps):
                    # Show feature map
                    map_idx = maps_idx[idx]
                    fig = plt.subplot(nrows, ncols, idx + 1)
                    ax = plt.imshow(layers_fm_list[layer_idx][map_idx], vmin=v_min, vmax=v_max, cmap='gray')
                    layer_path = layers['layer_path'][layer_idx]
                    # Plot title
                    plt.title(f'Image {img_idx} \nFeature map {map_idx} - {layer_path}')
                    # Hide axis
                    ax.axes.get_xaxis().set_visible(False)
                    ax.axes.get_yaxis().set_visible(False)
                    # Adjust space between plots
                    plt.subplots_adjust(wspace=0.02, hspace=0.0)
                    plt.tight_layout()

        else:

            for layer_idx in range(n_layers):

                plt.figure(figsize=fig_size)

                if scalar_data == True:
                    # Get min and max from set of feature maps
                    v_min = layers_fm_list[layer_idx].min()
                    v_max = layers_fm_list[layer_idx].max()
                else:
                    v_min = None
                    v_max = None

                for idx in range(64):
                    # Show feature map
                    map_idx = idx
                    fig = plt.subplot(nrows, ncols, idx + 1)
                    ax = plt.imshow(layers_fm_list[layer_idx][map_idx], vmin=v_min, vmax=v_max, cmap='gray')
                    layer_path = layers['layer_path'][layer_idx]
                    # Plot title
                    plt.title(f'Image {img_idx} \nFeature map {map_idx} - {layer_path}')
                    # Hide axis
                    ax.axes.get_xaxis().set_visible(False)
                    ax.axes.get_yaxis().set_visible(False)
                    # Adjust space between plots
                    plt.subplots_adjust(wspace=0.02, hspace=0.0)
                    plt.tight_layout()

    # TODO: Descrever a função
    # TODO: Parâmetro opcional para salvar a figura em determinada extensão em determinado diretório.
    def show_kernels_per_channel(self, layers, kernels_list, kernels_idx=None, channel_idx=0, scalar_data=False,
                                 fig_size=(20, 75), ncols=4):

        n_layers = len(kernels_list)
        nrows = 64 // ncols

        if kernels_idx != None:

            qty_maps = len(kernels_idx)

            for layer_idx in range(n_layers):

                plt.figure(figsize=fig_size)

                if scalar_data == True:
                    # Get min and max from set of kernels
                    v_min = kernels_list[layer_idx].min()
                    v_max = kernels_list[layer_idx].max()
                else:
                    v_min = None
                    v_max = None

                for idx in range(qty_maps):
                    # Show kernel
                    kernel_idx = kernels_idx[idx]
                    fig = plt.subplot(nrows, ncols, idx + 1)
                    ax = plt.imshow(kernels_list[layer_idx][kernel_idx][channel_idx], vmin=v_min, vmax=v_max,
                                    cmap='gray')
                    layer_path = layers['layer_path'][layer_idx]
                    # Plot title
                    plt.title(f'Kernel {kernel_idx} - Channel {channel_idx} - {layer_path}')
                    # Hide axis
                    ax.axes.get_xaxis().set_visible(False)
                    ax.axes.get_yaxis().set_visible(False)
                    # Adjust space between plots
                    plt.subplots_adjust(wspace=0.02, hspace=0.0)
                    plt.tight_layout()

        else:

            for layer_idx in range(n_layers):

                plt.figure(figsize=fig_size)

                if scalar_data == True:
                    # Get min and max from set of kernels
                    v_min = kernels_list[layer_idx].min()
                    v_max = kernels_list[layer_idx].max()
                else:
                    v_min = None
                    v_max = None

                for idx in range(64):
                    # Show kernel
                    kernel_idx = idx
                    fig = plt.subplot(nrows, ncols, idx + 1)
                    ax = plt.imshow(kernels_list[layer_idx][kernel_idx][channel_idx], vmin=v_min, vmax=v_max,
                                    cmap='gray')
                    layer_path = layers['layer_path'][layer_idx]
                    # Plot title
                    plt.title(f'Kernel {kernel_idx} - Channel {channel_idx} - {layer_path}')
                    # Hide axis
                    ax.axes.get_xaxis().set_visible(False)
                    ax.axes.get_yaxis().set_visible(False)
                    # Adjust space between plots
                    plt.subplots_adjust(wspace=0.02, hspace=0.0)
                    plt.tight_layout()

    def show_channels_per_kernel(self, layers, kernels_list, kernel_idx=0, channels_idx=None, scalar_data=False,
                                 fig_size=(20, 75), ncols=4):

        n_layers = len(kernels_list)
        nrows = 64 // ncols

        if channels_idx is not None:

            qty_channel = len(channels_idx)

            for layer_idx in range(n_layers):

                plt.figure(figsize=fig_size)

                if scalar_data == True:
                    # Get min and max from set of kernels
                    v_min = kernels_list[layer_idx].min()
                    v_max = kernels_list[layer_idx].max()
                else:
                    v_min = None
                    v_max = None

                for idx in range(qty_channel):
                    # Show kernel
                    channel_idx = channels_idx[idx]
                    fig = plt.subplot(nrows, ncols, idx + 1)
                    ax = plt.imshow(kernels_list[layer_idx][kernel_idx][channel_idx], vmax=v_max, cmap='gray')
                    layer_path = layers['layer_path'][layer_idx]
                    # Plot title
                    plt.title(f'Kernel {kernel_idx} - Channel {channel_idx} - {layer_path}')
                    # Hide axis
                    ax.axes.get_xaxis().set_visible(False)
                    ax.axes.get_yaxis().set_visible(False)
                    # Adjust space between plots
                    plt.subplots_adjust(wspace=0.02, hspace=0.0)
                    plt.tight_layout()

        else:

            for layer_idx in range(n_layers):

                plt.figure(figsize=fig_size)

                if scalar_data == True:
                    # Get min and max from set of kernels
                    v_min = kernels_list[layer_idx].min()
                    v_max = kernels_list[layer_idx].max()
                else:
                    v_min = None
                    v_max = None

                for idx in range(64):
                    # Show kernel
                    channel_idx = idx
                    fig = plt.subplot(nrows, ncols, idx + 1)
                    ax = plt.imshow(kernels_list[layer_idx][kernel_idx][channel_idx], vmax=v_max, cmap='gray')
                    layer_path = layers['layer_path'][layer_idx]
                    # Plot title
                    plt.title(f'Kernel {kernel_idx} - Channel {channel_idx} - {layer_path}')
                    # Hide axis
                    ax.axes.get_xaxis().set_visible(False)
                    ax.axes.get_yaxis().set_visible(False)
                    # Adjust space between plots
                    plt.subplots_adjust(wspace=0.02, hspace=0.0)
                    plt.tight_layout()
