from resunetexplorer.utils import ActivationSampler
import torch
import matplotlib.pyplot as plt


class ExtractResUNetMaps:

  def __init__(self, model, dataset, device):
    self.model = model
    self.dataset = dataset
    self.device = device

  def get_feature_maps(self, img_idx, layer):
    """
    Function that receives an image index and a ResUNet layer, send the 
    feature maps of its respective image and layer to CPU and returns the 
    feature maps.  
    """
     
    sampler = ActivationSampler(layer)

    img, label = self.dataset[img_idx]
    with torch.no_grad():
        img = img.to(self.device)[None]
        self.model(img);
        
    layer_feature_maps = sampler().to('cpu')[0]
    
    return layer_feature_maps

  # TODO: Descrever a função
  def get_multiple_feature_maps(self, img_idx, layers):
    layers_fm_list = []

    for i in range(len(layers)):
      layers_fm_list.append(self.get_feature_maps(img_idx, layers[i]))

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
  # TODO: Parâmetro opcional para salvar a figura em determinada extensão em determinado diretório.
  # TODO: Parâmetro opcional para mostrar as imagens na mesma escala de valor de intensidade. Deve ser o mín
  # e máx entre todas as figuras da camada. 
  def show_feature_maps(self, layers, layers_fm_list, img_idx = None,  maps_idx = None, fig_size = (20, 75), ncols = 4):    
      
    n_layers = len(layers_fm_list)
    nrows = 64//ncols

    if maps_idx != None:

      qty_maps = len(maps_idx)      
    
      for layer_idx in range(n_layers):
        
        plt.figure(figsize = fig_size)

        for idx in range(qty_maps):
          # Show feature map
          map_idx = maps_idx[idx] 
          fig = plt.subplot(nrows, ncols, idx+1)    
          ax = plt.imshow(layers_fm_list[layer_idx][map_idx], 'gray')
          layer_path = layers['network_part'][layer_idx]
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
        
        plt.figure(figsize = fig_size)

        for idx in range(64):
          # Show feature map
          map_idx = idx 
          fig = plt.subplot(nrows, ncols, idx+1)    
          ax = plt.imshow(layers_fm_list[layer_idx][map_idx], 'gray')
          layer_path = layers['network_part'][layer_idx]
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
  # TODO: Parâmetro opcional para mostrar as imagens na mesma escala de valor de intensidade. 
  def show_kernels_per_channel(self, layers,  kernels_list,  kernels_idx = None, channel_idx = 0, fig_size = (20, 75), ncols = 4):    
        
      n_layers = len(kernels_list)
      nrows = 64//ncols

      if kernels_idx != None:

        qty_maps = len(kernels_idx)      
      
        for layer_idx in range(n_layers):
          
          plt.figure(figsize = fig_size)

          for idx in range(qty_maps):
            # Show kernel
            kernel_idx = kernels_idx[idx] 
            fig = plt.subplot(nrows, ncols, idx+1)    
            ax = plt.imshow(kernels_list[layer_idx][kernel_idx][channel_idx], 'gray')
            layer_path = layers['network_part'][layer_idx]
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
          
          plt.figure(figsize = fig_size)

          for idx in range(64):
            # Show kernel
            kernel_idx = idx 
            fig = plt.subplot(nrows, ncols, idx+1)    
            ax = plt.imshow(kernels_list[layer_idx][kernel_idx][channel_idx], 'gray')
            layer_path = layers['network_part'][layer_idx]
            # Plot title
            plt.title(f'Kernel {kernel_idx} - Channel {channel_idx} - {layer_path}')
            # Hide axis
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            # Adjust space between plots
            plt.subplots_adjust(wspace=0.02, hspace=0.0)
            plt.tight_layout() 

  def show_channels_per_kernel(self, layers, kernels_list,  kernel_idx = 0, channels_idx = None, fig_size = (20, 75), ncols = 4):    
      
    n_layers = len(kernels_list)
    nrows = 64//ncols

    if channels_idx != None:

      qty_channel = len(channels_idx)      
    
      for layer_idx in range(n_layers):
        
        plt.figure(figsize = fig_size)

        for idx in range(qty_channel):
          # Show kernel
          channel_idx = channels_idx[idx] 
          fig = plt.subplot(nrows, ncols, idx+1)    
          ax = plt.imshow(kernels_list[layer_idx][kernel_idx][channel_idx], 'gray')
          layer_path = layers['network_part'][layer_idx]
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
        
        plt.figure(figsize = fig_size)

        for idx in range(64):
          # Show kernel
          channel_idx = idx 
          fig = plt.subplot(nrows, ncols, idx+1)    
          ax = plt.imshow(kernels_list[layer_idx][kernel_idx][channel_idx], 'gray')
          layer_path = layers['network_part'][layer_idx]
          # Plot title
          plt.title(f'Kernel {kernel_idx} - Channel {channel_idx} - {layer_path}')
          # Hide axis
          ax.axes.get_xaxis().set_visible(False)
          ax.axes.get_yaxis().set_visible(False)
          # Adjust space between plots
          plt.subplots_adjust(wspace=0.02, hspace=0.0)
          plt.tight_layout()
