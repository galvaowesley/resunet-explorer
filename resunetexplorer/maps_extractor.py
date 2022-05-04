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

  def get_multiple_feature_maps(self, img_idx, layers):
    layers_fm_list = []

    for i in range(len(layers)):
      layers_fm_list.append(self.get_feature_maps(img_idx, layers[i]))

    return layers_fm_list
  

  # TODO: Se não passar nenhum índice, mostrar uma figura com todas as feature maps para cada layer. Um grid plot. 
  # TODO: Organizar os plots em mais linhas que colunas. Por exemplo 128 plots por 4 colunas = 32 linhas
  # TODO: Parâmetro opcional para mostrar as imagens na mesma escala de valor de intensidade. 
  def show_feature_maps(img_idx: int, layer_names: list, network_part: str, layers_fm_list, maps_idx = None, fig_size = [25, 23]):


    qty_maps = len(maps_idx)
    print(layer_names)
    n_layers = len(layer_names)

    for layer_idx in range(n_layers):
      plt.figure(figsize = fig_size)

      for idx in range(qty_maps):
        # Show feature map
        map_idx = maps_idx[idx] 
        fig = plt.subplot(layer_idx + 1, qty_maps, idx+1)    
        ax = plt.imshow(layers_fm_list[layer_idx][map_idx], 'gray')
        # Plot title
        plt.title(f'Image {img_idx} - Feature map {map_idx} - {network_part}.{layer_names[layer_idx]}')
        # Hide axis
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        # Adjust space between plots
        plt.subplots_adjust(wspace=0.03, hspace=0)
        plt.tight_layout()