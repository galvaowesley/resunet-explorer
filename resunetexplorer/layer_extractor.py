class ExtractResUNetLayers:
  """Layers extractor class for PyTorch ResUNet. 

    Receives the model, the network part (encoder or decoder) and the names of the layers to be extracted

    Parameters
    ----------
    model : torchtrainer.models.resunet.ResUNet
        Directory containing the images to be read
    network_part : string
        A string the represents the name of ResUNet Network part. network_part = 'encoder' or 'decoder'
    layer_names : list 
        Contains the names of layers belonging network_part. 
        e.g. If network_part == 'encoder', a possible list of layers is: ['resblock1', 'resblock2', ...]
    
  """

  def __init__(self, model:str, network_part:str, layer_names:list):
    self.model = model
    self.network_part = network_part
    self.layer_names = layer_names

    # Get model network part (encoder or decoder) 
    self.model_network_part = getattr(self.model, self.network_part)
  
  def get_number_maps(self, layer):
    """
    """   
    # Get number of feature maps of layer
    n_maps = layer.conv1.out_channels

    return n_maps
  # TODO Tornar isso mais genérico, recebe o nome da layer por completo na hierarquia do modelo: ex: encoder.resblock1
  def get_layers(self):
    """
    """
    layers_dict = {
      "layer_name":[], 
      "network_part":[],
      "n_maps":[], 
      "layer":[], 

    }

    for i, name in enumerate(self.layer_names):
      # Get desired layer from model 
      layer = getattr(self.model_network_part, name)
      n_maps = self.get_number_maps(layer)
      
      # Dict appending
      layers_dict["layer_name"].append(name)
      layers_dict["network_part"].append(self.network_part)      
      layers_dict["n_maps"].append(n_maps)
      layers_dict["layer"].append(layer)

    return layers_dict
