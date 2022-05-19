from resunetexplorer.utils import get_submodule_str, get_submodule, get_output_shape, get_number_maps, model_up_to

class ExtractResUNetLayers:
   """Layers extractor class for PyTorch ResUNet.

      Receives the model, the network part (encoder or decoder) and the names of the layers to be extracted

      Parameters
      ----------
      model : Pytorch model      
      layers_paths : list
          Contains the paths to the layers.
          e.g. ['encoder.resblock1', 'encoder.resblock2', ...]

    """

    def __init__(self, model , layers_paths: list):
        
        self.model = model
        self.layers_paths = layers_paths

    # FIXME: A obtenção do submodulo não está funcionando para camadas da parte 'decoder' da rede. 
    def get_layers(self):
        """
    """
        layers_dict = {
            "network_part": [],
            "n_maps": [],
            "layer": [],

        }

        for i, path in enumerate(self.layers_paths):
            # Get desired layer from model
            layer = get_submodule(self.model, path)
            
            n_maps = get_number_maps(self.model, layer)
            # Dict appending
            layers_dict["network_part"].append(path)
            layers_dict["n_maps"].append(n_maps)
            layers_dict["layer"].append(layer)

        return layers_dict     