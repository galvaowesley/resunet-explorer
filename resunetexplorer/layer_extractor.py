import functools

# TODO: Usar as funções model_up_to(model, module) e get_output_shape() para capturar o 
# tamanho da saída e tornar mais geral. 
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

    def __init__(self, model: str, layers_paths: list):
        
        self.model = model
        self.layers_paths = layers_paths

   

    def get_number_maps(self, layer):
        """
        """
        # Get number of feature maps of layer
        n_maps = layer.out_channels

        return n_maps
    
     # Source: https://stackoverflow.com/a/31174427
    def _rgetattr(self, path: str, *default):
        """
        A function to get nested subobjects from model, given the nested attribute as string.  

        Parameters
        ----------
        
        path: 'attr1.attr2.etc'
        
        default: Optional default value, at any point in the path

        Returns 
        ----------
        model.attr1.attr2.etc
        """

        attrs = path.split('.')
        try:
            return functools.reduce(getattr, attrs, self.model)
        except AttributeError:
            if default:
                return default[0]
            raise


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
            layer = self._rgetattr(path)
            n_maps = self.get_number_maps(layer)

            # Dict appending
            layers_dict["network_part"].append(path)
            layers_dict["n_maps"].append(n_maps)
            layers_dict["layer"].append(layer)

        return layers_dict