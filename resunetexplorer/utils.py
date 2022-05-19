'''Utility functions and classes for working with Pytorch modules'''

from collections import OrderedDict
from torch import nn
import torch
import functools


class ActivationSampler(nn.Module):
    '''Generates a hook for sampling a layer activation. Can be used as

    sampler = ActivationSampler(layer_in_model)
    output = model(input)
    layer_activation = sampler()

    '''

    def __init__(self, model):
        super().__init__()
        self.model_name = model.__class__.__name__
        self.activation = None
        model.register_forward_hook(self.get_hook())

    def forward(self, x=None):
        return self.activation

    def get_hook(self):
        def hook(model, input, output):
            self.activation = output
        return hook

    def extra_repr(self):
        return f'{self.model_name}'

def get_submodule_str(model, module):
    """Return a string representation of `module` in the form 'layer_name.sublayer_name...'
    """

    for name, curr_module in model.named_modules():
        if curr_module is module:
            module_name = name
            break

    return module_name

# Source: https://stackoverflow.com/a/31174427
def get_submodule(model, path: str, *default):
    """A function to get nested subobjects from model, given the nested attribute (submodule path) as string.  

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
        return functools.reduce(getattr, attrs, model)
    except AttributeError:
        if default:
            return default[0]
        raise

def get_output_shape(model, img_shape):

    input_img = torch.zeros(img_shape)[None, None]
    input_img = input_img.to(next(model.parameters()).device)
    output = model(input_img)
    return output[0].shape

def get_number_maps(model, module):
    """Return the number of feature maps of a layer, given the model and its module
    """
    # Get sub model
    sub_model = model_up_to(model, module)
    # Get the output shape of sub model
    n_maps = get_output_shape(sub_model, (1, 1))
    # Return number of feature maps
    return n_maps[0]

def model_up_to(model, module):
  """Return a new model with all layers in model up to layer `module`."""
  
  split_module_str = get_submodule_str(model, module)
  split_modules_names = split_module_str.split('.')
  module = model
  splitted_model = []
  name_prefix = ''
  for idx, split_module_name in enumerate(split_modules_names):
      for child_module_name, child_module in module.named_children():
          if child_module_name==split_module_name:
              if idx==len(split_modules_names)-1:
                  # If at last module
                  full_name = f'{name_prefix}{child_module_name}'
                  splitted_model.append((full_name, child_module))
              module = child_module
              name_prefix += split_module_name + '_'
              break
          else:
              full_name = f'{name_prefix}{child_module_name}'
              splitted_model.append((full_name, child_module))

  new_model = torch.nn.Sequential(OrderedDict(splitted_model))
  
  return new_model