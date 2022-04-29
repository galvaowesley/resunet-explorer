from torch import nn


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