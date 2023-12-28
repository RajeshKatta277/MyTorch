import numpy as np
from ..Layers import FCLayer
class GradientDescent:
    def __init__(self,layers=None,learning_rate=0.01):
        self.model_layers=layers
        self.learning_rate=learning_rate
    def step(self):
        for layer in reversed(self.model_layers):
            if layer.layer_type=='fc' or layer.layer_type=='Conv':
                layer.weights=layer.weights-self.learning_rate*layer.dweights
                layer.bias=layer.bias-self.learning_rate*layer.dbias
