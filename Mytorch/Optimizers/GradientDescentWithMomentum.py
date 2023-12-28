import numpy as np
from ..Layers import FCLayer
class GradientDescentWithMomentum:
    def __init__(self,layers=None,momentum=0.9,learning_rate=0.01):
        self.model_layers=layers
        self.momentum=momentum
        self.learning_rate=learning_rate
    def step(self):
        for layer in reversed(self.model_layers):
            if layer.layer_type=='fc' or layer.layer_type=='Conv':
                if layer.m_weights is None or layer.m_bias is None:
                    layer.m_weights = np.zeros_like(layer.weights)
                    layer.m_bias=np.zeros_like(layer.bias)
                layer.m_weights = self.momentum * layer.m_weights + (1 - self.momentum) * layer.dweights
                layer.m_bias=self.momentum*layer.m_bias + (1-self.momentum)*layer.dbias

                layer.weights -= self.learning_rate * layer.m_weights
      
                layer.bias-=self.learning_rate*layer.m_bias
    def zero_grad(self):
        for layer in reversed(self.model_layers):
            if layer.layer_type=='fc' or layer.layer_type=='Conv':
                layer.m_weights=np.zeros_like(layer.weights)
                layer.m_bias=np.zeros_like(layer.bias)
