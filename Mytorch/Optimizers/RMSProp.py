import numpy as np
from ..Layers import FCLayer
class RMSProp:
    def __init__(self,layers=None,momentum=0.9,learning_rate=0.01):
        self.model_layers=layers
        self.learning_rate=learning_rate
        self.momentum=momentum
    def step(self):
        for layer in reversed(self.model_layers):
            if layer.layer_type=='fc' or layer.layer_type=='Conv':
                if layer.v_weights is None or layer.v_bias is None:
                    layer.v_weights = np.zeros_like(layer.weights)
                    layer.v_bias=np.zeros_like(layer.bias) 

                dw=layer.dweights**2
                db=layer.dbias**2
                layer.v_weights = self.momentum * layer.v_weights + (1 - self.momentum) *dw
                layer.v_bias=self.momentum*layer.v_bias + (1-self.momentum)*db

                layer.weights -= self.learning_rate * (layer.dweights/(np.sqrt(layer.v_weights)+1e-8))
      
                layer.bias-= self.learning_rate *layer.dbias/(np.sqrt(layer.v_bias)+1e-8)
    def zero_grad(self):
        for layer in reversed(self.model_layers):
            if layer.layer_type=='fc' or layer.layer_type=='Conv':
                layer.v_weights=np.zeros_like(layer.weights)
                layer.vbias=np.zeros_like(layer.bias)
