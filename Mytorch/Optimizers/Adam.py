import numpy as np
from ..Layers import FCLayer
class Adam:
    def __init__(self, layers=None, beta1=0.9, beta2=0.999,learning_rate=0.01, epsilon=1e-8):
        self.model_layers = layers
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def step(self):
        for layer in reversed(self.model_layers):
            if layer.layer_type=='fc' or layer.layer_type=='Conv':
                if layer.m_weights is None or layer.v_weights is None:
                    layer.m_weights = np.zeros_like(layer.weights)
                    layer.v_weights = np.zeros_like(layer.weights)
                    layer.m_bias = np.zeros_like(layer.bias)
                    layer.v_bias = np.zeros_like(layer.bias)


                layer.m_weights = self.beta1 * layer.m_weights + (1 - self.beta1) * layer.dweights
                layer.m_bias = self.beta1 * layer.m_bias + (1 - self.beta1) * layer.dbias

    
                layer.v_weights = self.beta2 * layer.v_weights + (1 - self.beta2) * (layer.dweights**2)
                layer.v_bias = self.beta2 * layer.v_bias + (1 - self.beta2) * (layer.dbias**2)


                m_hat_weights = layer.m_weights / (1 - self.beta1**(layer.iterations + 1))
                v_hat_weights = layer.v_weights / (1 - self.beta2**(layer.iterations + 1))
                m_hat_bias = layer.m_bias / (1 - self.beta1**(layer.iterations + 1))
                v_hat_bias = layer.v_bias / (1 - self.beta2**(layer.iterations + 1))

      
                layer.weights -= self.learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)
                layer.bias -= self.learning_rate * m_hat_bias / (np.sqrt(v_hat_bias) + self.epsilon)
                layer.iterations += 1
                
    def zero_grad(self):
        for layer in reversed(self.model_layers):
            if layer.layer_type=='fc' or layer.layer_type=='Conv':
                layer.m_weights = np.zeros_like(layer.weights)
                layer.v_weights = np.zeros_like(layer.weights)
                layer.m_bias = np.zeros_like(layer.bias)
                layer.v_bias = np.zeros_like(layer.bias)
                layer.iterations=0
