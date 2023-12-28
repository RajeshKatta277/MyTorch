import numpy as np

class Tanh:
    def __init__(self):
        self.layer_type='activation'
    def tanh(self,x):
        return np.tanh(x);

    def tanh_derivative(self,x):
        return 1-np.tanh(x)**2

    def forward(self,input):
        self.inputs=input
        self.output=self.tanh(self.inputs)
        return self.output
    
    def backward(self,dloss,prev_layer_type):
        if prev_layer_type=='fc':
            return (self.tanh_derivative(self.inputs)*dloss.T).T,self.layer_type
        elif prev_layer_type=='Conv' or prev_layer_type=='reshape':
            return self.tanh_derivative(self.inputs)*dloss,self.layer_type
