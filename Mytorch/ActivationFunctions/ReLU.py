import numpy as np
class ReLU:
    def __init__(self):
        self.layer_type='activation'
    
    def relu(self,x):
        return np.maximum(0, x)

    def relu_derivative(self,x):
        return np.where(x > 0, 1, 0)
    
    def forward(self,input):
        self.inputs=input
        self.output=self.relu(self.inputs)
        return self.output
    
    def backward(self,dloss,prev_layer_type):
        if prev_layer_type=='fc':
            return (self.relu_derivative(self.inputs)*dloss.T).T,self.layer_type
        elif prev_layer_type=='Conv' or prev_layer_type=='reshape':
            return self.relu_derivative(self.inputs)*dloss,self.layer_type
