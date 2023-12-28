import numpy as np
class Sigmoid:
    def __init__(self):
        self.layer_type='activation'
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self,x):
        sigmoid_x = self.sigmoid(x)
        return sigmoid_x * (1 - sigmoid_x)

    def forward(self,input):
        self.inputs=input
        self.output=self.sigmoid(self.inputs)
        return self.output
    
    def backward(self,dloss,prev_layer_type):
        if prev_layer_type=='fc':
            return (self.sigmoid_derivative(self.inputs)*dloss.T).T,self.layer_type
        elif prev_layer_type=='Conv' or prev_layer_type=='reshape':
            return self.sigmoid_derivative(self.inputs)*dloss, self.layer_type
