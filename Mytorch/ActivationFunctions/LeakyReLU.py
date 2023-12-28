import numpy as np
class LeakyReLU:
    def __init__(self,alpha=0.2):
        self.alpha=alpha
        self.layer_type='activation'
    
    def leakyrelu(self,x):
        return np.where(x>=0,x,self.alpha*x)

    def leakyrelu_derivative(self,x):
        return np.where(x >= 0, 1, self.alpha)
    
    def forward(self,input):
        self.inputs=input
        self.output=self.leakyrelu(self.inputs)
        return self.output
    
    def backward(self,dloss,prev_layer_type):
        if prev_layer_type=='fc':
            return (self.leakyrelu_derivative(self.inputs)*dloss.T).T ,self.layer_type 
        elif prev_layer_type=='Conv' or prev_layer_type=='reshape':
            return self.leakyrelu_derivative(self.inputs)*dloss,self.layer_type
                
