import numpy as np

class Softmax:
    def __init__(self):
        self.layer_type='activation'
    def forward(self,input):
        output=np.exp(input)
        self.output=output/np.sum(output)
        return self.output
    def backward(self,dloss):
        s=self.output.reshape(-1,1)
        return np.dot(dloss,(np.diagflat(s)-np.dot(s,s.T))) ,self.layer_type
