import numpy as np
class FCLayer:
    def __init__(self,input_size,output_size):
        limit = np.sqrt(6 / (input_size + output_size))
        self.weights=np.random.uniform(-limit,limit,size=(output_size,input_size))
        self.bias=np.zeros((output_size,1))
        self.dweights=np.zeros_like(self.weights)
        self.dbias=np.zeros_like(self.bias)
        self.iterations=0
        self.m_weights=None 
        self.m_bias=None
        self.v_weights=None
        self.v_bias=None
        self.layer_type='fc'

    def forward(self,input_data):
        self.inputs=input_data
        self.output=np.dot(self.weights,self.inputs)+self.bias
        return self.output
    def backward(self,dloss,prev_layer_type):
        dinput=np.dot(dloss,self.weights)
        dweights=np.dot(dloss.T,self.inputs.T)
        self.dweights=dweights
        self.dbias=dloss.T
        return dinput,self.layer_type
