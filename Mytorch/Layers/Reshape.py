import numpy as np
class Reshape:
  def __init__(self,input_shape,output_shape):
    self.layer_type='reshape'
    self.input_shape=input_shape
    self.output_shape=output_shape
  def forward(self,input):
    return np.reshape(input,self.output_shape)
  def backward(self,output_gradient,prev_layer_type):
    output_gradient=output_gradient.T
    return np.reshape(output_gradient,self.input_shape) ,self.layer_type 
