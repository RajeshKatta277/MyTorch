import numpy as np
class Network:
    def __init__(self):
        self.layers=[]
        
    def add_layers(self,layer):
        self.layers.append(layer)

    def predict(self,input_data):
        samples = len(input_data)
        result = []
        
        for i in range(samples):
          if self.layers[0].layer_type=='fc':
            output = input_data[i].T
          elif self.layers[0].layer_type=='Conv':
            output=input_data[i]  
          for layer in self.layers:
            output = layer.forward(output)
          result.append(output.T)

        result=np.array(result)
        result=result.reshape(result.shape[0],result.shape[2])

        return result


