import numpy as np
class MeanSquaredError:
    def __init__(self):
        pass
    def Loss(self,y_true,y_pred):
        loss=np.mean(np.mean((y_true - y_pred) ** 2,axis=0))
        return loss

    def backward(self,net,y_true,y_pred):
        error= np.mean(2*(y_pred-y_true),axis=0)
        error=error.reshape(1,len(error))
        prev_layer_type='fc'
        for layer in reversed(net.layers):
            error,prev_layer_type = layer.backward(error,prev_layer_type)
