import numpy as np

class CrossEntropy:
    def __init__(self):
        pass
    def Loss(self,y_true, y_pred):
        epsilon=1e-15
        y_pred=np.clip(y_pred,epsilon,1-epsilon)
        loss=np.mean(-y_true * np.log(y_pred),axis=0)
        loss=np.mean(loss)
        return loss
    def backward(self,net,y_true,y_pred):
        epsilon=1e-9
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        error=np.mean(-y_true/y_pred,axis=0)
        error=error.reshape((1,len(error)))
        prev_layer_type='fc'
        for layer in reversed(net.layers):
            error,prev_layer_type=layer.backward(error,prev_layer_type)
