import numpy as np

class BinaryCrossEntropy:
    def __init__(self):
        pass

    def Loss(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = np.mean(np.mean(- y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred), axis=0))
        return loss

    def backward(self, net, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        error = np.mean(-y_true / y_pred + (1 - y_true) / (1 - y_pred), axis=0)
        error = error.reshape((1, len(error))) 
        prev_layer_type='fc'
        for layer in reversed(net.layers):
    
            error,prev_layer_type = layer.backward(error,prev_layer_type) 
