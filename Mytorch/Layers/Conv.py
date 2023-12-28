import numpy as np
from scipy import signal

class Conv:
    def __init__(self, kernel_size, input_depth, output_depth):
        self.input_depth = input_depth
        self.output_depth = output_depth
        self.kernel_size = kernel_size
        self.kernel_shape = (self.output_depth, self.input_depth, self.kernel_size, self.kernel_size)
        self.weights = np.random.randn(*self.kernel_shape)
        self.bias = None
        self.layer_type = 'Conv'
        self.iterations=0
        self.m_weights=None 
        self.m_bias=None
        self.v_weights=None
        self.v_bias=None

    def get_bias(self, input_shape):
        input_depth, input_height, input_width = input_shape
        self.output_shape = (self.output_depth, input_height - self.kernel_size + 1, input_width - self.kernel_size + 1)
        self.bias = np.random.randn(*self.output_shape)
        return self.bias

    def forward(self, x):
        self.input = x
        if self.bias is not None:
            self.output = np.copy(self.bias)
        else:
            self.output = np.copy(self.get_bias(self.input.shape))
        for i in range(self.output_depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.weights[i, j], 'valid')
        return self.output

    def backward(self, error, prev_layer_type):
        self.dweights = np.zeros_like(self.weights)
        dinput = np.zeros_like(self.input)
        self.dbias = np.zeros_like(self.bias)
        for i in range(self.output_depth):
            for j in range(self.input_depth):
                self.dweights[i, j] = signal.correlate2d(self.input[j], error[i], 'valid')
                self.dbias[i] = error[i]
                dinput[j] += signal.convolve2d(error[i], self.weights[i, j], 'full')

        return dinput, self.layer_type
