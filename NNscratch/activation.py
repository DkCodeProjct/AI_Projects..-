import numpy as np
import nnfs

from nnfs.datasets import spiral_data
nnfs.init()


#
##  Activation Funcs help to Model non-linear data,
#    Data that Didnt linearly Separable, 
#    And make Very curvy squgle lines to fit the data
# 
# #



X, y = spiral_data(100, 3)


class DenseLayer:
    def __init__(self, nInputs, nNeurons):
        self.weights = 0.10*np.random.randn(nInputs, nNeurons)
        self.biases = np.zeros((1, nNeurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)



layer1 = DenseLayer(2, 4)
layer1.forward(X)
activation1 = ActivationReLU()
activation1.forward(layer1.output)


print(activation1.output)