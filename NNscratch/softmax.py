import math
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()



class DenseLayer:
    def __init__(self, nInputs, nNeurons):
        self.weights = np.random.randn(nInputs, nNeurons)
        self.biases = np.zeros((1, nNeurons))
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class ActivationSoftMAX:
    def forward(self, inputs):
        expVal = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        prob = expVal / np.sum(expVal, axis=1, keepdims=True)
        self.output = prob

X, y = spiral_data(samples=100, classes=3)

layer1 = DenseLayer(2, 4)
activation1 = ActivationReLU()

layer1.forward(X)
activation1.forward(layer1.output)

layer2 = DenseLayer(4, 3)
activation2 = ActivationReLU()

layer2.forward(activation1.output)
activation2.forward(layer2.output)

layer3 = DenseLayer(3, 2)
activation3 = ActivationSoftMAX()

layer3.forward(activation2.output)
activation3.forward(layer3.output)

print(activation3.output[:5])
