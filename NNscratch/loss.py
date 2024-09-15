'''
log is solving for x /\
e ** x = b 
'''

import numpy as np
import math
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

"""
b = 5.2
print(np.log(b)) # == 1.6486586255873816

log_b = 1.6486586255873816
logorithm = math.e ** log_b
print(round(logorithm, 2)) # == 5.2
## math.e ** log_b = b
##
## e ** x = b
"""
"""
softMaxOutput = [0.7, 0.1, 0,2]
targetOutput = [1, 0, 0] # one hot-encoding

loss = -np.sum([targt + np.log(output) for output, targt in zip(softMaxOutput, targetOutput)])
print(loss)
"""

class DenseLayer:
    def __init__(self, nInputs, nNeurons):
        self.weights = 0.10*np.random.randn(nInputs, nNeurons)
        self.biases = np.zeros((1, nNeurons))
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
    
class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
    
class ActivationSoftMax:
    def forward(self, inputs):
        expVal = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        prob = expVal / np.sum(expVal, axis=1, keepdims=True)
        self.output = prob



            
class Loss:
    def calculateLoss(self, output, y):
        sampleLosses = self.forward(output, y)
        dataLoss = np.mean(sampleLosses)
        return dataLoss
    
class CatagoricalCrossEntropy(Loss):
    def forward(self, yPred, yTrue):
        samples = len(yPred)
        yPredClip = np.clip(yPred, 1e-7, 1-1e-7)

        if len(yTrue.shape) == 1:
            correctConfidence = yPredClip[range(samples)]
        elif len(yTrue.shape) == 2:
            correctConfidence = np.sum(yPredClip * yTrue, axis=1)

        negLossLiklyhood = -np.log(correctConfidence)
        return negLossLiklyhood
            

X,  y = spiral_data(samples=100, classes=3)

layer1 = DenseLayer(2, 4)
activation1 = ActivationReLU()

layer1.forward(X)
activation1.forward(layer1.output)

layer2 = DenseLayer(4, 3)
activation2 = ActivationReLU()

layer2.forward(activation1.output)
activation2.forward(layer2.output)

layer3 = DenseLayer(3, 2)
activation3 = ActivationSoftMax()

layer3.forward(activation2.output)
activation3.forward(layer3.output)

print(activation3.output[:5])

lossFunc = CatagoricalCrossEntropy()
loss = lossFunc.calculateLoss(activation3.output, y)
print(loss)