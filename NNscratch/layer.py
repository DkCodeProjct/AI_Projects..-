import numpy as np
np.random.seed(0)

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

class DenseLayer:
    def __init__(self, nInputs, nNeurons):
        self.weights = 0.10* np.random.randn(nInputs, nNeurons)
        self.biases = np.zeros((1, nNeurons))
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = DenseLayer(4, 5) # 4 is for how many fratures in X, it's 4, and 3samples
layer2 = DenseLayer(5, 2) # :: And this 5 And 3 are neurons, 
                          #    5 is the Hidden layer and 3 is the outputLayer
                          #    you can deside it, but the layer1=5 became layer2 input=5


layer1.forward(X)
#print(layer1.output)
layer2.forward(layer1.output) # so layer1 get input as X_dataset 
print(layer2.output)          # and layer2 get input as layer1 output --> and output with 3 output Neuron

#print(0.10*np.random.randn(4, 3))