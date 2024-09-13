#.//////////////////// 
#
# Get The Dot Product 
#
#.////////////////////

# Dot Product :: u ∙ v = (u1​ x v1​) + (u2​ x v2​) + (un​ x vn​)

import numpy as np
inputs = [1, 2, 3.6, 4.9]

weights = [[-0.3, 0.1, -1.22, 0.92 ],
           [0.24, 0.23, -0.34, 2.30],
           [0.56, 0.4, 0.5, -0.56]]

biases = [2, 3.4, 0.5]

layerOutput = []
for weight, bais in zip(weights, biases):
    neuronOutpt = 0
    for wight, nInput in zip(weight, inputs):
        neuronOutpt += wight * nInput
    neuronOutpt += bais
    layerOutput.append(round(neuronOutpt, 2))
print(layerOutput)


# Dot Product
# when mul with matrix The order matters Weights should be first
dotProduct = np.dot(weights, inputs)
print(f'Dot: {dotProduct}')
