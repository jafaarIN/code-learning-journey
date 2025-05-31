# I won't be including comments from prior steps
# Activation functions determine what point a neuron should activate or deactivate
# There are several activation functions to be used, we do not use linear activation functions however
# Otherwise the model will not be able to fit for non linear problems
# Thus we use ReLU activation (Rectified Linear Unit) or Sigmoid activation
# Rectified Linear unit works like so: if the input is negative, the output is 0; otherwise the output is the input
# Sigmoid activation works through the following formula: 1/(1 + e^(-x)) where e is the constant for euler's number. However, this type of function faces the vanishing gradient issue, so ReLU will be used instead

import numpy as np;

X = [[1, 2, 3, 4, 5],
    [1.1, 2.3, 3.5, 4.1, 5.2],
    [5.4, 2.3, 4.5, 3.2, 5.3]]

class Layer_Creation:
    def __init__(self, n_inputs, n_neurons):
        self.inputs = n_inputs
        self.weights = 0.1 * (np.random.randn(n_inputs, n_neurons))
        self.biases = np.zeros((1, n_neurons)) # Creating the biases array (1 row of "n_neurons" amount of columns) where each value is 0
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class activation_ReLU: # We create a class for the activation function
    def forward(self, inputs):
        self.output = np.maximum(0, inputs) # np.maximum() returns the biggest out of two values; perfect to create the ReLU function

layer1 = Layer_Creation(5, 6)
layer1.forward(X)

activation1 = activation_ReLU()
activation1.forward(layer1.output) # Performing the ReLU function on the output of the first layer

print(activation1.output)
