# I won't be including comments from prior steps
# ReLU provides non-linearity to the network, great
# However, the values it provides us with are meaningless in terms of calculating a loss
# The values provided are not relative to each other, so we have no way of really rectifying mistakes
# Thus we introduce Softmax activation into the equation
# Softmax activation normalises the values into a probability distribution between 0 and 1
# This is done so that the predicted values can be compared to the target values and be rectified

import numpy as np;

X = [[1, 2, 3, 4, 5],
    [1.1, 2.3, 3.5, 4.1, 5.2],
    [5.4, 2.3, 4.5, 3.2, 5.3]]

class Layer_Creation:
    def __init__(self, n_inputs, n_neurons):
        self.inputs = n_inputs
        self.weights = 0.1 * (np.random.randn(n_inputs, n_neurons))
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs) # np.maximum() returns the biggest out of two values; perfect to create the ReLU function

class activation_Softmax:
  def forward(self, inputs):
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # Calculating euler's number to the power of each value and also subtracting np.max(inputs... stabilises the exp values by preventing an overflow error.
    # We do np.max to subtract the largest value from each row before taking the exponential. np.exp() grows suber fast. Big inputs explode into very large values, subtracting the max helps avoid overflow and keeps the math stable
    # We use axis=1 to operate across all columns, so it gives the max per row, axis=0 would give you the max per column, which isn't what we want in this case
    # keepdims=True makes sure the result of np.max keeps the same number of dimensions in order to prevent broadcasting from breaking
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True) # Normalises each row so that the values add to 1 in order to provide us with a proper probability distribution
    self.output = probabilities

layer1 = Layer_Creation(5, 6)
layer2 = Layer_Creation(6, 4)

activation1 = activation_ReLU()
activation2 = activation_Softmax()

layer1.forward(X)
activation1.forward(layer1.output)
layer2.forward(activation1.output) # Passing the output of the ReLU activation of layer 1 into layer 2
activation2.forward(layer2.output) # Passing the output of layer 2 into the softmax activation function
print(activation1.output)
