# Neural networks consist of inputs, 2 or more hidden layers, and of course, neurons
# Via the inputs and the introduction of weights and biases, the hidden layers perform activation functions on what's called the weighted sum
# The output of whatever comes from said activation function becomes the input of the next layer
# Once the initial input parses through all activation function processes, this is where backpropogation and cross-entropy occurs to relegate the calculated output relative to the desired output

# We begin by importing the 'numpy' library to aid in calculations such as dot products etc
import numpy as np;
# Then we initialise a set of inputs
X = [[1, 2, 3, 4, 5], # Like so, 'X' is the industry standard name for hard-coded inputs
    [1.1, 2.3, 3.5, 4.1, 5.2], # We use a batch of inputs rather than a singular array as it aids with generalasation
    [5.4, 2.3, 4.5, 3.2, 5.3]] # We must ensure that the input lengths are all the same to avoid a shape error

class Layer_Creation: # We will use a class to create an object for the layers we will create
    def __init__(self, inputs, n_neurons):
        self.inputs = inputs
        self.weights = 0.1 * (np.random.randn(len(inputs[0]), n_neurons))
        # np.randn(n_inputs, n_neurons) generates a 2d m x n array, where each element is picked out from a normal distribution (where the mean = 0 and the standard deviation is 1) forming a gaussian bell curve centered at 0
        # we mutliply each element in that array by 0.1 to essentially scale it down tenfold
        self.biases = np.zeros((1, n_neurons)) # Creating the biases array (1 row of "n_neurons" amount of columns) where each value is 0
        # We don't randomise the bias set to begin with as we do with weights because it is more effective for the layers to compute the biases than to introduce unnecessary noise before the model even learns anything
    def forward(self):
        self.output = np.dot(self.inputs, self.weights) + self.biases # We simply compute the dot product of the inputs and weights, then add the biases
        # y = mx + c, this equation aligns pretty well with what we've just done
        # Where y is the output, m is the weight, x is the input and c is the bias which acts as an offset as we can see

layer1 = Layer_Creation(X, 5)
layer1.forward()
layer2 = Layer_Creation(layer1.output, 3)
layer2.forward()

print(layer2.output)
