# Now, we've got our loss working
# But right now, our network just tells us how wrong the algorithm is, it doesn't actually do anything with that info
# To make it learn, we now introduce Backpropagation
# This is the process of working backwards from the loss, calculating the gradients (slopes) of each weight, and nudging them to make better predictions next time
# Thus we will be implementing backward() methods for each of the classes

import math
import numpy as np

X = np.array([[1, 2, 4, 5],
     [1, 4, 2, 1],
     [3, 31, 2, 1]])
y = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.inputs = inputs # we save inputs for use in the backward pass
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues) # dL/dW = X^T * dL/dZ
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True) # dL/db = sum across batch
        # Gradient on values to pass to previous layer
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy() # We copy so we don't modify original gradients
        self.dinputs[self.inputs <= 0] = 0 # ReLU passes gradient only for positive inputs

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=1)
        self.output = probabilities
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues) # Prepare gradient array as the same shape as dvalues
        # For each sample, calculate the jacobian matrix of the softmax activation function
        for i, (single_output, single_dvalue) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            # Jacobian matrix for softmax: diag(s) - s * s^T, where s is the softmax output vector
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Chain rule application, multiply jacobian by the gradient from the next layer
            self.dinputs[i] = np.dot(jacobian_matrix, single_dvalue)
      
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
      
class Loss_Categorical_Cross_entropy(Loss):
    def forward(self, y_pred, y_true): 
        y_true = np.array(y_true)
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        else:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        negative_log_likelihood = -np.log(correct_confidences)
        return negative_log_likelihood 
    def backward(self, dvalues, y_true):
        samples = len(dvalues) # No. of samples in batch
        labels = len(dvalues[0]) # No. of classes

        # If targets are class indices, convert to OHE for gradient calculation
 
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient of softmax outputs
        # Formula derived from dL/dz = -y_true / y_pred
        self.dinputs = -y_true / dvalues
        # Normalise gradient by the number of samples to balance batch size impact
        self.dinputs = self.dinputs / samples


layer1 = Layer_Dense(4, 3)
layer2 = Layer_Dense(3, 4)

activation1 = Activation_ReLU()
activation2 = Activation_Softmax()

loss_function = Loss_Categorical_Cross_entropy()

# Now for the fun part, we can actually make it train
# We'll go ahead and create a simple loop to do multiple forward and backward passes

learning_rate = 0.1
epochs = 1000

for epoch in range(epochs): # going through 1000 epochs
    # Forward pass
    layer1.forward(X)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)
    
    loss = loss_function.calculate(activation2.output, y)
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == np.argmax(y, axis=1))

    if epoch % 100 == 0: # tracking progress by printing stats every 100 epochs
        print(f"epoch {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}")
    
    # Backward pass
    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.dinputs)
    layer2.backward(activation2.dinputs)
    activation1.backward(layer2.dinputs)
    layer1.backward(activation1.dinputs)

    # Adjusting weights/biases, basic gradient descent
    layer1.weights -= learning_rate * layer1.dweights
    layer1.biases -= learning_rate * layer1.dbiases
    layer2.weights -= learning_rate * layer2.dweights
    layer2.biases -= learning_rate * layer2.dbiases
print(np.argmax(activation2.output, axis=1))

# You should expect to see the loss decreasing over the epochs resulting in more accurate predictions
# We can improve a lot from here (better optimisers, batching, more layers, etc)
# But for now, this is a full working example of simple, working machine learning.
