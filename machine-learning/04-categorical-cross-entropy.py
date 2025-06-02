# We've done our softmax activation function, so now we are left with probability distributions regarding how confident our network believes it's predictions to be
# From here, the next step is to figure out how wrong the network's predictions are relative to the desired outputs
# We will achieve this through Categorical Cross Entropy
# Now, the formula for CCE is as follows: ∑ y1​ x -log(y2), this formula may look confusing at first but it is actually really simple and simplifies even further once you introduce one hot encoding
# y1 is the target value, y2 is the predicted value
# The formula is as simple as this- the sum of each target value multiplied by the negative logarithm of it's corresponding prediction
# But here's the thing, in our case, what actually IS the target values? 
# As said previously, we are using One-Hot-Encoded vectors, where our targets may look something like this:
# Eg: for a sample in a batch like [1,2,3], you may have an OHE target vector such as [1,0,0]
# So these OHE target vectors essentially consist of a 1 in the column of its corresponding index position within the targetted row, and a 0 in every other column as demonstrated above
# Now, why does this matter? Well
# As seen in the formula for CCE, the loss consists of the sum of each target value multiplied by the -log of its corresponding prediction
# So, in this case, when we apply the formula this is what happens
# In the sample above: (1 x -log(1)) + (0 x -log(2)) + (0 x -log(3))
# Without even calculating anything yet, we begin to notice something awfully convenient, we know that anything timesed by 0 becomes 0
# So any value that isn't the target or corresponds to the target value can simply be disregarded
# Thus simplifies into: (1 x -log(1)) + (0) + (0) which in other words is just (1 x -log(1))
# JUST TO NOTE, we wouldn't actually have -log(1), the value would be clipped to something more like 0.9 because log(1) is 0
# So, by introducing One-Hot-Encoding, the formula for categorical cross entropy simplifies from ∑ y1​ x -log(y2) to the simply the negative logarithm of the predicted y value corresponding to the target value
# So ∑ y1​ x -log(y2) -> -log(y2), this is very easy for us to code in languages such as python.
# Now once we know the loss, back-propogation occurs, but let's implement this into what we have so far

import math
import numpy as np
import nnfs

X = [[1, 2, 4, 5],
     [1, 4, 2, 1],
     [3, 4, 2, 1]]
y = [[1, 0, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]]
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=1)
        self.output = probabilities 
      
class Loss: # Creating a Loss superclass
    def calculate(self, output, y): # Defining a function, to store the 
        sample_losses = self.forward(output, y) # Calls the subclass' forward method
        data_loss = np.mean(sample_losses) # Calculates the mean loss of the sample losses
        return data_loss # Returns it
      
class Loss_Categorical_Cross_entropy(Loss): # Creating a class for the loss function
    def forward(self, y_pred, y_true): # forward function, allowing the predicted y value to be passed and the corresponding target value alongside it
        y_true = np.array(y_true) # Converting the python array of y into a numpy array so that we can check it's shape in the next couple lines to check whether it's OHE or not
        samples = len(y_pred) # The amount of lists (samples) in the predicted batch output
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7) # This prevents issues with the log function
        # The issue with the log function, is that if the predicted value is 0, then the log function will output inf (infinite) which is clearly erroneous
        # So we clip the the values so that they can be in the range of 0-1 (uninclusive of 0 or 1) which is what we've done above
        if len(y_true.shape) == 1: # We check the dimension of the target values, if it's one dimensional, i.e, a single list, then it's not one hot encoded.
            # scalar values
            correct_confidences = y_pred_clipped[range(samples), y_true] # Using the amount of samples and the y_true values, we can easily index the correct_confidences according to the values of what's in y_true
        else: # or elif len(y_true.shape) == 2: Either works, but the same sort of principle applies, if it's not one dimensional (i.e, it's 2 dimensional), then it's one hot encoded
            # one hot encoded
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1) # We don't actually need to index anything, simply multiplying each predicted row by each target row filters out quite nicely to the correct confidences, then by adding up each row, we create a list that would be the corret confidences
        negative_log_likelihood = -np.log(correct_confidences)
        # Now, we apply the actual CCE logic, as explained above, we only need to work out the negative logarithm or the predicted y values, which is what we're doing here
        return negative_log_likelihood 


layer1 = Layer_Dense(4, 3)
layer2 = Layer_Dense(3, 4)

activation1 = Activation_ReLU()
activation2 = Activation_Softmax()

layer1.forward(X)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation2.forward(layer2.output)

loss_function = Loss_Categorical_Cross_entropy() # Creating a class for the loss function
Loss = loss_function.calculate(activation2.output, y) # Retrieving the loss values by passing the softmax probability distribution (activation2) and the target values into our loss function
