import numpy as np

def sigmoid(x, deriv=False):
    if(deriv == True):
        return x*(1-x)
    return 1/(1 + np.exp(-x))

#input data, each column represent a dif neuron
X = np.loadtxt("3NN/X.txt",delimiter=",")
#output, are the one-hot encoded labels
y = np.loadtxt("3NN/Y.txt",delimiter=",").reshape(X.__len__(),1)

np.random.seed(1) # The seed for the random generator is set so that it will return the same random numbers each time, which is sometimes useful for debugging.

# Now we intialize the weights to random values. syn0 are the weights between the input layer and the hidden layer.  It is a 3x4 matrix because there are two input weights plus a bias term (=3) and four nodes in the hidden layer (=4). syn1 are the weights between the hidden layer and the output layer. It is a 4x1 matrix because there are 4 nodes in the hidden layer and one output. Note that there is no bias term feeding the output layer in this example. The weights are initially generated randomly because optimization tends not to work well when all the weights start at the same value. Note that neither of the neural networks shown in the video describe the example.

#synapses
syn0 = 2*np.random.random((X.size/X.__len__(),X.__len__())) - 1  # 3x4 matrix of weights ((2 inputs + 1 bias) x 4 nodes in the hidden layer)
syn1 = 2*np.random.random((X.__len__(),1)) - 1  # 4x1 matrix of weights. (4 nodes x 1 output) - no bias term in the hidden layer.

# This is the main training loop. The output shows the evolution of the error between the model and desired. The error steadily decreases.
for j in xrange(60000):

    # Calculate forward through the network.
    l0 = X
    l1 = sigmoid(np.dot(l0, syn0))
    l2 = sigmoid(np.dot(l1, syn1))

    # Error back propagation of errors using the chain rule.
    l2_error = y - l2
    if(j % 10000) == 0:   # Only print the error every 10000 steps, to save time and limit the amount of output.
        print("Error: " + str(np.mean(np.abs(l2_error))))

    l2_adjustment = l2_error*sigmoid(l2, deriv=True)
    l1_error = l2_adjustment.dot(syn1.T)
    l1_adjustment = l1_error * sigmoid(l1,deriv=True)

    #update weights (no learning rate term)
    syn1 += l1.T.dot(l2_adjustment)
    syn0 += l0.T.dot(l1_adjustment)

print("Output after training")
print(l2)

def predict(X1):
    l0 = np.zeros((4, 7))
    l0[0] = X1
    l1 = sigmoid(np.dot(l0, syn0))
    l2 = sigmoid(np.dot(l1, syn1))
    return l2[0] #since process X1[0] output would be l2[0]

test_dataset=[1,9,19,33,16,2,1]

result = predict(test_dataset)
print("Output of example should be:" + repr(result))


