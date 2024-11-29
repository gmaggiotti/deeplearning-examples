import numpy as np


# # input data, each column represent a dif neuron
# X = np.loadtxt("1NN/X.txt", delimiter=",")
#
# # output, are the one-hot encoded labels
# y = np.loadtxt("1NN/Y.txt", delimiter=",").reshape(X.__len__(), 1)

X = np.arange(50)
delta = np.random.uniform(-2,2, size=(50,))
y = .6 * X + delta

np.random.seed(1)  # The seed for the random generator is set so that it will return the same random numbers each time,
# which is sometimes useful for debugging.

# Now we intialize the weights to random values. w0 is the weight between the input layer and the hidden layer.

# synapses
w0 = np.random.random()

# This is the main training loop. The output shows the evolution of the error between the model and desired. The
# error steadily decreases.
pre_error= 100
error = 0
for j in range(400):

    # Calculate forward through the network.
    l1 = X.dot(w0)

    # Error back propagation of errors using the chain rule.
    l1_error = y.T - l1
    error = np.mean(np.abs(l1_error))
    print("epoch {}: {}".format(j, error))
    if(error > pre_error):
        break

    # update weights using gradient descent
    adjustment = X.T.dot(l1_error) / X.shape[0]
    w0 += 0.001 * adjustment

    pre_error = error

    # update weights (no learning rate term)
    w0 += adjustment.sum()/adjustment.size * 0.001


import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np


plt.plot(X, y, 'o', color='black');

x1 = np.linspace(0, 50, 50)
y1 = x1*w0
plt.plot(x1, y1, '-', color='green');
plt.show()

