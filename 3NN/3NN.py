import numpy as np

def sigmoid(x, deriv=False):
    if(deriv == True):
        return x*(1-x)
    return 1/(1 + np.exp(-x))

#input data, each column represent a dif neuron
X = 2*np.loadtxt("3NN/X.txt",delimiter=",")/30 - 1
#output, are the one-hot encoded labels
y = np.loadtxt("3NN/Y.txt",delimiter=",").reshape(X.__len__(),1)

np.random.seed(1)

#synapses
w0 = 2*np.random.random((X.size/X.__len__(),X.__len__())) - 1
w1 = 2*np.random.random((X.__len__(),1)) - 1

for j in xrange(60000):

    # Calculate forward through the network.
    l0 = X
    l1 = sigmoid(np.dot(l0, w0))
    l2 = sigmoid(np.dot(l1, w1))

    # Error back propagation of errors using the chain rule.
    l2_error = y - l2
    if(j % 10000) == 0:
        print("Error: " + str(np.mean(np.abs(l2_error))))

    l2_adjustment = l2_error*sigmoid(l2, deriv=True) #(y-a).d/dw(-a), a = sigmoid(Sum Xi*Wi)
    l1_error = l2_adjustment.dot(w1.T)
    l1_adjustment = l1_error * sigmoid(l1,deriv=True) #(y-a).d/dw(-a), a = sigmoid(Sum Xi*Wi)

    #update weights (no learning rate term)
    w1 += l1.T.dot(l2_adjustment)
    w0 += l0.T.dot(l1_adjustment)

print("Output after training")
print(l2)

def predict(X1):
    l0 = np.zeros((X.__len__(),X.size/X.__len__()))
    max = np.matrix(X1).max()
    l0[0] = 2*np.asanyarray(X1, dtype=np.float32)/max - 1
    l1 = sigmoid(np.dot(l0, w0))
    l2 = sigmoid(np.dot(l1, w1))
    return l2[0][0] #since process X1[0] output would be l2[0]

test_dataset=[1,9,19,33,16,2,1]
result = predict(test_dataset)
print("expected output 1, predicted output " + repr(result))
assert (result > 0.95), "Test Failed. Exepected result > 0.95"

test_dataset=[1,0, 1, 4, 1,3,1]
result = predict(test_dataset)
print("expected output 0, predicted output " + repr(result))
assert (result < 0.95), "Test Failed. Exepected result < 0.95"

