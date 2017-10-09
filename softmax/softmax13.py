import numpy as np
import matplotlib.pyplot as plt

scores = [3.0, 1.0, 0.2]


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


print(softmax(scores))

# Plot softmax curves
#x = np.arange(-2.0, 6.0, 0.1)

plt.plot(scores, softmax(scores).T, linewidth=2)
result = map(lambda xi: xi * 10, scores)
plt.plot(scores, softmax(result).T, linewidth=2)
plt.show()
print "normal"
print softmax(scores)
print "x10"
print softmax(result)

#this means that if we increas the size of the output(scores), the classifier becomes very confident about
# the predictions

