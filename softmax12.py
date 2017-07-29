import numpy as np
import matplotlib.pyplot as plt

scores = [3.0, 1.0, 0.2]


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


print(softmax(scores))

# Plot softmax curves
x = np.arange(-6.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()

print softmax(scores)

#this means that if we increase the size of the output(scores), the classifier becomes very confident about
# the predictions

