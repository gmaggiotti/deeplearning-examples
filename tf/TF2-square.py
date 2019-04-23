from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import numpy as np

x_dataset    = np.array([-10, -4, -3, -2, 0, 2, 3, 4, 5, 10], dtype=float)
y_dataset = np.array([100, 16, 9, 4, 0, 4, 9, 16, 25, 100], dtype=float)

for i,c in enumerate(x_dataset):
    print("{} cuadratic function".format(c, y_dataset[i]))

tf.set_random_seed(7)
# Dense layer is a fully-connected layer
l0 = tf.keras.layers.Dense(units=4, input_shape=[1], activation='relu')
l1 = tf.keras.layers.Dense(units=4, activation='relu')
l2 = tf.keras.layers.Dense(units=1, activation='relu')
model = tf.keras.Sequential([l0, l1, l2])

optimizer = tf.keras.optimizers.Adam(0.1)
model.compile(loss='mean_squared_error', optimizer=optimizer)

history = model.fit(x_dataset, y_dataset, epochs=500, verbose=True)

print("Finished training the model")
print("predict 2.3^2 to {}".format(model.predict([2.3])))
x = [x for x in range(-10,11)]
y_square = [y**2 for y in range(-10,11)]
pred_y_sq = [model.predict([i])[0][0] for i in range(-10,11)]


import matplotlib.pyplot as plt
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(x,y_square)
plt.plot(x,pred_y_sq)
plt.show()
