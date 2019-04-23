from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import numpy as np

celsius_q    = np.array([-4, -3, -2,  0, 2, 3, 4],  dtype=float)
fahrenheit_a = np.array([16, 9,  4, 0, 4, 9, 16],  dtype=float)

for i,c in enumerate(celsius_q):
    print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))

tf.set_random_seed(7)
# Dense layer is a fully-connected layer
l0 = tf.keras.layers.Dense(units=4, input_shape=[1], activation='relu')
l1 = tf.keras.layers.Dense(units=4, activation='relu')
l2 = tf.keras.layers.Dense(units=1, activation='relu')
model = tf.keras.Sequential([l0, l1, l2])

optimizer = tf.keras.optimizers.Adam(0.1)
model.compile(loss='mean_squared_error', optimizer=optimizer)

history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=True)

print("Finished training the model")
print("predict 100 to {}".format(model.predict([2.3])))
x = [x for x in range(-10,11)]
y_square = [y**2 for y in range(-10,11)]
pred_y_sq = [model.predict([i])[0][0] for i in range(-10,11)]


import matplotlib.pyplot as plt
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(x,y_square)
plt.plot(x,pred_y_sq)
plt.show()
