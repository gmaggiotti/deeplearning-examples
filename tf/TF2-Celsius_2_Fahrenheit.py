from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import numpy as np

celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

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
print("predict 100 to {}".format(model.predict([100.0])))

import matplotlib.pyplot as plt
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])
plt.show()
