import tensorflow as tf
import numpy as np

n_samples = 4
n_neurons = 7
x = tf.placeholder(tf.float32, shape=[n_samples, n_neurons])
y = tf.placeholder(tf.float32, shape=[n_samples, 1])

W = tf.Variable( tf.random_normal([n_neurons,1], seed=1), name="W", dtype=tf.float32)
b = tf.Variable( tf.zeros([n_samples,1]), name="bias", dtype=tf.float32)

logits = tf.sigmoid(tf.matmul(x, W) + b)
out = tf.reduce_mean( logits )

### calculate the error
loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits( logits=logits, labels=y))

LR = 0.001
### run the optimization
optimizer = tf.train.GradientDescentOptimizer(learning_rate=LR).minimize(loss)
#optimizer = tf.train.AdamOptimizer().minimize(loss)

init = tf.global_variables_initializer()
sess = tf.InteractiveSession() #initializes a tensorflow session
sess.run(init)

x_data = [
    [0, 5, 25, 30, 15, 2, 1],
    [0, 6, 15, 10, 3, 1, 1],
    [1, 0, 1, 4, 1, 3, 1],
    [0, 1, 0, 1, 1, 1, 1]
]

y_data = [[1], [1], [0], [0]]

for epoch in range(10000):
    if epoch % 1000 == 0:
        ### run the optimizer
        sess.run(optimizer, feed_dict={x: x_data, y: y_data})
        lo = sess.run([ logits], feed_dict={x: x_data, y: y_data})
        print np.mean(np.abs(y_data - np.array(lo)))
        #print 'loss: ', l, ' error:', 1 - o , 'e:',np.mean(np.abs(y_data-lo))


print('EOC')