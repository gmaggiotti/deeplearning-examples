import tensorflow as tf
import numpy as np

x_data = [
    [0, 5, 25, 30, 15, 2, 1],
    [0, 6, 15, 10,  3, 1, 1],
    [1, 0,  1,  4,  1, 3, 1],
    [0, 1,  0,  1,  1, 1, 1]
]
y_data = [[1], [1], [0], [0]]

x_data = 2*np.array(x_data)/30.0 -1

n_samples = 4
n_neurons = 7
x = tf.placeholder(tf.float32, shape=[n_samples, n_neurons])
y = tf.placeholder(tf.float32, shape=[n_samples, 1])

W = tf.Variable( tf.random_normal([n_neurons,1], seed=0), name="W", dtype=tf.float32)
b = tf.Variable( tf.zeros([n_samples,1]), name="bias", dtype=tf.float32)

logits = tf.sigmoid(tf.matmul(x, W) + b)
out = tf.reduce_mean( logits )

### calculate the error
loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits( logits=logits, labels=y))

LR = 0.01
### run the optimization
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=LR).minimize(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())     #init W & b
    writer = tf.summary.FileWriter("./tensorboard_log", sess.graph)

    for epoch in range(1000):
        ### run the optimizer
        opt, lo = sess.run([optimizer, loss], feed_dict={x: x_data, y: y_data})
        if epoch % 100 == 0:
            print lo

    print "y_prime: ",sess.run(logits, feed_dict={x: x_data,y: y_data })
    print('EOC')

