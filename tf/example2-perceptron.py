import tensorflow as tf
import numpy as np

x_data = [
    [0, 5, 25, 30, 15, 2, 1],
    [0, 6, 15, 10,  3, 1, 1],
    [1, 0,  1,  4,  1, 3, 1],
    [0, 1,  0,  1,  1, 1, 1]
]
y_data = [[1], [1], [0], [0]]

n_samples = 4
n_neurons = 7
x = tf.placeholder(tf.float32, shape=[n_samples, n_neurons])
y = tf.placeholder(tf.float32, shape=[n_samples, 1])

W = tf.Variable( tf.random_normal([n_neurons,1], seed=0), name="W", dtype=tf.float32)
b = tf.Variable( tf.zeros([n_samples,1]), name="bias", dtype=tf.float32)


y_prime = tf.matmul(x, W) + b
loss = tf.reduce_min( tf.square(y - y_prime))

LR = 0.01
### run the optimization
optimizer = tf.train.GradientDescentOptimizer(learning_rate=LR).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())     #init W & b
    writer = tf.summary.FileWriter("./tensorboard_log", sess.graph)

    for epoch in range(360000):
        if epoch % 40000 == 0:
            ### run the optimizer
            opt, lo = sess.run([optimizer, loss], feed_dict={x: x_data, y: y_data})
            print lo


    print('EOC')

