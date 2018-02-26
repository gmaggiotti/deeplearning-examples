import tensorflow as tf
import numpy as np

x_data = [1, 2, 3, 4]
y_data = [0, -1, -2, -3]


x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([1,1], seed=0), name="W", dtype=tf.float32)
b = tf.Variable(tf.zeros([1,1]), name="bias", dtype=tf.float32)

y_prime = W * x + b
loss = tf.reduce_sum( tf.square(y_prime - y) )

LR = 0.01
### run the optimization
optimizer = tf.train.GradientDescentOptimizer(learning_rate=LR).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())     #init W & b
    writer = tf.summary.FileWriter("./tensorboard_log", sess.graph)

    for epoch in range(1000):
        ### run the optimizer
        opt, lo = sess.run([optimizer, loss], feed_dict={x: x_data, y: y_data})
        if epoch % 100 == 0:
            print lo

    print sess.run(loss, feed_dict={x: x_data,y: y_data })
    print sess.run([W,b])
    print('EOC')

