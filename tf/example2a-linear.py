import tensorflow as tf
import numpy as np

x_data = [
    [0, 5, 25, 30, 15, 2, 1],
    [0, 6, 15, 10,  3, 1, 1],
    [1, 0,  1,  4,  1, 3, 1],
    [0, 1,  0,  1,  1, 1, 1]
]
y_data = [[1], [1], [0], [0]]
x_data = 2 * np.array(x_data) / 30.0 - 1
y_data = np.float32(y_data)

n_samples = 4
n_neurons = 7
x = tf.placeholder(tf.float32, shape=[n_samples, n_neurons])
y = tf.placeholder(tf.float32, shape=[n_samples, 1])

W = tf.Variable( tf.random_normal([n_neurons,1], seed=0), name="W", dtype=tf.float32)
b = tf.Variable( tf.zeros([n_samples,1]), name="bias", dtype=tf.float32)


y_prime = tf.matmul(x, W) + b
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

    print "y_prime: ",sess.run(y_prime, feed_dict={x: x_data,y: y_data })

    def predict(X1):
        l0 = np.zeros((n_samples,n_neurons))
        max = np.matrix(X1).max()
        l0[0] = 2*np.asanyarray(X1, dtype=np.float32)/max - 1
        result = sess.run(y_prime, feed_dict={x: l0,y: y_data })
        return result[0]

    test_dataset=[1,9,19,33,16,2,1]
    result = predict(test_dataset)
    print("expected output 1, predicted output " + repr(result))
    assert (result > 0.95), "Test Failed. Expected result > 0.95"

    test_dataset=[1,0, 1, 0, 1,3,1]
    result = predict(test_dataset)
    print("expected output 0, predicted output " + repr(result))
    assert (result < 0.95), "Test Failed. Expected result < 0.95"
    print('EOC')

