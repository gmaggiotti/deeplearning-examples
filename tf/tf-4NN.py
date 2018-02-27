import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os

def read_dataset():
    path = os.path.dirname(os.path.abspath(__file__))
    X = np.loadtxt(path + "/train_dataset.csv1", delimiter=",")
    max = np.matrix(X).max()
    X = 2 * X / max - 1
    Y = np.loadtxt(path + "/label_dataset.csv1", delimiter=",").reshape(X.__len__(), 1)
    return shuffle(X, Y, random_state=0)

X,Y = read_dataset()
LR = 0.03
epochs = 1000
neurons = X.shape[1]
samples = X.shape[0]


x = tf.placeholder(tf.float32, shape=[samples, neurons])
y = tf.placeholder(tf.float32, shape=[samples, 1])

W0 = tf.Variable(tf.truncated_normal([neurons, samples], seed=0), name="W0", dtype=tf.float32)
b0 = tf.Variable(tf.truncated_normal([samples, 1]), name="bias0", dtype=tf.float32)
W1 = tf.Variable(tf.truncated_normal([samples, samples], seed=0), name="W1", dtype=tf.float32)
b1 = tf.Variable(tf.truncated_normal([samples, 1]), name="bias1", dtype=tf.float32)
W2 = tf.Variable(tf.truncated_normal([samples, 1], seed=0), name="W2", dtype=tf.float32)
b2 = tf.Variable(tf.truncated_normal([samples, 1]), name="bias2", dtype=tf.float32)

l0 = tf.sigmoid(tf.add(tf.matmul(x, W0), b0))
l1 = tf.sigmoid(tf.add(tf.matmul(l0, W1), b1))
l2 = tf.sigmoid(tf.matmul(l1, W2) + b2)

### calculate the error
loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits( logits=l2, labels=y))


### run the optimization
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=LR).minimize(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())     #init W & b
    for epoch in range(epochs):
        ### run the optimizer
        l1_, opt, lo = sess.run([l2,optimizer, loss], feed_dict={x: X, y: Y})
        if epoch % 100 == 0:
            print "error: " , np.mean(np.abs( Y - l1_ ))

    print "y_prime: ",np.round(sess.run(l2, feed_dict={x: X,y: Y }))

    print('EOC')

