import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os

def read_dataset():
    path = os.path.dirname(os.path.abspath(__file__))
    X = np.loadtxt(path + "/train_dataset.csv", delimiter=",")
    max = np.matrix(X).max()
    X = 2 * X / max - 1
    Y = np.loadtxt(path + "/label_dataset.csv", delimiter=",").reshape(X.__len__(), 1)
    return shuffle(X, Y, random_state=0)

X,Y = read_dataset()
LR = 0.3
epochs = 1000
neurons = X.shape[1]
samples = X.shape[0]


x = tf.placeholder(tf.float32, shape=[samples, neurons])
y = tf.placeholder(tf.float32, shape=[samples, 1])

W0 = tf.Variable(tf.random_normal([neurons, samples], seed=0), name="W", dtype=tf.float32)
b0 = tf.Variable(tf.zeros([samples, 1]), name="bias", dtype=tf.float32)
W1 = tf.Variable(tf.random_normal([samples, 1], seed=0), name="W", dtype=tf.float32)
b1 = tf.Variable(tf.zeros([samples, 1]), name="bias", dtype=tf.float32)

l0 = tf.sigmoid(tf.matmul(x, W0) + b0 )
l1 = tf.sigmoid(tf.matmul(l0, W1) + b1 )

### calculate the error
loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits( logits=l1, labels=y))


### run the optimization
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=LR).minimize(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())     #init W & b
    for epoch in range(epochs):
        ### run the optimizer
        opt, lo = sess.run([optimizer, loss], feed_dict={x: X, y: Y})
        if epoch % 100 == 0:
            print lo

    print "y_prime: ",np.round(sess.run(l1, feed_dict={x: X,y: Y }))

    print('EOC')

