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
    return shuffle(X, Y, random_state=1)

X,Y = read_dataset()
train_x, test_x, train_y, test_y = train_test_split(X,Y,test_size=0.1, random_state=1)

LR = 0.03
epochs = 1000
neurons = train_x.shape[1]
samples = train_x.shape[0]

x = tf.placeholder(tf.float32, shape=[None, neurons])
y = tf.placeholder(tf.float32, shape=[None, 1])
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
optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss)

def predict(X1):
    X1.resize((samples, neurons), refcheck=False)
    result = sess.run(l2, feed_dict={x: X1,y: test_y })
    return result[:test_y.shape[0]]


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())     #init W & b
    for epoch in range(epochs):
        ### run the optimizer
        l2_, opt, lo = sess.run([l2,optimizer, loss], feed_dict={x: train_x, y: train_y})
        if epoch % (epochs*.1) == 0:
            error = np.mean(np.abs( train_y - l2_ ))
            accurency = 1 - np.sum(np.abs( (predict(test_x) - test_y))/samples)
            print "error: " , error , " accurency of testset:", accurency

print('EOC')

