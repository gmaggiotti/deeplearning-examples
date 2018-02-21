import tensorflow as tf
n_samples = 3
n_neurons = 3
x = tf.placeholder(tf.float32, shape=[n_samples, n_neurons])
y = tf.placeholder(tf.float32, shape=[n_samples, 1])

W = tf.Variable( tf.random_normal([n_neurons,1]), name="W", dtype=tf.float32)
b = tf.Variable( tf.zeros([n_neurons,1]), name="bias", dtype=tf.float32)

logits = tf.sigmoid(tf.matmul(x, W) + b)
out = tf.reduce_mean( logits )

### calculate the error
loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits( logits=logits, labels=y))
LR = 0.02
### run the optimization
optimizer = tf.train.GradientDescentOptimizer(learning_rate=LR).minimize(loss)


init = tf.global_variables_initializer()
sess = tf.InteractiveSession() #initializes a tensorflow session
sess.run(init)

x_data = [
    [7.0, 2.0, 3.0],
    [0.0, 0.0, 0.0],
    [9.0, 6.0, 5.0],
    [1.0, 1,0, 0.0]
]

y_data = [[1.0], [0], [1.0], [0.0]]

for epoch in range(1000001):
    if epoch % 50000 == 0:
        ### run the optimizer
        sess.run(optimizer, feed_dict={x: x_data, y: y_data})
        l, o = sess.run([loss, out], feed_dict={x: x_data, y: y_data})
        print 'loss: ', l, ' error:', 1 - o


print('EOC')