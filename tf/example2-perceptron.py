import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[2, 2])
y = tf.placeholder(tf.float32, shape=[2, 1])

W = tf.Variable( tf.random_normal([2,1]), name="W", dtype=tf.float32)
b = tf.Variable( tf.zeros([2,1]), name="bias", dtype=tf.float32)

z = tf.matmul(x,W) + b
out = tf.sigmoid(z)

loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits( logits=z, labels=y))

learning_rate = 0.02
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.InteractiveSession() #initializes a tensorflow session
sess.run(init)

for epoch in range(100001):
    if epoch % 10000 == 0:
        sess.run(optimizer, feed_dict={x:[[1.0,2.0],[1.0,2.0]], y:[[1.0], [2.0]] })
        print('loss: ',sess.run(loss, feed_dict={x:[[1.0,2.0],[1.0,2.0]], y:[[1.0], [2.0]] }))