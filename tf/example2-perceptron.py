import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[3, 2])
y = tf.placeholder(tf.float32, shape=[3, 1])

W = tf.Variable( tf.random_normal([2,1]), name="W", dtype=tf.float32)
b = tf.Variable( tf.zeros([3,1]), name="bias", dtype=tf.float32)

logits = tf.matmul(x,W) + b
out = tf.reduce_mean( tf.sigmoid(logits) )

loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits( logits=logits, labels=y))
learning_rate = 0.02
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)


init = tf.global_variables_initializer()
sess = tf.InteractiveSession() #initializes a tensorflow session
sess.run(init)

x_data = [[1.0,2.0],[1.0,2.0],[3.0,2.0]]
y_data = [[1.0], [2.0], [5.0]]
for epoch in range(100001):
    if epoch % 10000 == 0:
        sess.run(optimizer, feed_dict={x:x_data, y:y_data })
        l, o = sess.run([loss, out], feed_dict={x:x_data, y:y_data })
        print 'loss: ', l, ' error:', 1 - o