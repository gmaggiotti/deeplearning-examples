import tensorflow as tf

sess = tf.reset_default_graph() #Clears the default graph stack and resets the global default graph.
sess = tf.InteractiveSession() #initializes a tensorflow session

a = tf.placeholder(dtype=tf.float32)
b = tf.placeholder(dtype=tf.float32)
c = tf.add(a,b)

w1 = tf.Variable( tf.random_normal((2,2)), name="w1", dtype=tf.float32)
b1 = tf.Variable( tf.zeros((2,2)), name="bias", dtype=tf.float32)

res =sess.run(tf.global_variables_initializer())
print(res)

res = sess.run(c, feed_dict={a:2.0, b:3.0})
print(res)