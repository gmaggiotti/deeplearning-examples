# -*- coding: utf-8 -*-

import numpy as np #matrix math 
import tensorflow as tf #machine learningt
import helpers #for formatting data into batches and generating random sequence data

tf.reset_default_graph()
imported_meta = tf.train.import_meta_graph("./model_final/trained_variables.ckpt.meta")
with tf.Session() as sess:
    imported_meta.restore(sess, tf.train.latest_checkpoint('./model_final/'))
    h_est2 = sess.run('decoder_prediction')