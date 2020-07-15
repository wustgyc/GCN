
import numpy as np
import tensorflow as tf
import keras.backend as K
a=[[1],[1],[2]]
b=[1,2,3]
a=tf.convert_to_tensor(a,dtype=float)
b=tf.convert_to_tensor(b,dtype=float)

a=a+b

with tf.Session() as sess:
    print(sess.run(a))


