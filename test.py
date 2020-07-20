import numpy as np
from sklearn.impute import SimpleImputer

import tensorflow as tf
import keras.backend as K
a=[[1.,2,3],[1,2,3],[1,2,3]]
# a=tf.convert_to_tensor(a,dtype=float)
a=(K.sum(a,axis=-1)-0.5)/2
a=K.reshape(a,(-1,1))
a=K.mean(a,axis=-1)
with tf.Session() as sess:
    print(sess.run(a))