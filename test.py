import numpy as np
from sklearn.impute import SimpleImputer

import tensorflow as tf
import keras.backend as K
a=[1,23,4,5]
a=tf.convert_to_tensor(a,dtype=float)
print(a.shape[0])
a=(K.sum(a,axis=-1)-0.5)/2
a=K.reshape(a,(-1,1))
a=K.mean(a,axis=-1)
with tf.Session() as sess:
    print(sess.run(a))