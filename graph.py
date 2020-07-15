from __future__ import print_function

from keras import activations, initializers, constraints
from keras import regularizers
from keras.engine import Layer
import keras.backend as K
from Parameter import BATCH_SIZE
import tensorflow as tf
class GraphConvolution(Layer):
    """Basic graph convolution layer as in https://arxiv.org/abs/1609.02907"""
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform', #Gaussian distribution
                 bias_initializer='zeros',
                 kernel_regularizer=None, 
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = True

    #配置这个才能保证load_model后自定义层参数与先前传入的一致
    def get_config(self):
        config = {'units': self.units,'activation':self.activation}
        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shapes):#input_shapes包含了batch_size，与Input里的input_shape不同
        features_shape = input_shapes[0]
        output_shape = (features_shape[0],features_shape[1] ,self.units)
        return output_shape  # (batch_size,row,column)

    def build(self, input_shapes):
        features_shape = input_shapes[0]
        assert len(features_shape) == 3
        input_dim = features_shape[2]

        self.kernel = self.add_weight(shape=(input_dim,
                                             self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, mask=None):
        features = inputs[0]
        graph = inputs[1]

        #features[i]和graph[i]为定值
        features=features[0]
        #A*v
        output=K.dot(graph,features)
        #A*V*W
        output = K.dot(output, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        return self.activation(output)

