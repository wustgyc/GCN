import os
import psutil
from keras import regularizers,initializers,constraints,activations
from keras.layers import*
from keras.models import Model,load_model,Sequential
from keras.optimizers import Adam
import keras.backend as K
from keras.callbacks import LambdaCallback,Callback,TensorBoard
from sklearn.metrics import roc_auc_score,classification_report
import matplotlib.pyplot as plt
from keras.initializers import *
from sklearn.preprocessing import minmax_scale
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold,train_test_split
from keras.callbacks import LearningRateScheduler
###
import warnings
import math
warnings.filterwarnings("ignore")
IDE=1
TIME_STEP=14
NB_EPOCH=1000
BATCH_SIZE=1024
L_UNIT=18
LEARN_RATE=0.1
SMART_NUMBER=12
M=1
W_COEFFICIENT=0


class Metrics(Callback):
    def on_epoch_end(self, batch, logs={}):
        train_predict = (np.asarray(self.model.predict([X_train[0],X_train[1]], batch_size=1024))).round()  #+0.1避免误差
        train_targ = Y_train
        train_dict = classification_report(train_targ, train_predict)
        print(train_dict)
class EulerDist(Layer):
    """Basic graph convolution layer as in https://arxiv.org/abs/1609.02907"""
    def __init__(self,
                 activation=None,
                 kernel_initializer='glorot_uniform', #Gaussian distribution
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(EulerDist, self).__init__(**kwargs)
        self.result=None
        self.activation=activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)


    def build(self, input_shapes):
        input_dim=input_shapes[0][-1]  #双输入，input_shapes是一个元组
        self.kernel = self.add_weight(shape=(input_dim,1),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.built = True

    def call(self, inputs):
        tmp = K.square(inputs[0] - inputs[1])
        self.result = K.abs(tmp)
        self.result = K.dot(tmp, self.kernel)
        self.result = self.activation(self.result)
        return self.result

    # return output shape
    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)
def step_decay(epoch):
    init_lrate = 0.1
    drop = 0.5
    epochs_drop = 50
    lrate = init_lrate * math.pow(drop, math.floor(1 + epoch) / epochs_drop)
    return lrate
def loss_siamese(y_true,y_pred):
    p0 = 1. - y_true
    p1=y_true
    g0=K.square(y_pred)
    g1=K.square(K.maximum(0.,M-y_pred))
    return K.mean(p0*g0+p1*g1,axis=-1)
def create_pair(sample,label):
    sample_1 = []
    sample_2=[]
    label_pair=[]
    for i in range(len(label)):
        for j in range(i,len(label)):
            sample_1.append(sample[i])
            sample_2.append(sample[j])
            label_pair.append(int(label[i]!=label[j]))
    print("数据对创建成功,",len(sample_1),"对数据")
    return np.array([sample_1,sample_2]),np.array(label_pair)
def create_model(L_unit=L_UNIT):
    x = Model()
    x.add(LSTM(dropout=0.5,
                recurrent_dropout=0.5,
                units=L_unit,
                return_sequences=True,
                kernel_regularizer=regularizers.l2(W_COEFFICIENT),
                bias_regularizer=regularizers.l2(W_COEFFICIENT)))

    x.add(LSTM(units=L_unit,
                dropout=0.5,
                recurrent_dropout=0.5,
                return_sequences=True,
                kernel_regularizer=regularizers.l2(W_COEFFICIENT),
                bias_regularizer=regularizers.l2(W_COEFFICIENT)))

    x.add(LSTM(units=L_unit,
                dropout=0.5,
                recurrent_dropout=0.5,
                return_sequences=True,
                kernel_regularizer=regularizers.l2(W_COEFFICIENT),
                bias_regularizer=regularizers.l2(W_COEFFICIENT)))


    x.add(LSTM(units=L_unit,
                dropout=0.5,
                recurrent_dropout=0.5,
                return_sequences=False,
                kernel_regularizer=regularizers.l2(W_COEFFICIENT),
                bias_regularizer=regularizers.l2(W_COEFFICIENT)))


    x.add(Dense(units=128,
                activation='relu',
                kernel_regularizer=regularizers.l2(W_COEFFICIENT),
                bias_regularizer=regularizers.l2(W_COEFFICIENT)))

    x.add(Dense(units=256,
                kernel_regularizer=regularizers.l2(W_COEFFICIENT),
                use_bias=False))

    shared_model = x
    left_input = Input(shape=(14,12))
    right_input = Input(shape=(14,12))

    eulstm_distance = EulerDist(activation="sigmoid",kernel_regularizer=regularizers.l2(W_COEFFICIENT))([shared_model(left_input), shared_model(right_input)])

    model = Model(inputs=[left_input, right_input], outputs=[eulstm_distance])
    model.compile(loss=loss_siamese, optimizer=Adam(), metrics=['accuracy'])

    return model
def UnderSampler(sample,label,ratio=1,random_state=None):
    assert len(sample)==len(label)
    train_sample=[]
    train_label=[]
    threshold=np.sum(label) / (len(label) - np.sum(label)) * ratio
    random_num = np.random.RandomState(random_state)
    for i, row in enumerate(label):
        if row == 1:
            train_sample.append(sample[i])
            train_label.append(label[i])
        else:
            if random_num.rand() <= threshold:
                train_sample.append(sample[i])
                train_label.append(label[i])
    print("下采样完成!")
    return np.array(train_sample),np.array(train_label)
def print_memory_use():
    info = psutil.virtual_memory()
    print(u'内存使用：', int(psutil.Process(os.getpid()).memory_info().rss/1024/1024),"M")
if __name__ == '__main__':

    X=np.load("./temp_data/sample.npy")
    Y = np.load("./temp_data/label.npy")

    # -----------------------------------标准化
    X = minmax_scale(np.reshape(X, (-1, 12)))
    X = np.reshape(X, (-1, TIME_STEP, 12))

    print("数据读取成功")
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,shuffle=True,random_state=0,test_size=0.3)

    # -----------------------------------下采样，要不然内存读不下
    X_train,Y_train=UnderSampler(X_train,Y_train,ratio=1,random_state=1)

    # -----------------------------------获取数据对
    X_train,Y_train=create_pair(X_train,Y_train)
    X_train=X_train[:,:6000]
    Y_train=Y_train[:6000]
    # X_train=np.ones((2,6000,14,12))
    # Y_train=np.zeros(6000)
    # for i in range(6000):
    #     if i %2==0:
    #         X_train[0][i]=np.ones((14,12))
    #         X_train[1][i]=np.zeros((14, 12))
    # for i in range(6000):
    #     if i % 2 == 0:
    #         Y_train[i]=1
    # -----------------------------------训练
    model=create_model(18)
    lrate = LearningRateScheduler(step_decay)
    tbCallBack = TensorBoard(log_dir="./tensorboard/log2", histogram_freq=1,write_grads=True,batch_size=BATCH_SIZE) #tensorboard可视化
    metrics=Metrics()
    model.fit([X_train[0],X_train[1]], Y_train,
              verbose=1,
              batch_size=BATCH_SIZE,
              shuffle=True,
              epochs=NB_EPOCH,
              callbacks=[lrate,tbCallBack],
              validation_split=0.3
              )
