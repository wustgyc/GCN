#代码2020-7-11版本:实现了siamese LSTM,后续还需要针对论文进行改进
#目标:复现ATC2020论文:HDDse
#环境设置为:VSCode 2019
#          tensorflow 2.1.0-gpu
#          keras 2.4.2  
from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from sklearn.metrics import classification_report
from keras.callbacks import LambdaCallback,Callback,TensorBoard
from keras.models import Model
from keras.layers import Input, Dense, Dropout, LSTM, Layer, Bidirectional
from keras.optimizers import Adam
from keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import StratifiedKFold,train_test_split
TIME_STEP=14
NB_EPOCH=1000
BATCH_SIZE=256
SMART_NUMBER=12
M=1
W_COEFFICIENT=1e-3

class EluDist(Layer):  # 封装成keras层的欧式距离计算

    # 初始化EluDist层，此时不需要任何参数输入
    def __init__(self, **kwargs):
        self.result = None
        super(EluDist, self).__init__(**kwargs)

    # 建立EluDist层
    def build(self, input_shape):
        self.kernel = self.add_weight(name = 'kernel',
                                      shape = (256,1),
                                      initializer = 'uniform',
                                      trainable = True)
        super(EluDist, self).build(input_shape)

    # 计算欧式距离
    def call(self, vects, **kwargs):
        ###是否需要for循环
        x , y = vects
        self.result = K.abs(x - y)
        self.result = K.dot(self.result , self.kernel)#β乘上欧式距离
        self.result = K.sigmoid(self.result)
        return self.result

    # 返回结果
    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)

#计算损失函数
def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

#创建训练网络
def create_base_network(input_shape):
    #Base network to be shared (eq. to feature extraction).
    #四个隐藏层,即四层BiLSTM和一个全连接层
    #根据论文中给出的,Dropout使用的比例为0.5
    input = Input(shape=input_shape)
    ##在这一步加入了L2正则化项,但是目前不知道这种做法是否正确
    x = Bidirectional(LSTM(units = 128,dropout = 0.5,return_sequences=True, kernel_regularizer = tf.keras.regularizers.l2(1e-3)),input_shape=input_shape)(input)
    x = Bidirectional(LSTM(units = 128,dropout = 0.5,return_sequences=True))(x)
    x = Bidirectional(LSTM(units = 128,dropout = 0.5,return_sequences=True))(x)
    x = Bidirectional(LSTM(units = 128,dropout = 0.5,return_sequences=False))(x)
    x = Dense(128)(x)
    x = Dropout(0.5)(x)
    x = Dense(256)(x)
    return Model(input, x)

#学习率衰减
def step_decay(epoch):
    if epoch % 50 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.5)
        print("lr changed to {}".format(lr * 0.5))
    return K.get_value(model.optimizer.lr)

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

def create_pair(sample,label):
    sample_1 = []
    sample_2=[]
    label_pair=[]
    for i in range(len(label)):
        for j in range(i,len(label)):
            sample_1.append(sample[i])
            sample_2.append(sample[j])
            label_pair.append(int(label[i]==label[j]))
    print("数据对创建成功,",len(sample_1),"对数据")
    return np.array([sample_1,sample_2]),np.array(label_pair)

if __name__ == '__main__':
    X = np.load("./temp_data/sample.npy")
    Y = np.load("./temp_data/label.npy")
    X.astype('float32')
    Y.astype('int')
    # -----------------------------------标准化
    X = minmax_scale(np.reshape(X, (-1, 12)))
    X = np.reshape(X, (-1, 14, 12))

    print("数据读取成功")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=True, random_state=0, test_size=0.3)

    # -----------------------------------下采样，要不然内存读不下
    X_train, Y_train = UnderSampler(X_train, Y_train, ratio=1, random_state=1)

    # -----------------------------------获取数据对
    X_train, Y_train = create_pair(X_train, Y_train)
    X_train = X_train[:, :6000]
    Y_train = Y_train[:6000]



    n_epochs = 100
    input_shape=(14,12)
    base_network = create_base_network(input_shape)

    input_a = Input(shape = input_shape)
    input_b = Input(shape = input_shape)

    #processed_a和processed_a是孪生LSTM网络处理之后的结果
    processed_a = base_network(input_a)#shape为(None.256)
    processed_b = base_network(input_b)

    distance = EluDist()([processed_a, processed_b])

    learn_rate = LearningRateScheduler(step_decay)

    model = Model([input_a, input_b], distance)
    adam = Adam()##定义优化器

    model.compile(loss = contrastive_loss, optimizer = adam)##激活model

    tbCallBack = TensorBoard(log_dir="./tensorboard/log2", histogram_freq=1, write_grads=True,
                             batch_size=BATCH_SIZE)  # tensorboard可视化

    history=model.fit([X_train[0], X_train[1]], Y_train,
              verbose=1,
              batch_size=BATCH_SIZE,
              shuffle=True,
              epochs=NB_EPOCH,
              callbacks=[learn_rate, tbCallBack],
              validation_split=0.3
              )

    # 绘制训练 & 验证的损失值
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend('Train', loc='upper left')
    plt.show()

    save_path = "./HDDse_Siamese_LSTM/数据集/"
    model.save(save_path + "model_train.h5")
