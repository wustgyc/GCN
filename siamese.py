import os
import psutil
import tensorflow as tf
from keras import regularizers,initializers,constraints,activations
from keras.layers import*
from keras.models import Model,load_model,Sequential
from keras.optimizers import Adam
import keras.backend as K
from keras.callbacks import LambdaCallback,Callback,TensorBoard
from sklearn.metrics import roc_auc_score,classification_report
from keras.initializers import *
from sklearn.preprocessing import minmax_scale
import numpy as np
from sklearn.model_selection import StratifiedKFold,train_test_split
from keras.callbacks import LearningRateScheduler
import parameter
import matplotlib.pyplot as plt
import warnings
import math
#warnings.filterwarnings("ignore")
from pandas import Series, DataFrame
def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall_1(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(y_true * y_pred))
    possible_positives = K.sum(K.round(y_true))
    recall_1 = true_positives / (possible_positives + K.epsilon())
    return recall_1

def recall_0(y_true, y_pred):

    true_neg=K.sum(K.round((1-y_true) * (1-y_pred)))
    possible_neg=K.sum(K.round(1-y_true))
    recall_0=true_neg/(possible_neg+K.epsilon())
    return recall_0

def threshold_judge(y_pred):
    for i,row in enumerate(y_pred):
        if row >parameter.PRED_THRSHOLD:
            y_pred[i]=1
        else:
            y_pred[i] =0
    return y_pred

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_recalls = []
        self.val_precisions = []
        self.val_auc=[]
        self.val_loss=[]
        self.val_recalls_0=[]
        self.val_precisions_0=[]

        self.train_recalls = []
        self.train_precisions = []
        self.train_auc = []
        self.train_loss=[]
        self.train_recalls_0=[]
        self.train_precisions_0=[]

    def on_epoch_end(self, epoch, logs={}):
        #train
        train_predict =threshold_judge(np.asarray(self.model.predict([X_train[0],X_train[1]],batch_size=parameter.BATCH_SIZE)))
        train_targ = Y_train
        train_dict = classification_report(train_targ, train_predict, output_dict=True)

        _train_recalls = train_dict['1']['recall']
        _train_precisions = train_dict['1']['precision']
        _train_precision_0 = train_dict['0']['precision']
        _train_recall_0 = train_dict['0']['recall']

        _train_auc = roc_auc_score(train_targ, train_predict)
        self.train_recalls.append(_train_recalls)
        self.train_precisions.append(_train_precisions)
        self.train_auc.append(_train_auc)
        self.train_loss.append(logs.get("loss"))
        self.train_precisions_0.append((_train_precision_0))
        self.train_recalls_0.append(_train_recall_0)

        #val
        val_predict = threshold_judge(np.asarray(self.model.predict([self.validation_data[0],self.validation_data[1]],batch_size=parameter.BATCH_SIZE)))
        val_targ = self.validation_data[2]
        val_dict = classification_report(val_targ, val_predict, output_dict=True)

        _val_recall = val_dict['1']['recall']
        _val_precision = val_dict['1']['precision']
        _val_precision_0=val_dict['0']['precision']
        _val_recall_0=val_dict['0']['recall']
        _val_auc = roc_auc_score(val_targ, val_predict)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        self.val_auc.append(_val_auc)
        self.val_loss.append(logs.get("val_loss"))
        self.val_precisions_0.append(_val_precision_0)
        self.val_recalls_0.append((_val_recall_0))

        #print(len(self.train_recalls))
        #print("train_precisions_0", _train_precision_0,"  train_recalls_0", _train_recall_0)
        #print("train_precisions_1", _train_precisions,"  train_recalls_1", _train_recalls)
        # print("train_auc", _train_auc)
        # print("val_precisions_0", _val_precision_0, "  val_recalls_0", _val_recall_0)
        # print("val_precisions_1", _val_precision,"  val_recalls_1", _val_recall)
        # print("val_auc", _val_auc)
        #每20轮更新一次图像
        if len(self.train_recalls)%5==0:
             plot(self)
        return

class Middle_Output(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        sequence_output_1 = Model(inputs=model.layers[2].layers[0].input, outputs=model.layers[2].layers[5].output)
        ans_1 = sequence_output_1.predict(X_train[0])
        ans_2 = sequence_output_1.predict(X_train[1])
        one_distribute=[]
        zero_distribute = []
        output1=[]
        for i in range(len(X_train[0])):
            output1.append(np.sum(np.abs(ans_1[i])))
            if Y_train[i]==1:
                one_distribute.append(np.sum(np.abs(ans_1[i]-ans_2[i])))
            else:
                zero_distribute.append(np.sum(np.abs(ans_1[i]-ans_2[i])))

        one_distribute=Series(one_distribute,name="one")
        zero_distribute = Series(zero_distribute,name="zeor")
        output1=Series(output1)
        one_distribute.plot(kind='kde')
        zero_distribute.plot(kind='kde')

        plt.savefig(str(epoch))
        plt.close("all")
        output1.plot(kind='kde')
        plt.savefig(str(epoch)+"out")
        plt.close("all")
        # plt.hist(x=one_distribute,bins=20,density=True,color="red",alpha=0.5)
        # plt.hist(x=zero_distribute, bins=20, density=True, color="blue", alpha=0.5)
        # plt.show()
        # plt.close("all")

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
    g1=K.square(K.maximum(0.,parameter.M-y_pred))
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

def create_model():
    shared_model = Sequential()
    shared_model.add(Bidirectional(LSTM(
                dropout=0.5,
                units=parameter.L_UNIT,
                return_sequences=True,
                kernel_regularizer=regularizers.l2(parameter.W_COEFFICIENT),
                bias_regularizer=regularizers.l2(parameter.W_COEFFICIENT))))

    shared_model.add(Bidirectional(LSTM(units=parameter.L_UNIT,
                dropout=0.5,
                return_sequences=True,
                kernel_regularizer=regularizers.l2(parameter.W_COEFFICIENT),
                bias_regularizer=regularizers.l2(parameter.W_COEFFICIENT))))

    shared_model.add(Bidirectional(LSTM(units=parameter.L_UNIT,
                dropout=0.5,
                return_sequences=True,
                kernel_regularizer=regularizers.l2(parameter.W_COEFFICIENT),
                bias_regularizer=regularizers.l2(parameter.W_COEFFICIENT))))


    shared_model.add(Bidirectional(LSTM(units=parameter.L_UNIT,
                dropout=0.5,
                return_sequences=False,
                kernel_regularizer=regularizers.l2(parameter.W_COEFFICIENT),
                bias_regularizer=regularizers.l2(parameter.W_COEFFICIENT))))


    shared_model.add(Dense(units=32,
                activation='relu',
                kernel_regularizer=regularizers.l2(parameter.W_COEFFICIENT),
                bias_regularizer=regularizers.l2(parameter.W_COEFFICIENT)))

    shared_model.add(Dense(units=64,
                kernel_regularizer=regularizers.l2(parameter.W_COEFFICIENT),
                bias_regularizer=regularizers.l2(parameter.W_COEFFICIENT)))

    left_input = Input(shape=(parameter.TIME_STEP,parameter.SMART_NUMBER))
    right_input = Input(shape=(parameter.TIME_STEP,parameter.SMART_NUMBER))

    eulstm_distance = EulerDist(activation="sigmoid",kernel_regularizer=regularizers.l2(parameter.W_COEFFICIENT))([shared_model(left_input), shared_model(right_input)])

    model = Model(inputs=[left_input, right_input], outputs=[eulstm_distance])
    model.compile(loss=loss_siamese, optimizer=Adam(), metrics=[recall_0,recall_1])

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

def plot(metrics,dirpath="./result_figure/"):
    plt.plot(metrics.val_loss, 'b-')
    plt.plot(metrics.train_loss, 'r-')
    plt.title('loss report')
    plt.ylabel('evaluation')
    plt.xlabel('epoch')
    plt.legend(["val", "train"], loc='lower right')
    plt.savefig(dirpath+'loss.png')
    plt.close("all")

    plt.plot(metrics.val_precisions_0, 'b-')
    plt.plot(metrics.train_precisions_0, 'r-')
    plt.title('precision_0 report')
    plt.ylabel('evaluation')
    plt.xlabel('epoch')
    plt.legend(["val", "train"], loc='lower right')
    plt.savefig(dirpath+'presision_0.png')
    plt.close("all")

    plt.plot(metrics.val_recalls_0, 'b-')
    plt.plot(metrics.train_recalls_0, 'r-')
    plt.title('recall_0 report')
    plt.ylabel('evaluation')
    plt.xlabel('epoch')
    plt.legend(["val", "train"], loc='lower right')
    plt.savefig(dirpath+'recall_0.png')
    plt.close("all")

    plt.plot(metrics.val_precisions, 'b-')
    plt.plot(metrics.train_precisions, 'r-')
    plt.title('precision_1 report')
    plt.ylabel('evaluation')
    plt.xlabel('epoch')
    plt.legend(["val", "train"], loc='lower right')
    plt.savefig(dirpath+'presision_1.png')
    plt.close("all")

    plt.plot(metrics.val_recalls, 'b-')
    plt.plot(metrics.train_recalls, 'r-')
    plt.title('recall_1 report')
    plt.ylabel('evaluation')
    plt.xlabel('epoch')
    plt.legend(["val", "train"], loc='lower right')
    plt.savefig(dirpath+'recall_1.png')
    plt.close("all")

    plt.plot(metrics.val_auc, 'b-')
    plt.plot(metrics.train_auc, 'r-')
    plt.title('auc report')
    plt.ylabel('evaluation')
    plt.xlabel('epoch')
    plt.legend(["val", "train"], loc='lower right')
    plt.savefig(dirpath+'auc.png')
    plt.close("all")

if __name__ == '__main__':
    #读取融合数据
    X=np.load("./temp_data/sample_ST12000NM0007.npy")
    Y = np.load("./temp_data/label_ST12000NM0007.npy")
    X_1,Y_1=UnderSampler(X,Y,ratio=1,random_state=1)
    X = np.load("./temp_data/sample_ST4000DM000.npy")
    Y = np.load("./temp_data/label_ST4000DM000.npy")
    X_2, Y_2 = UnderSampler(X, Y, ratio=1, random_state=1)
    X = np.load("./temp_data/sample_ST10000NM0086.npy")
    Y = np.load("./temp_data/label_ST10000NM0086.npy")
    X_3, Y_3 = UnderSampler(X, Y, ratio=1, random_state=1)
    X=np.vstack((X_1,X_2))
    Y=np.hstack((Y_1,Y_2))
    X = np.vstack((X, X_3))
    Y = np.hstack((Y, Y_3))

    # -----------------------------------标准化
    X = minmax_scale(np.reshape(X, (-1, parameter.SMART_NUMBER)))
    X = np.reshape(X, (-1, parameter.TIME_STEP, parameter.SMART_NUMBER))

    # -----------------------------------分离测试集
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,shuffle=True,random_state=0,test_size=0.3)

    # -----------------------------------获取数据对
    X_train,Y_train=create_pair(X_train,Y_train)
    X_train=X_train[:,:20000]
    Y_train=Y_train[:20000]

    # X_train=np.zeros((2,6000,14,9))
    # Y_train=np.zeros(6000)
    # for i in range(6000):
    #     if i%2==0:
    #         X_train[0][i]=np.zeros((14,9))
    #         X_train[1][i] = np.zeros((14, 9))
    #         Y_train[i]=0
    #     else:
    #         X_train[0][i] = np.zeros((14, 9))
    #         X_train[1][i] = np.ones((14, 9))
    #         Y_train[i] = 1

    # -----------------------------------训练
    model=create_model()
    middle_output=Middle_Output()
    lrate = LearningRateScheduler(step_decay)
    tbCallBack = TensorBoard(log_dir="./tensorboard/log2", histogram_freq=1,batch_size=parameter.BATCH_SIZE,update_freq="epoch") #tensorboard可视化
    metrics=Metrics()
    model.fit([X_train[0],X_train[1]], Y_train,
              verbose=1,
              batch_size=parameter.BATCH_SIZE,
              shuffle=True,
              epochs=parameter.NB_EPOCH,
              callbacks=[lrate,tbCallBack],
              validation_split=0.3
              )


