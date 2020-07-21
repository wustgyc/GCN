import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.layers import*
from keras.models import Model,load_model,Sequential
from keras import regularizers,initializers,constraints,activations
from keras.optimizers import Adam
from keras.callbacks import LambdaCallback,Callback,TensorBoard
from sklearn.metrics import roc_auc_score,classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from keras.initializers import *
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
import warnings
import parameter
warnings.filterwarnings("ignore")

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
        train_predict = compute_threshold(np.asarray(self.model.predict(sample[train_range],batch_size=parameter.BATCH_SIZE)))
        train_targ = label[train_range]
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
        val_predict = compute_threshold(np.asarray(self.model.predict(sample[val_range],batch_size=parameter.BATCH_SIZE)))
        val_targ = label[val_range]
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
        if len(self.train_recalls)%50==0:
             plot(self)
        return

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
def compute_threshold(x):
    for i,row in enumerate(x):
        if x[i]==1:
            x[i]=1
            continue
        if x[i] / (1 - x[i]) * parameter.IDE >=1:
            x[i]=1
        else:
            x[i]=0
    return x
def create_model():
    model=Sequential()
    kernel_initializer = glorot_uniform(seed=0)
    recurrent_initializer = orthogonal(seed=0)
    model.add(CuDNNLSTM(units=parameter.L_UNIT,
                        input_shape=(sample.shape[1],sample.shape[2]),
                        return_sequences=True,
                        kernel_initializer=kernel_initializer,
                        recurrent_initializer=recurrent_initializer,
                        kernel_regularizer=regularizers.l2(),
                        bias_regularizer=regularizers.l2(),
                        recurrent_regularizer=regularizers.l2()))


    kernel_initializer = glorot_uniform(seed=3)
    recurrent_initializer = orthogonal(seed=3)
    model.add(CuDNNLSTM(units=parameter.L_UNIT,
                        return_sequences=False,
                        kernel_initializer=kernel_initializer,
                        recurrent_initializer=recurrent_initializer,
                        kernel_regularizer=regularizers.l2(),
                        bias_regularizer=regularizers.l2(),
                        recurrent_regularizer=regularizers.l2()))

    kernel_initializer = glorot_uniform(seed=4)
    model.add(Dense(units=1,activation="sigmoid",kernel_initializer=kernel_initializer))
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=parameter.LEARN_RATE),weighted_metrics=['accuracy'])
    return model



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
sample = np.vstack((X, X_3))
label = np.hstack((Y, Y_3))

sample = minmax_scale(np.reshape(sample, (-1, parameter.SMART_NUMBER)))
sample = np.reshape(sample, (-1, parameter.TIME_STEP, parameter.SMART_NUMBER))


kfold = StratifiedKFold(n_splits=4, shuffle=True,random_state=4)
LR=[-1]
BS=[-1]
for lr in LR:                     #网格搜索
    for bs in BS:
        recall_0=[]
        recall_1=[]
        pre_0=[]
        pre_1=[]
        auc=[]
        for i,(train_range,val_range) in enumerate(kfold.split(sample,label)):         #交叉验证
            if i==1:
                break;
            model=create_model()
            model.summary()
            metrics=Metrics()
            tbCallBack = TensorBoard(log_dir="./tensorboard/log", histogram_freq=1,batch_size=parameter.BATCH_SIZE) #tensorboard可视化
            model.fit(sample[train_range], label[train_range],
                    verbose=1,
                    batch_size=parameter.BATCH_SIZE,
                    shuffle=True,
                    epochs=parameter.NB_EPOCH,
                    validation_data=(sample[val_range],label[val_range]),
                    callbacks=[metrics,tbCallBack]
                    )
            dirpath="./result_figure/kfold_{}_".format(i)
            plot(metrics,dirpath)

            for j in range(-20,0):
                recall_0.append(metrics.val_recalls_0[j])

            for j in range(-20,0):
                recall_1.append(metrics.val_recalls[j])

            for j in range(-20,0):
                pre_0.append(metrics.val_precisions_0[j])

            for j in range(-20,0):
                pre_1.append(metrics.val_precisions[j])

            for j in range(-20,0):
                auc.append(metrics.val_auc[j])
        print("lr-bs:",lr,bs)
        print("mean:",np.mean(pre_0),"std",np.std(pre_0))
        print("mean:",np.mean(pre_1),"std",np.std(pre_1))
        print("mean:", np.mean(recall_0), "std", np.std(recall_0))
        print("mean:", np.mean(recall_1), "std", np.std(recall_1))
        print("mean:", np.mean(auc), "std", np.std(auc))
        print()




