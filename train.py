from __future__ import print_function
from keras import *
from keras.layers import*
from keras.models import Model,load_model
from keras.optimizers import Adam
from keras.callbacks import LambdaCallback, Callback, EarlyStopping, TensorBoard
from sklearn.metrics import roc_auc_score,classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from graph import GraphConvolution
from utils import *
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from Parameter import *
import os
import warnings
warnings.filterwarnings("ignore")
os.environ["PATH"] += os.pathsep + 'D:/Graphviz2.38/bin'
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
        train_predict = compute_threshold(np.asarray(self.model.predict(train_data,batch_size=BATCH_SIZE)))
        train_targ = train_label
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
        val_predict = compute_threshold(np.asarray(self.model.predict(test_data,batch_size=1024*64)))
        val_targ = test_label
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
def make_model(learn_rate=0.001,m=12,pool_s=2,L_unit=18):
    #Input
    I_we = Input(shape=(SMART_NUM, DIVIDE))
    G = Input(shape=(SMART_NUM, SMART_NUM))
    I_sa = Input(shape=(TIME_STEP, SMART_NUM))

    #GCN
    I_we_norm = BatchNormalization()(I_we)
    Gcn = GraphConvolution(DIVIDE, activation='relu')([I_we_norm, G])
    Gcn = GraphConvolution(DIVIDE, activation='relu')([Gcn, G])

    #LSTM
    LSTM=CuDNNLSTM(units=L_unit,
                  return_sequences=True,
                  kernel_regularizer=regularizers.l2(),
                  bias_regularizer=regularizers.l2(),
                  recurrent_regularizer=regularizers.l2())(I_sa)

    LSTM = CuDNNLSTM(units=L_unit,
                     return_sequences=False,
                     kernel_regularizer=regularizers.l2(),
                     bias_regularizer=regularizers.l2(),
                     recurrent_regularizer=regularizers.l2())(LSTM)

    #MFB
    M_1 = Dense(m)(Gcn)
    M_2 = Dense(m)(LSTM)
    M_2 = RepeatVector(SMART_NUM)(M_2)
    Mfb = Multiply()([M_2, M_1])
    Mfb = Reshape(target_shape=(SMART_NUM * m,1))(Mfb)
    Mfb = AveragePooling1D(pool_size=pool_s, strides=pool_s, padding='valid')(Mfb)
    Mfb = Reshape(target_shape=(int(SMART_NUM * m / pool_s),))(Mfb)

    Output = Dense(1,activation='sigmoid')(Mfb)
    # Compile model
    model = Model(inputs=[G,I_we, I_sa], outputs=Output)

    #freeze layer


    model.compile(loss='mean_squared_error', optimizer=Adam(lr=learn_rate), weighted_metrics=['accuracy'])
    return model
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
def compute_threshold(x):
    for i,row in enumerate(x):
        if x[i]==1:
            x[i]=1
            continue
        if x[i] / (1 - x[i]) * IDE >=1:
            x[i]=1
        else:
            x[i]=0
    return x

#获取数据
X, A, sample, label= load_data()

#标准化处理
A = preprocess_adj(A, SYM_NORM)
sample=scale(np.reshape(sample,(-1,SMART_NUM)))
sample=np.reshape(sample,(-1,TIME_STEP,SMART_NUM))




#分离测试集
X_train,X_test,Y_train,Y_test=train_test_split(sample,label,shuffle=True,random_state=0,test_size=0.2)

#下采样
X_train,Y_train=UnderSampler(X_train,Y_train,IDE,random_state=1) #先把测试机丢一边，后面再改

#准备输入
train_data=[[A]*len(X_train),[X]*len(X_train),X_train]
train_label=Y_train

test_data=[[A]*len(X_test),[X]*len(X_test),X_test]
test_label=Y_test

# Train
metrics=Metrics()
tbCallBack = TensorBoard(log_dir="./tensorboard/log", histogram_freq=1,write_grads=True,batch_size=BATCH_SIZE) #tensorboard可视化
model=make_model()


clf=make_model(learn_rate=LEARN_RATE,pool_s=POOL_SIZE,m=M)
#early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=2)
clf.fit(train_data, train_label,
        epochs=NB_EPOCH,
        verbose=1,
        batch_size=BATCH_SIZE,
        callbacks=[metrics,tbCallBack],
        validation_split=0.2,
        shuffle=True)

recall_0=[]
recall_1=[]
pre_0=[]
pre_1=[]
auc=[]
for j in range(-20, 0):
    recall_0.append(metrics.val_recalls_0[j])

for j in range(-20, 0):
    recall_1.append(metrics.val_recalls[j])

for j in range(-20, 0):
    pre_0.append(metrics.val_precisions_0[j])

for j in range(-20, 0):
    pre_1.append(metrics.val_precisions[j])

for j in range(-20, 0):
    auc.append(metrics.val_auc[j])

print("mean:",np.mean(pre_0),"std",np.std(pre_0))
print("mean:",np.mean(pre_1),"std",np.std(pre_1))
print("mean:", np.mean(recall_0), "std", np.std(recall_0))
print("mean:", np.mean(recall_1), "std", np.std(recall_1))
print("mean:", np.mean(auc), "std", np.std(auc))
print()


#plot_model(model, to_file='model.png')

