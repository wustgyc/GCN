from __future__ import print_function
import scipy.sparse as sp
import numpy as np
from sklearn.utils import shuffle
from Parameter import*
def serialize_and_split(time_step,random=True):
    sample=np.load("./temp_data/sample.npy",allow_pickle=True)
    label=np.load("./temp_data/label.npy",allow_pickle=True)

    print("加载数据成功!")

    sample_flatten =[]
    label_flatten=[]
    assert len(sample)==len(label)
    for disk_i,disk in enumerate(sample):
        assert len(sample[disk_i]) == len(label[disk_i])

        res=len(sample[disk_i])%time_step     #时间窗口无法填满的时候，舍弃一些数据

        sample[disk_i]=sample[disk_i][res:]   #多余的数据应该删前面的，因为如果删后面的可能会把所有fail样本删掉(如果time_step太大的话)
        sample[disk_i]=np.reshape(sample[disk_i],(-1,time_step,12))

        label[disk_i] = label[disk_i][res:]
        label[disk_i] = np.reshape(label[disk_i],(-1,time_step))

        for row in sample[disk_i]:
            sample_flatten.append(row)
        for row in label[disk_i]:
            label_flatten.append(row)

    sample=np.array(sample_flatten)
    label=np.array(label_flatten)

    step_label=np.zeros((len(label)),dtype=int)
    #为每个分组做标签
    for i,row in enumerate(label):
        if np.sum(row)>=7:               #之前已对标签做过改进,fail前6天均视为fail
            step_label[i]=1
        else:
            step_label[i]=0

    if random==True:
        sample, step_label = shuffle(sample, step_label, random_state=0)  #同时打样本，让fail均匀分布，有利于test-train划分

    # train_range=range(0,int(train_size_rate*len(sample)))
    # test_range=range(int(train_size_rate*len(sample)),len(sample))
    print("序列化完毕!")

    return sample,step_label
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
    return np.array(train_sample),np.array(train_label,dtype=int)
def load_data():
    features=np.load("./temp_data/smart_static.npy",allow_pickle=True)
    adj=np.load("./temp_data/correlation_matrix.npy",allow_pickle=True)
    # 序列化
    sample = np.load("./temp_data/sample.npy")
    label = np.load("./temp_data/label.npy")
    return features, adj, sample,label
def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d)
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj)
    return a_norm
def preprocess_adj(adj, symmetric=True):
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj.tocsr().todense()
def split_train_and_validation(data,label,ratio):
    return range(0,int(ratio*len(label))),range(int(ratio*len(label)),len(label))

