from sklearn import svm
from get_correlation_matrix import get_sample
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report,recall_score,roc_auc_score
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
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
    print(len(train_sample))
    return np.array(train_sample),np.array(train_label)

data=get_sample(-1)
print("读取成功")
sample=[]
label=[]
for row in data:
    sample.append(row[4:])
    label.append(row[3])
del data
sample,label=shuffle(sample,label,random_state=1)
ra=RandomUnderSampler()
sample, label = ra.fit_sample(sample, label)

sample=np.array(sample)
label=np.array(label)


scaler = StandardScaler()
sample= scaler.fit_transform(sample)

x_train, x_test, y_train, y_test = train_test_split(sample, label, test_size=0.2)
# rbf核函数，设置数据权重

# svc = svm.SVC(kernel='rbf')
# kfold = StratifiedKFold(n_splits=4, shuffle=True,random_state=4)
# #c_range = np.logspace(10, 15, 6, base=2)
# c_range=[2048,3072,4096,6144,7168,8192]
# gamma_range = np.logspace(-5, 2, 8, base=2)

# fdr=0
# far=-1
# gg=gamma_range[0]
# cc=c_range[0]
# for c in c_range:
#     for g in gamma_range:
#         _val_recall=[]
#         _val_precision = []
#         _val_precision_0 = []
#         _val_recall_0 = []
#         for i,(train_range,val_range) in enumerate(kfold.split(x_train,y_train)):
#             svc = svm.SVC(kernel='rbf',C=c,gamma=g)
#             svc.fit(x_train[train_range],y_train[train_range])
#
#             pred=svc.predict(x_train[val_range])
#             targ=y_train[val_range]
#             val_dict = classification_report(targ, pred, output_dict=True)
#
#             _val_recall.append(val_dict['1']['recall'])
#             _val_precision.append(val_dict['1']['precision'])
#             _val_precision_0.append(val_dict['0']['precision'])
#             _val_recall_0.append(val_dict['0']['recall'])
#
#         print(c, g, " : recall_0:", np.mean(_val_recall_0),np.std(_val_recall_0), " recall_1", np.mean(_val_recall),np.std(_val_recall))
#         if np.mean(_val_recall_0) > 0.9 and np.mean(_val_recall) > fdr:
#             fdr = np.mean(_val_recall)
#             far = 1 - np.mean(_val_recall_0)
#             gg = g
#             cc = c
#
# print(gg,cc,fdr,far)







svc = svm.SVC(kernel='rbf')
c_range = np.logspace(10, 16, 7, base=2)
gamma_range = np.logspace(-11, -4, 8, base=2)
# 网格搜索交叉验证的参数范围，cv=3,3折交叉
param_grid = [{'C': c_range, 'gamma': gamma_range}]
grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1,verbose=1,scoring='roc_auc',refit=True)
# 训练模型
grid.fit(x_train, y_train)

for p, s in zip(grid.cv_results_['params'],
	grid.cv_results_['mean_test_score']):
	print(p, s)
print("Best parameters set found on test set:")
# 输出最优的模型参数
print(grid.best_params_,grid.best_score_)
# 在测试集上测试最优的模型的泛化能力.
y_true, y_pred = y_test, grid.predict(x_test)
print(classification_report(y_true, y_pred))
print(roc_auc_score(y_true, y_pred))

