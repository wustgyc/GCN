# 这是用来生成全部训练数据的代码，数据集的产生分为两步
# 训练集和测试集采用的方法将不同
# 1. 产生健康磁盘的14天连续数据


import tensorflow as tf
import numpy as np
import pandas as ps
import random
import os


# 该方法希望将disk数据按照serial_number打乱，然后构成一个乱序的数据集
def shuffleDataSet(inputFile):
    # 读取csv
    data = ps.read_csv(inputFile)

    # 根据serial_number获取所有的disk序列号
    shuffleIndex = data['serial_number'].unique()
    # print(shuffleIndex)

    # 打乱序列号
    random.shuffle(shuffleIndex)

    dataSet = []

    # 循环将隶属同一个serial_number放到同一个list中
    for i in range(len(shuffleIndex)):
        new_data = data.loc[data['serial_number'] == shuffleIndex[i]]
        dataSet.append(new_data)

    return dataSet


# 好倒霉，发现数据集中并非每个磁盘都有91天的数据,因此，需要剔除那些不足的数据
def removeBadData(dataSet):
    newDataSet = []
    for i in range(len(dataSet)):
        if (len(dataSet[i]) == 91):
            newDataSet.append(dataSet[i])
    return newDataSet


def findBadData(dataSet):
    badDataSet = []

    for i in range(len(dataSet)):
        if (len(dataSet[i]) != 91):
            badDataSet.append(dataSet[i])
    return badDataSet


# 输入排序好的数据，输出训练集
def loadTrainSet(dataSet):
    trainNum = len(dataSet)

    for i in range(trainNum):
        dataSet[i]['Date'] = ps.to_datetime(dataSet[i]['date'])
        dataSet[i].index = dataSet[i]['Date']
        dataSet[i].sort_index(inplace=True)
        trainSet = dataSet[i][0:60]
        x_train = trainSet.loc[:, ['smart_1_normalized', 'smart_4_raw',
                                   'smart_5_raw', 'smart_12_raw',
                                   'smart_187_normalized', 'smart_193_normalized',
                                   'smart_196_raw', 'smart_197_raw',
                                   'smart_198_raw']]
        x_train.to_csv('x_train.csv', mode='a', index=False, header=False)

        y_train = trainSet.loc[:, ['failure']]
        y_train.to_csv('y_train.csv', mode='a', index=False, header=False)


# 输入排序好的数据，输出测试集
def loadTestSet(dataSet):
    trainNum = len(dataSet)

    for i in range(trainNum):
        dataSet[i]['Date'] = ps.to_datetime(dataSet[i]['date'])
        dataSet[i].index = dataSet[i]['Date']
        dataSet[i].sort_index(inplace=True)
        trainSet = dataSet[i][60:91]
        x_train = trainSet.loc[:, ['smart_1_normalized', 'smart_4_raw',
                                   'smart_5_raw', 'smart_12_raw',
                                   'smart_187_normalized', 'smart_193_normalized',
                                   'smart_196_raw', 'smart_197_raw',
                                   'smart_198_raw']]
        x_train.to_csv('x_test.csv', mode='a', index=False, header=False)

        y_train = trainSet.loc[:, ['failure']]
        y_train.to_csv('y_test.csv', mode='a', index=False, header=False)


# 输入错误样本的训练集
def loadBadTrainSet(dataSet):
    trainNum = len(dataSet)

    for i in range(trainNum):

        dataSet[i]['Date'] = ps.to_datetime(dataSet[i]['date'])
        dataSet[i].index = dataSet[i]['Date']
        dataSet[i].sort_index(inplace=True)
        sampleNum = len(dataSet[i])

        # 由于故障磁盘没有那么多天数据,所以需要判断
        if (sampleNum < 60):
            # 前面数据不动
            trainSet = dataSet[i][sampleNum - 20:sampleNum]

            x_train = trainSet.loc[:, ['smart_1_normalized', 'smart_4_raw',
                                       'smart_5_raw', 'smart_12_raw',
                                       'smart_187_normalized', 'smart_193_normalized',
                                       'smart_196_raw', 'smart_197_raw',
                                       'smart_198_raw']]
            x_train.to_csv('x_bad_train.csv', mode='a', index=False, header=False)


# 输入错误样本的测试集
def loadBadTestSet(dataSet):
    trainNum = len(dataSet)

    for i in range(trainNum):

        dataSet[i]['Date'] = ps.to_datetime(dataSet[i]['date'])
        dataSet[i].index = dataSet[i]['Date']
        dataSet[i].sort_index(inplace=True)
        sampleNum = len(dataSet[i])

        # 由于故障磁盘没有那么多天数据,所以需要判断
        if (sampleNum < 91 and sampleNum > 60):
            trainSet = dataSet[i][sampleNum - 20:sampleNum]
            x_train = trainSet.loc[:, ['smart_1_normalized', 'smart_4_raw',
                                       'smart_5_raw', 'smart_12_raw',
                                       'smart_187_normalized', 'smart_193_normalized',
                                       'smart_196_raw', 'smart_197_raw',
                                       'smart_198_raw']]
            x_train.to_csv('x_bad_test.csv', mode='a', index=False, header=False)


# 产生磁盘14天的集合数据
def geneDisksSet(x_dataSet, y_dataSet):
    sampleNum = (int)(len(x_dataSet) / (60))

    disks = []
    label = []

    for i in range(sampleNum):
        j = 14
        while (j <= 60):
            index = i * 60 + j

            disks.append(x_dataSet.loc[index - 14:index - 1])
            label.append(y_dataSet.loc[index - 14:index - 1].sum())
            j += 1

    return disks, label


# 产生故障磁盘14天的集合数据
def geneFailureDisksSet(x_dataSet):
    sampleNum = (int)(len(x_dataSet) / (20))

    disks = []

    for i in range(sampleNum):
        j = 14
        while (j <= 20):
            index = i * 20 + j

            disks.append(x_dataSet.loc[index - 14:index - 1])
            j += 1

    return disks


# 总方法:产生健康/故障磁盘训练数据
def geneTrainData(fileName):
    # 第一步打乱数据，并要求其按照序列号组织好
    dataSet = shuffleDataSet(fileName)

    # 获得该数据集中不足60天的故障磁盘数据，产生故障磁盘数据集
    badDataSet = findBadData(dataSet)

    # 从总数据集中剔除错误的数据集,产生健康磁盘数据集
    goodDataSet = removeBadData(dataSet)

    # 针对健康磁盘数据集，分离出属性和标签，属性保存在文件x_train.csv,标签保存在y_train.csv
    # 每次运行代码时需要提前删除这两个文件,因为采用的是文件可增写的方式
    loadTrainSet(goodDataSet)

    # 接着从文件中读写出属性和标签
    x_trainSet = ps.read_csv('x_train.csv', header=None)
    y_trainSet = ps.read_csv('y_train.csv', header=None)

    # 然后产生连续14天的磁盘数据
    goodDisk, goodDiskLabel = geneDisksSet(x_trainSet, y_trainSet)

    # 写入健康磁盘文件health_disks.csv
    for i in range(len(goodDisk)):
        goodDisk[i].to_csv('health_disks.csv', mode='a', index=False, header=False)

    # ----------------健康磁盘数据处理完毕------------------#

    # 针对故障磁盘数据，同样也需要先分离属性和标签，和上面一样，但是步骤多了一步
    # 需要将不满60天的数据补充完整，所以采用的方法不同，并且存储的文件名也不同
    # 文件名分别为：x_bad_train.csv, y_bad_train.csv,
    loadBadTrainSet(badDataSet)

    # 接着从文件中读写出属性和标签
    x_trainSet = ps.read_csv('x_bad_train.csv', header=None)

    # 然后产生连续14天的磁盘数据存入文件failure_disks.csv,这里处理需要选出其中包含故障天数的样本
    badDisk = geneFailureDisksSet(x_trainSet)

    # 挑选出连续14标签为0的加入健康样本，其余放入故障样本中
    for i in range(len(badDisk)):
        badDisk[i].to_csv('failure_disks.csv', mode='a', index=False, header=False)

    # ----------------故障磁盘数据处理完毕------------------#

    print('train set is ok!!!')


# 总方法:产生健康/故障磁盘测试数据，与上面的方法不同在于读取的日期不同，上面0-60，这里61-91,以及文件名不同
def geneSetData(fileName):
    # 第一步打乱数据，并要求其按照序列号组织好
    dataSet = shuffleDataSet(fileName)

    # 获得该数据集中不足60天的故障磁盘数据，产生故障磁盘数据集
    badDataSet = findBadData(dataSet)

    # 从总数据集中剔除错误的数据集,产生健康磁盘数据集
    goodDataSet = removeBadData(dataSet)

    # 针对健康磁盘数据集，分离出属性和标签，属性保存在文件x_train.csv,标签保存在y_train.csv
    # 每次运行代码时需要提前删除这两个文件,因为采用的是文件可增写的方式
    loadTestSet(goodDataSet)  # ------------------------------------------------------------这里不同

    # 接着从文件中读写出属性和标签
    x_trainSet = ps.read_csv('x_test.csv',
                             header=None)  # ------------------------------------------------------------这里不同
    y_trainSet = ps.read_csv('y_test.csv',
                             header=None)  # ------------------------------------------------------------这里不同

    # 然后产生连续14天的磁盘数据
    goodDisk, goodDiskLabel = geneDisksSet(x_trainSet, y_trainSet)

    # 写入健康磁盘文件test_health_disks.csv
    for i in range(len(goodDisk)):
        goodDisk[i].to_csv('test_health_disks.csv', mode='a', index=False,
                           header=False)  # ------------------------------------------------------------这里不同

    # ----------------健康磁盘数据处理完毕------------------#

    # 针对故障磁盘数据，同样也需要先分离属性和标签，和上面一样，但是步骤多了一步
    # 需要将不满60天的数据补充完整，所以采用的方法不同，并且存储的文件名也不同
    # 文件名分别为：x_bad_test.csv, y_bad_test.csv,
    loadBadTestSet(badDataSet)  # ------------------------------------------------------------这里不同

    # 接着从文件中读写出属性和标签
    x_trainSet = ps.read_csv('x_bad_test.csv',
                             header=None)  # ------------------------------------------------------------这里不同

    # 然后产生连续14天的磁盘数据存入文件test_failure_disks.csv,这里处理需要选出其中包含故障天数的样本
    badDisk = geneFailureDisksSet(x_trainSet)

    # 挑选出连续14标签为0的加入健康样本，其余放入故障样本中
    for i in range(len(badDisk)):
        badDisk[i].to_csv('test_failure_disks.csv', mode='a', index=False,
                          header=False)  # ------------------------------------------------------------这里不同

    # ----------------故障磁盘数据处理完毕------------------#

    print('test set is ok!!!')


# 产生训练集
# geneTrainData('WDC_disks.csv')
# geneSetData('WDC_disks.csv')

# geneTrainData('HIT_disks.csv')
geneSetData('HIT_disks.csv')

# geneTrainData('not_ST12_disks.csv')
# geneSetData('ST12_disks.csv')

# geneTrainData('all_disks.csv')



