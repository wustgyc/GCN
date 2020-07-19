import numpy as np
import pandas as pd
import parameter
from get_correlation_matrix import get_sample
from sklearn.impute import SimpleImputer
from datetime import datetime


def datelist(beginDate, endDate):
    # beginDate, endDate是形如‘20160601’的字符串或datetime格式
    date_l = [datetime.strftime(x, '%Y-%m-%d') for x in list(pd.date_range(start=beginDate, end=endDate))]
    return date_l


def date_sub(date1, date2):
    fmt = '%Y-%m-%d'
    dt1 = datetime.strptime(date1, fmt).date()
    dt2 = datetime.strptime(date2, fmt).date()
    return (dt1 - dt2).days


def data_cleaning():
    # 0----------------------------获取数据
    sample = get_sample(-1,model="ST10000NM0086")  # 全部
    print("获取数据成功!")
    # 1.1-----------------------------删除无效数据
    for i, row in enumerate(sample):
        while i < len(sample) and sample[i][4] == "" and sample[i][5] == "" and sample[i][6] == "" and sample[i][
            7] == "" and sample[i][8] == "" and sample[i][9] == "" and sample[i][10] == "" and sample[i][11] == "" and \
                sample[i][12] == "":
            del sample[i]

    # 1.2----------------------------去重（STA不必去重）
    print("去重成功!")

    # 2----------------------------按serial_number分出子表
    serial_list = list()
    serial_number = ""
    serial_list.append([])
    for i, row in enumerate(sample):
        if i + 1 < len(sample) and sample[i + 1][1] != serial_number:
            serial_number = sample[i + 1][1]
            serial_list[len(serial_list) - 1].append(row)
            serial_list.append([])
            continue
        # 坏盘复活当作新盘对待
        if i + 1 < len(sample) and row[3] == 1 and sample[i + 1][1] == serial_number:
            serial_list[len(serial_list) - 1].append(row)
            serial_list.append([])
            continue

        # 某块磁盘数据太久未收集当作新盘对待
        if i + 1 < len(sample) and date_sub(sample[i + 1][0], sample[i][0]) >= parameter.DIVIDE_DISTANCE:
            serial_list[len(serial_list) - 1].append(row)
            serial_list.append([])
            continue
        serial_list[len(serial_list) - 1].append(row)
    if len(serial_list[len(serial_list) - 1]) == 0:
        del serial_list[len(serial_list) - 1]
    print("按serial_number分离子表成功!")

    # 3----------------------------补全未收集数据,使数据连续
    # 3.1--------------------------删除稀疏数据
    for disk_id, disk in enumerate(serial_list):
        # and后第一项删除了密度太小的数据，第二项删除了长度太短的数据
        while disk_id < len(serial_list) and (len(serial_list[disk_id]) / (
                date_sub(serial_list[disk_id][len(serial_list[disk_id]) - 1][0],
                         serial_list[disk_id][0][0]) + 1) < parameter.DROP_RATE or len(
                serial_list[disk_id]) <= parameter.DROP_NUM):
            del serial_list[disk_id]
    # 3.2--------------------------补全非稀疏数据
    for disk_id, disk in enumerate(serial_list):
        date_list = datelist(disk[0][0], "2019-12-31")
        j = 0
        for i, row in enumerate(disk):
            while date_list[j] < row[0]:  # 未收集该天的数据
                tmp = list(row)  # 用下一个样本填充此处缺失的样本，观察数据知，下一个样本几乎不会是fail的，而且是fail的情况下时间宽度也只有一天
                tmp[0] = date_list[j]  # 除时间外其他数据保持一致
                tmp[3] = 0  # fail补0
                serial_list[disk_id].insert(j, tuple(tmp))
                j += 1
            # 日期对齐了
            # 在遍历列表过程中向前插入数据，会在下一步遍历到刚刚插入的数据，因此要等到disk[i+1][0]>date_list[j]时j再往前走一天
            if i + 1 < len(disk) and disk[i + 1][0] > date_list[j]:
                j += 1
    print("样本补全成功!")

    # 4---------------------------字段内缺失值补全，放在后面，此处数据无明显结构，还有各种类型的变量，不易处理

    # 5----------------------------改进标签（磁盘故障前6天failure=1）
    label = list()  # tuple不可更改，用list记录
    for disk_id, disk in enumerate(serial_list):
        label.append([])
        for sample_id, row in enumerate(disk):
            label[disk_id].append(serial_list[disk_id][sample_id][3])  # 记录标签
            serial_list[disk_id][sample_id] = serial_list[disk_id][sample_id][4:]  # sample全数字化
            if row[3] == 1:  # 此处存疑，有的磁盘fail=1但后面还有fail=0的样本，不知道是错误数据，还是盘又修好了
                for i in range(7):
                    if sample_id - i < 0:
                        break;
                    label[disk_id][sample_id - i] = 1
    # 5.1把""替换为np.nan，这样serial_list全是数字
    for disk_id, disk in enumerate(serial_list):
        for sample_id, row in enumerate(disk):
            for i, ch in enumerate(row):
                if ch == "":
                    tmp = list(serial_list[disk_id][sample_id])
                    tmp[i] = np.nan
                    serial_list[disk_id][sample_id] = tuple(tmp)

    print("标签改进成功!")
    return serial_list, label


def data_serializing():
    sample, label = data_cleaning()
    print("加载数据成功!")

    sample_flatten = []
    label_flatten = []
    assert len(sample) == len(label)
    # 序列化
    for disk_i, disk in enumerate(sample):
        assert len(sample[disk_i]) == len(label[disk_i])

        res = len(sample[disk_i]) % parameter.TIME_STEP  # 时间窗口无法填满的时候，舍弃一些数据

        sample[disk_i] = sample[disk_i][res:]  # 多余的数据应该删前面的，因为如果删后面的可能会把所有fail样本删掉(如果time_step太大的话)
        sample[disk_i] = np.reshape(sample[disk_i], (-1, parameter.TIME_STEP, parameter.SMART_NUMBER))
        label[disk_i] = label[disk_i][res:]
        label[disk_i] = np.reshape(label[disk_i], (-1, parameter.TIME_STEP))

        for row in sample[disk_i]:
            sample_flatten.append(row)
        for row in label[disk_i]:
            label_flatten.append(row)

    sample = np.array(sample_flatten)
    label = np.array(label_flatten)

    step_label = np.zeros((len(label)), dtype=int)
    # 为每个分组做标签
    for i, row in enumerate(label):
        if np.sum(row) >= 7:  # 之前已对标签做过改进,fail前6天均视为fail
            step_label[i] = 1
        else:
            step_label[i] = 0

    # 缺失值补全
    #防止某列全nan被删除,若有，把第一个nan设为0
    sample = np.reshape(sample, (-1, parameter.SMART_NUMBER))
    for i in range(parameter.SMART_NUMBER):
        count=0
        for j,row in enumerate(sample):
            if np.isnan(row[i]):
                count+=1
        if count==len(sample):
            sample[0][i]=0
    imp_median = SimpleImputer(missing_values=np.nan, strategy='median')  # 中值补全
    sample = imp_median.fit_transform(sample)
    sample = np.reshape(sample, (-1, parameter.TIME_STEP, parameter.SMART_NUMBER))

    np.save("./temp_data/sample_ST10000NM0086.npy", sample)  #数据依次按serial_number,date排序
    np.save("./temp_data/label_ST10000NM0086.npy", step_label)
    print("序列化完毕!")


if __name__ == '__main__':
    data_serializing()
