import sqlite3
import numpy
import math
import random
SMART_NUMBER=12
ZERO_DIVIDE=-1
smart_id=numpy.zeros(SMART_NUMBER)
smart_dire=numpy.zeros(SMART_NUMBER)
smart_id=(1,8,10,13,17,19,24,373,387,394,396,398)
smart_dire=(1,0,0,1,1,1,0,1,1,0,0,0)
op = (">","<=")
data_type=("raw","normalized")
def get_SMART_disrtibution():
    #初始化数据库
    conn = sqlite3.connect("./SMART/drive_stats.db")
    c = conn.cursor()

    #变量
    smart_division = numpy.zeros((SMART_NUMBER, 10))  # 分为9个区间,(10个端点)
    smart_static = numpy.zeros((SMART_NUMBER, 9), dtype=float)
    # 区间的数值划分参考backblaze统计图的划分法
    smart_division[0] = (80, 95, 110, 125, 140, 155, 170, 185, 200, 256)
    smart_division[1] = (0, 13, 26, 39, 52, 65, 78, 91, 104, 500)
    smart_division[2] = (-0.1, 1, 4, 16, 70, 260, 1100, 4500, 17000, 70000)
    smart_division[3] = (58, 76, 94, 112, 130, 148, 166, 184, 202, 256)
    smart_division[4] = (34, 43, 52, 61, 70, 79, 88, 97, 106, 256)
    smart_division[5] = (0, 19, 39, 59, 79, 99, 100, 150, 200, 256)
    smart_division[6] = (-0.1, 13, 26, 39, 52, 65, 78, 91, 104, 600)
    smart_division[7] = (0, 13, 26, 39, 52, 65, 78, 91, 104, 256)
    smart_division[8] = (10, 40, 70, 100, 130, 160, 190, 220, 250, 256)
    smart_division[9] = (-0.1, 0.1, 2, 6, 16, 40, 100, 250, 650, 70000)
    smart_division[10] = (-0.1, 0.1, 2, 4, 6, 8, 10, 12, 14, 70000)
    smart_division[11] = (-0.1, 0.1, 2, 4, 8, 16, 35, 70, 130, 70000)
    #计算smart_static
    for i in range(0, SMART_NUMBER):
        true_id = math.ceil(smart_id[i] / 2)
        normal_or_raw = smart_id[i] % 2  # normal_or_raw=1代表normalized，raw=0代表raw
        for j in range(0,9):
            query = "select count(*) from STA_disk where smart_{}_{}>{} and smart_{}_{}<={}"
            #sql=query.format(true_id, data_type[normal_or_raw], smart_division[i][j],true_id, data_type[normal_or_raw], smart_division[i][j+1])
            cursor = c.execute(query.format(true_id, data_type[normal_or_raw], smart_division[i][j],true_id, data_type[normal_or_raw], smart_division[i][j+1]))
            for row in cursor:
                divid_all=row[0]
            query = "select count(*) from STA_disk where smart_{}_{}>{} and smart_{}_{}<={} and failure=1"
            sql = query.format(true_id, data_type[normal_or_raw], smart_division[i][j], true_id, data_type[normal_or_raw],smart_division[i][j + 1])
            cursor = c.execute(query.format(true_id, data_type[normal_or_raw], smart_division[i][j], true_id, data_type[normal_or_raw],smart_division[i][j + 1]))
            for row in cursor:
                divid_fail=row[0]
            #若在该smart区间没有盘（大然也就没有坏盘），坏盘率按0处理
            if(divid_all==0):
                smart_static[i][j]=0
                continue
            smart_static[i][j]=float(divid_fail)/float(divid_all)

    numpy.save("./temp_data/smart_static.npy",smart_static)
    cursor.close()
    conn.commit()
    conn.close()
    return smart_static

def get_correlation_matrix():
    #变量
    # smart_threshold设定为对年故障率贡献大于5%的SMART值界限
    # smart_10_nomolize在统计中所有磁盘都为100,smart_12_raw不能判断是越大越好还是越小越好
    # op数组：op[0]代表大于threshold磁盘不易坏,op[1]代表小于threshold磁盘不易坏，由于直方图区间为左开右闭，>不取=，小于取=
    # smart_id数组：SMART编号=ceil(value/2)，value%2=1为normalize，value%2=0为raw
    smart_threshold=numpy.zeros(SMART_NUMBER) #取自backblaze官网的统计表格(目测) https://www.backblaze.com/blog-smart-stats-2014-8.html#S8N,也可根据smart_static[][]自行设置
    smart_failure_list=numpy.zeros(SMART_NUMBER)
    correlation_matrix = numpy.zeros((SMART_NUMBER,SMART_NUMBER),dtype=float)
    smart_threshold = (95, 26, 4, 94, 43, 100, 26, 91, 130, 2, 0, 0)

    #初始化数据库
    conn = sqlite3.connect("./SMART/drive_stats.db")
    c = conn.cursor()

    #计算
    #1.smart_failure_list
    for i in range(0,SMART_NUMBER):
        true_id = math.ceil(smart_id[i] / 2)
        normal_or_raw=smart_id[i]%2 #normal_or_raw=1代表normalized，raw=0代表raw
        query= "select count(*) from STA_disk_failure where smart_{}_{}{}{}"
        cursor=c.execute(query.format(true_id,data_type[normal_or_raw],op[smart_dire[i]], smart_threshold[i]))
        for row in cursor:
            smart_failure_list[i]=row[0]

    #2.correlation_matrix
    for i in range(0,SMART_NUMBER):
        for j in range(0,SMART_NUMBER):
            true_id_i = math.ceil(smart_id[i] / 2)
            true_id_j = math.ceil(smart_id[j] / 2)
            normal_or_raw_i = smart_id[i] % 2
            normal_or_raw_j = smart_id[j] % 2
            query = "select count(*) from STA_disk_failure where smart_{}_{}{}{} and smart_{}_{}{}{}"
            cursor = c.execute(query.format(true_id_i, data_type[normal_or_raw_i], op[smart_dire[i]], smart_threshold[i],true_id_j, data_type[normal_or_raw_j], op[smart_dire[j]], smart_threshold[j]))
            for row in cursor:
                smart_i_j_failure=row[0]
            #避免除0,若smart_i=0，认为smart_i与smart_j无关，故correlation_ij=0
            if(smart_failure_list[i]==0):
                correlation_matrix[i][j] = 0
                continue
            correlation_matrix[i][j]=float(smart_i_j_failure)/float(smart_failure_list[i])

    numpy.save("./temp_data/correlation_matrix.npy",correlation_matrix)
    cursor.close()
    conn.commit()
    conn.close()

    return correlation_matrix

def get_sample(num,model=""):
    conn = sqlite3.connect("./SMART/drive_stats.db")
    c = conn.cursor()
    query = "select date,serial_number,model,failure,smart_1_normalized,smart_4_raw,smart_5_raw,smart_12_raw,smart_187_normalized,smart_193_normalized,smart_196_raw,smart_197_raw,smart_198_raw\
             from drive_stats where model=\"{}\" and failure!=-1 order by serial_number,date"
    cursor = c.execute(query.format(model))
    result=cursor.fetchall()

    assert num < len(result)
    if num==-1:
        num=len(result)
    #太大装不下,总共2000W条=16G,本机设置pycharm可用内存为5G
    sample=list()
    for i in range(num):
        sample.append(result[i])
    cursor.close()
    conn.commit()
    conn.close()

    return sample


