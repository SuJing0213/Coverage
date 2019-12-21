import csv
import numpy as np


def loadDataSetnoheader(birth_weight_file):
    dislist = []
    with open(birth_weight_file) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        for row in csv_reader:  # 将csv 文件中的数据保存到birth_data中
            dislist.append(row)

    return dislist


def loadDataSet(birth_weight_file):
    dislist = []
    with open(birth_weight_file) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        birth_header = next(csv_reader)  # 读取第一行每一列的标题
        for row in csv_reader:  # 将csv 文件中的数据保存到birth_data中
            dislist.append(row)
    return dislist


def savecsvnoheader(name, list):
    with open(name, "w", newline="") as myfileT:
        mywrite = csv.writer(myfileT)
        # mywrite.writerow(header)
        for row in list:
            mywrite.writerow(row)
    myfileT.close()


if __name__ == '__main__':
    # clistname='ff'
    csvname = 'conv/convpoolfc-test.csv'
    cplist = loadDataSetnoheader(csvname)
    picnum = 10000
    # conv1 = cplist[0]
    # conv2 = cplist[1]
    # pool1 = cplist[2]
    # pool2 = cplist[3]
    # fc1 = cplist[4]
    # fc2 = cplist[5]
    # fc3 = cplist[6]
    #
    # neurenumfc1 = 120
    # neurenumfc2 = 84
    # neurenumfc3 = 10
    # neurenum1 = 6
    # neurenum2 = 16
    # neurenump1 = 6
    # neurenump2 = 16
    a = [6, 16, 6, 16, 120, 84, 10]

    covernum = 0
    for lay in range(7):
        print('lay', lay)
        neurenum = a[lay]
        data = cplist[lay]
        value = 0
        for i in range(neurenum):
            for pic in range(picnum):
                if float(data[pic * neurenum + i]) > 0:
                    value = value + 1
                    covernum = covernum + 1
                    break
        print(value / a[lay])
    print(covernum / 258)
