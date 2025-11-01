# ======================================================================
#
# -*- coding: utf-8 -*-
#
# ======================================================================

# 'data' is the sample to be tested, 'row' is the number of rows, 'col' is the number of columns
# CV1 represents rows, CV2 represents columns
##交叉验证（cv1 cv2 cv3 cv4）
##导入所需的库和模块：numpy用于数值计算，KFold用于交叉验证
import numpy as np
from sklearn.model_selection import KFold
np.random.seed(2024)
##定义了一个名为 kfold 的函数，接受以下参数：
#data：输入的数据。
#k：交叉验证的折数。
#row：数据的行数（默认为0）。
#col：数据的列数（默认为0）。
#cv：交叉验证的模式（默认为3）
def kfold(data, k, row=0, col=0, cv=3):
    dlen = len(data)
##根据 cv 参数值的不同，确定交叉验证中测试集的大小。然后创建空列表 test_res 和 train_res
##用于存储测试集和训练集的索引。接着，创建一个长度为 lens 的列表 d，并使用 KFold 进行索引的划分。
    if cv != 4:
        if cv == 1:
            lens = row
        elif cv == 2:
            lens = col
        else:
            lens = dlen
        test_res = []
        train_res = []
        ##创建一个从0到lens-1 的整数列表d；lens:整数值，表示数据集的长度或大小.
        d = list(range(lens))
        ##创建一个kFold对象，将数据划分为5个部分，(shuffle=True)进行随机排序，把数据平均分成五份，四份拿来训练，一份拿来测试。
        kf = KFold(5, shuffle=True,random_state=2024)
        ##使用KFold对象的 split 方法对 d 进行索引的划分,将返回的划分后的索引赋值给d
        d = kf.split(d)
        ##遍历d中的每个划分，将划分后的测试集索引存储在test_res列表中，将划分后的训练集索引存储在 train_res列表中。
        for i in d:
            test_res.append(list(i[1]))
            train_res.append(list(i[0]))
        if cv == 3:
            return train_res, test_res
        else:
            train_s = []
            test_s = []
            for i in range(k):
                train_ = []
                test_ = []
                for j in range(dlen):
                    if data[j][cv - 1] in test_res[i]:
                        test_.append(j)
                    else:
                        train_.append(j)
                train_s.append(train_)
                test_s.append(test_)
            return train_s, test_s
    else:
        r = list(range(row))
        c = list(range(col))
        kf = KFold(5, shuffle=True,random_state=2024)
        r = kf.split(r)
        c = kf.split(c)
        r_test_res = []
        r_train_res = []
        c_test_res = []
        c_train_res = []
        for i in r:
            r_test_res.append(list(i[1]))
            r_train_res.append(list(i[0]))
        for i in c:
            c_test_res.append(list(i[1]))
            c_train_res.append(list(i[0]))
        train_s = []
        test_s = []
        for i in range(k):
            train_ = []
            test_ = []
            for m in range(dlen):
                flag_1 = False
                flag_2 = False
                if data[m][0] in r_test_res[i]:
                    flag_1 = True
                if data[m][1] in c_test_res[i]:
                    flag_2 = True
                if flag_1 and flag_2:
                    test_.append(m)
                if (not flag_1) and (not flag_2):
                    train_.append(m)
            train_s.append(train_)
            test_s.append(test_)
        return train_s, test_s


def get_one_hot(targets, nb_classes) -> object:
    """
    :rtype: object
    """
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])