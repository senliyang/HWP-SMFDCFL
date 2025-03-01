import numpy as np
import pickle
import pandas as pd


interMatrix = pd.read_csv('A_SNF/MD_A.csv', header=0, index_col=0).values
print(interMatrix)
##interMatrix 的行数和列数分别赋值给变量rows和cols
rows, cols = interMatrix.shape
print('matrix shape:', interMatrix.shape)
##创建一个空列表。
rd_pairs = []
##然后使用嵌套的for循环遍历rows和cols，并将索引 i、j 和 interMatrix[i,j]组成的列表添加到 rd_pairs 中。
for i in range(rows):
    for j in range(cols):
        rd_pairs.append([i,j,interMatrix[i,j]])
rd_pairs = np.array(rd_pairs).reshape(-1,3)
print(rd_pairs)
np.savetxt("A_SNF/pair.txt", rd_pairs, fmt='%d')
print("Finished")