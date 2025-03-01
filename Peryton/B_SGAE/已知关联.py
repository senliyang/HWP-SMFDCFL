import numpy as np
import pandas as pd

# 读取数据
df = pd.read_csv('interaction.txt', delimiter='\t',header=None,index_col=None)

# 获取所有值为1的元素的坐标
coordinates_1 = np.argwhere(df.T.values == 1)

# 获取所有值为0的元素的坐标
coordinates_0 = np.argwhere(df.T.values == 0)

# 将坐标加1，使索引从1开始
coordinates_1 += 1
coordinates_0 += 1

# 合并坐标并添加标记
combined_coordinates = np.vstack([
    np.hstack([coordinates_1, np.ones((coordinates_1.shape[0], 1), dtype=int)]),  # 添加1标记
    np.hstack([coordinates_0, np.zeros((coordinates_0.shape[0], 1), dtype=int)])   # 添加0标记
])

# 保存合并后的坐标到新的txt文件
np.savetxt('coordinates.txt', combined_coordinates, fmt='%d', delimiter=' ', header='row col value')
