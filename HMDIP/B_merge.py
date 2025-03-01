import pandas as pd
import numpy as np

# 融合线性特征和非线性特征

# 线性特征
microbe_NMF = pd.read_csv("C_NMF/NMF_64_microbe_feature.csv",header=None)
disease_NMF = pd.read_csv("C_NMF/NMF_64_disease_feature.csv",header=None)

# SGAE 非线性特征
microbe_SGAE = pd.read_csv("B_SGAE/SGAE_64_microbe_feature.csv",header=None)

# VATE 非线性特征
disease_VATE = pd.read_csv("D_VGAE/VGAE_64_disease_feature.csv",header=None)

print(microbe_NMF.shape)
print(disease_NMF.shape)
print(microbe_SGAE.shape)
print(disease_VATE.shape)

# 特征合并
microbe_feature = np.hstack((microbe_NMF, microbe_SGAE))
disease_feature = np.hstack((disease_NMF, disease_VATE))

# 将 NumPy 数组转换为 DataFrame
microbe_feature = pd.DataFrame(microbe_feature)
disease_feature = pd.DataFrame(disease_feature)

# 保存到 CSV 文件
microbe_feature.to_csv("E_data/64_microbe_feature.csv",header=None,index=None)
disease_feature.to_csv("E_data/64_disease_feature.csv",header=None,index=None)

print(microbe_feature.shape)
print(disease_feature.shape)

print("Finished")