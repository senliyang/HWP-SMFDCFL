import pandas as pd
import numpy as np

# ---------- 读取线性特征 ----------
microbe_DPMF = pd.read_csv("FM1_U.txt", header=None).values
disease_DPMF = pd.read_csv("FD1_V.txt", header=None).values

# ---------- 读取非线性特征 (DCFL 输出的 .npy 文件) ----------
microbe_DCFL = np.load("FM2_microbe.npy")
disease_DCFL = np.load("FD2_disease.npy")

print("microbe_DPMF:", microbe_DPMF.shape)
print("disease_DPMF:", disease_DPMF.shape)
print("microbe_DCFL:", microbe_DCFL.shape)
print("disease_DCFL:", disease_DCFL.shape)

# ---------- 特征维度检查 ----------
if microbe_DPMF.shape[0] != microbe_DCFL.shape[0]:
    raise ValueError("微生物样本数量不一致，请检查 DPMF 与 DCFL 输出顺序！")
if disease_DPMF.shape[0] != disease_DCFL.shape[0]:
    raise ValueError("疾病样本数量不一致，请检查 DPMF 与 DCFL 输出顺序！")

# ---------- 融合线性与非线性特征 ----------
microbe_feature = np.hstack((microbe_DPMF, microbe_DCFL))
disease_feature = np.hstack((disease_DPMF, disease_DCFL))

# ---------- 保存 ----------
pd.DataFrame(microbe_feature).to_csv("64_microbe_feature.csv", header=None, index=None)
pd.DataFrame(disease_feature).to_csv("64_disease_feature.csv", header=None, index=None)

print("microbe_feature:", microbe_feature.shape)
print("disease_feature:", disease_feature.shape)
print("Finished ✓")
