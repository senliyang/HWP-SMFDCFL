import numpy as np
import pandas as pd


f_r = pd.read_csv('E_data/32_microbe_feature.csv',header=None,index_col=None).values
f_d = pd.read_csv('E_data/32_disease_feature.csv',header=None,index_col=None).values
print(f_r.shape)
print(f_d.shape)

all_associations = pd.read_csv('A_SNF' + '/pair.txt', sep=' ', names=['r', 'd', 'label'])

label = pd.read_csv('A_SNF/interaction.txt', header=None, index_col=None,sep="\t")

label.to_csv("E_data/label.csv",header=False,index=False)

dataset = []

for i in range(int(all_associations.shape[0])):
    r = all_associations.iloc[i, 1]
    c = all_associations.iloc[i, 0]
    label = all_associations.iloc[i, 2]
    dataset.append(np.hstack((f_r[r], f_d[c], label)))

all_dataset = pd.DataFrame(dataset)

all_dataset.to_csv("E_data/32_data.csv",header=None,index=None)

print("Fnished!")