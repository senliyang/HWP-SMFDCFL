from __future__ import division, print_function, absolute_import
from model import *
import numpy as np
import torch


# 定义卷积层

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=400,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-7,
                    help='Weight decay.')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

set_seed(args.seed, args.cuda)

nm = 1177
nd = 134
nc = 4499
import pandas as pd
# 读取数据
miRNAnumber = pd.read_excel('mname.xlsx')
# diseasenumber = pd.read_excel('dname.xlsx')

# MS = np.loadtxt("../A_CKA-MKL/CKA_microbe.csv", delimiter=',')
MS = np.loadtxt("SNF-M.txt", delimiter=' ')

MS = normalized(MS)
# DS = normalized(DS)

# DS = torch.from_numpy(DS).float()
MS = torch.from_numpy(MS).float()
A = np.zeros((nm, nd), dtype=float)
ConnectData = np.loadtxt(r'coordinates.txt', dtype=int) - 1

for i in range(nc):
    A[ConnectData[i, 0], ConnectData[i, 1]] = 1

# 取所有未关联样本对的坐标用data0_index保存起来
data0_index = np.argwhere(A == 0)  # (184155,2)
np.savetxt('unknown association.txt', data0_index, delimiter=' ', fmt='%d')
A = torch.from_numpy(A).float()
# miRNA_feature = torch.cat((A,MS),1)
# disease_feature = torch.cat((A.t(),DS),1)
miRNA_feature = MS
# disease_feature = DS

if args.cuda:
    A = A.cuda()
    # DS = DS.cuda()
    MS = MS.cuda()
    miRNA_feature = miRNA_feature.cuda()
    # disease_feature = disease_feature.cuda()

def train(sgae, y0, adj, epoch):
    optp = torch.optim.Adam(sgae.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for e in range(1, epoch + 1):
        sgae.train()
        z1, z2, z3, y1, y2, y3 = sgae(adj, y0)
        loss = sgae.my_mse_loss(adj, y0)
        optp.zero_grad()
        loss.backward()
        optp.step()
        sgae.eval()
        with torch.no_grad():
            z1, z2, z3, y1, y2, y3 = sgae(adj,y0)

        if e % 20 == 0 and e != 0:
            print('Epoch %d | Loss: %.4f' % (e, loss.item()))

    return z1, z2, z3


def trainres(mf, adjm):
    sgaem = SGAE(256, 128, 64, 1177 )
    # sgaed = SGAE(256, 128, 64, 134 )
    if args.cuda:
        sgaem = sgaem.cuda()
        # sgaed = sgaed.cuda()
    zm1, zm2, zm = train(sgaem, mf, adjm, args.epochs)
    # zd1, zd2, zd = train(sgaed, df, adjd,args.epochs)
    return zm


zm = trainres(miRNA_feature,MS)
zm = zm.cpu().detach().numpy()
np.savetxt('SGAE_64_microbe_feature.csv', zm, delimiter=',')  # fmt 控制浮点数精度
