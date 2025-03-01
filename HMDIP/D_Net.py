import torch
import torch.nn as nn
torch.manual_seed(2024)  # 设置 PyTorch 随机种子（CPU）
torch.cuda.manual_seed_all(2024)  # 设置 GPU 随机种子（如果使用 GPU）# 确保每次程序运行时生成相同的结果
torch.backends.cudnn.deterministic = True  # 使用确定性算法，保证结果一致
torch.backends.cudnn.benchmark = False  # 禁用 CUDNN 自动寻找最合适的算法
class transNet2(nn.Module):
    def __init__(self, in_feature_num,hidden,output):
        super(transNet2, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_feature_num, hidden),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden // 2, output)
)

    def forward(self, x):
        out = self.fc(x)
        out = torch.sigmoid(out)
        return out.squeeze(-1)