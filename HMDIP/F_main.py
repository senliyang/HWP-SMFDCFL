import os
import pandas as pd
from E_CV import *
from D_Net import transNet2
import torch
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import auc, roc_curve, precision_recall_curve, accuracy_score, recall_score, f1_score, precision_score
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
torch.manual_seed(2024)  # 设置 PyTorch 随机种子（CPU）
time =1
k = 5
N_FEATURES = 256
learning_rate = 0.001
epoch_num = 800
device = torch.device("cpu")
# 确保每次程序运行时生成相同的结果
torch.backends.cudnn.deterministic = True  # 使用确定性算法，保证结果一致
torch.backends.cudnn.benchmark = False  # 禁用 CUDNN 自动寻找最合适的算法

# main program
for cv in range(1, 5):
        print("第cv"+str(cv)+"----------------------------------------------------------------------")
        n_acc1 = []
        n_precision1 = []
        n_recall1 = []
        n_f11 = []
        n_AUC1 = []
        n_AUPR1 = []

        n_acc2 = []
        n_precision2 = []
        n_recall2 = []
        n_f12 = []
        n_AUC2 = []
        n_AUPR2 = []

        n_acc3 = []
        n_precision3 = []
        n_recall3 = []
        n_f13 = []
        n_AUC3 = []
        n_AUPR3 = []

        data_file = 'E_data' + '/64_data.csv'
        label_file = 'E_data' + '/label.csv'
        ##header=None不要列名，index_col=None：指定不使用任何列作为索引列。然把Dataframe对象转换为numpy数组。
        data = pd.read_csv(data_file, header=None, index_col=None).to_numpy()
        dlen = len(data)
        label = pd.read_csv(label_file, index_col=None, header=None).to_numpy()
        ##复制一个副本，用来保存原始数据的标签。
        label_copy = label.copy()
        ##表示标签数值的行数和列数。
        row, col = label.shape
        if cv == 4:
            #如果cv的值为4，那么创建一个名为 c 的 NumPy 数组，其中的元素是所有可能的标签索引的组合 (i, j)
            c = np.array([(i, j) for i in range(row) for j in range(col)])
        else:
            ##如果 cv 的值不为4，那么创建两个名为 a 和 b 的 NumPy 数组，分别存储标签为1和0的索引的组合 (i, j)
            a = np.array([(i, j) for i in range(row) for j in range(col) if label[i][j]])
            b = np.array([(i, j) for i in range(row) for j in range(col) if label[i][j] == 0])
            ##对数组b进行重排。
            np.random.shuffle(b)
            sample = len(a)
            ##将数组b截取为a的长度。
            b = b[:sample]
            ##定义空的NumPy 数组，用于存储计算结果
        mPREs = np.array([])
        mACCs = np.array([])
        mRECs = np.array([])
        mAUCs = np.array([])
        mAUPRs = np.array([])
        mF1 = np.array([])

        for j in range(time):
            if cv == 4:
                ##将变量 c 划分为训练集和测试集，结果分别存储在 c_tr 和 c_te 中
                c_tr, c_te = kfold(c, k=k, row=row, col=col, cv=cv)
            elif cv == 3:
                ##将变量 a 和 b 分别划分为训练集和测试集，结果分别存储在 a_tr、a_te、b_tr 和 b_te 中
                a_tr, a_te = kfold(a, k=k, row=row, col=col, cv=cv)
                b_tr, b_te = kfold(b, k=k, row=row, col=col, cv=cv)
            else:
                ##将变量 a 和 b 合并为变量 c，然后将 c 划分为训练集和测试集，结果存储在 c_tr 和 c_te 中
                c = np.vstack([a, b])
                c_tr, c_te = kfold(c, k=k, row=row, col=col, cv=cv)

            acc1 = 0
            precision1 = 0
            recall1 = 0
            f11 = 0
            AUC1 = 0
            AUPR1 = 0
            acc2 = 0
            precision2 = 0
            recall2 = 0
            f12 = 0
            AUC2 = 0
            AUPR2 = 0
            acc3 = 0
            precision3 = 0
            recall3 = 0
            f13 = 0
            AUC3 = 0
            AUPR3 = 0
            roc_curves = []
            pr_curves = []
            for i in range(k):

                if cv == 4:
                    b_tr = []
                    a_tr = []
                    # print(c_tr[i])
                    for ep in c_tr[i]:
                        if label_copy[c[ep][0]][c[ep][1]]:
                            b_tr.append(c[ep])
                        else:
                            a_tr.append(c[ep])
                    b_te = []
                    a_te = []
                    for ep in c_te[i]:
                        if label_copy[c[ep][0]][c[ep][1]]:
                            b_te.append(c[ep])
                        else:
                            a_te.append(c[ep])
                    b_te = np.array(b_te)
                    b_tr = np.array(b_tr)
                    a_te = np.array(a_te)
                    a_tr = np.array(a_tr)
                    np.random.shuffle(b_te)
                    np.random.shuffle(a_te)
                    np.random.shuffle(b_tr)
                    np.random.shuffle(a_tr)
                    a_tr = a_tr[:len(b_tr)]
                    a_te = a_te[:len(b_te)]
                    train_sample = np.vstack([a_tr, b_tr])
                    test_sample = np.vstack([a_te, b_te])
                elif cv == 3:
                    train_sample = np.vstack([np.array(a[a_tr[i]]), np.array(b[b_tr[i]])])
                    test_sample = np.vstack([np.array(a[a_te[i]]), np.array(b[b_te[i]])])
                else:
                    train_sample = np.array(c[c_tr[i]])
                    test_sample = np.array(c[c_te[i]])
                train_land = train_sample[:, 0] * col + train_sample[:, 1]
                test_land = test_sample[:, 0] * col + test_sample[:, 1]
                np.random.shuffle(train_land)
                np.random.shuffle(test_land)

                X_tr = data[train_land][:, :-1]
                y_tr = data[train_land][:, -1]
                X_te = data[test_land][:, :-1]
                y_te = data[test_land][:, -1]
                print("第"+str(i+1)+"折交叉验证----------------------------------------------------------------")
##深度神经网络模型DNN
                print('Begin Model1 Training:')
                ##创建一个 transNet2模型实例，N_FEATURES是输入特征的数量256，200为隐藏层的大小，1为输出层的大小，
                model = transNet2(N_FEATURES, 200, 1).to(device)
                ##创建一个Adam优化器，用于优化model1的参数，model.parameters() 返回模型中需要优化的参数，
                # lr=learning_rate 设置学习率为 learning_rate。
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                ##创建一个均方误差损失函数的实例，用于计算模型输出与目标值之间的损失。
                loss_fn = torch.nn.MSELoss().to(device)
                for epoch in range(epoch_num):
                    model.train()
                    ##将训练输入特征 X_tr 和目标值 y_tr 转换为 PyTorch 的浮点张量,用于存储和处理数据。
                    feature_train = torch.FloatTensor(X_tr)
                    target_train = torch.FloatTensor(y_tr)
                    ##将训练输入特征和目标值移动到指定的设备上（例如 GPU），提高训练速度。
                    train_tx = feature_train.to(device)
                    train_ty = target_train.to(device)
                    ##训练输入特征进行前向传播，生成模型的预测结果。
                    pred = model(train_tx)
                    ##计算预测结果与训练目标值之间的损失。
                    loss = loss_fn(pred, train_ty)
                    ##将优化器中的梯度缓存清零，准备进行反向传播。以确保每个训练迭代的梯度计算是基于当前迭代的损失值而不是累积的梯度。
                    optimizer.zero_grad()
                    ##对损失进行反向传播，计算梯度
                    loss.backward()
                    ##根据梯度更新模型的参数
                    optimizer.step()
                    if epoch % 50 == 0:
                        print(loss.item())
                ##对模型进行评估。
                model.eval()
                ##码将输入特征数据 X_te 转换为 PyTorch 中的浮点张量
                feature_test = torch.FloatTensor(X_te)
                ##目标值数据 y_te 转换为 PyTorch 中的长整型张量
                target_test = torch.LongTensor(y_te)
                test_tx = feature_test.to(device)
                test_ty = target_test.to(device)
                ##预测
                pred = model(test_tx)
                prob1 = pred.data.numpy()

                KT_y_prob_1 = np.arange(0, dtype=float)
                for k1 in prob1:
                    KT_y_prob_1 = np.append(KT_y_prob_1, k1)
                light_y1 = []
                for w1 in KT_y_prob_1:  # 0 1
                    if w1 > 0.5:
                        light_y1.append(1)
                    else:
                        light_y1.append(0)
                acc1 += accuracy_score(target_test, light_y1)
                precision1 += precision_score(target_test, light_y1)
                recall1 += recall_score(target_test, light_y1)
                f11 += f1_score(target_test, light_y1)

                fpr1, tpr1, thresholds1 = roc_curve(target_test, KT_y_prob_1)
                prec1, rec1, thr1 = precision_recall_curve(target_test, KT_y_prob_1)
                AUC1 += auc(fpr1, tpr1)
                AUPR1 += auc(rec1, prec1)


##interpretable boosting machine
                print('Begin Model2 Training:')
                ebm = ExplainableBoostingClassifier(learning_rate=0.01, interactions=100, early_stopping_tolerance=1e-2,max_bins=256, max_rounds=900, min_samples_leaf=3, max_leaves=3,random_state=2024)
                ebm.fit(X_tr, y_tr)
                y_ture = y_te  # 直接使用一维数组
                prob2 = ebm.predict_proba(X_te)[:, 1]
                KT_y_prob_2 = np.arange(0, dtype=float)
                for k2 in prob2:
                    KT_y_prob_2 = np.append(KT_y_prob_2, k2)
                light_y2 = []
                for w2 in KT_y_prob_2:  # 0 1
                    if w2 > 0.5:
                        light_y2.append(1)
                    else:
                        light_y2.append(0)

                acc2 += accuracy_score(y_ture, light_y2)
                precision2 += precision_score(y_ture, light_y2)
                recall2 += recall_score(y_ture, light_y2)
                f12 += f1_score(y_ture, light_y2)

                fpr2, tpr2, thresholds2 = roc_curve(y_ture, KT_y_prob_2)
                prec2, rec2, thr2 = precision_recall_curve(y_ture, KT_y_prob_2)
                AUC2 += auc(fpr2, tpr2)
                AUPR2 += auc(rec2, prec2)

                prob3 = 0.3 * prob1 + 0.7 * prob2


                KT_y_prob_3 = np.arange(0, dtype=float)
                for k3 in prob3:
                    KT_y_prob_3 = np.append(KT_y_prob_3, k3)
                light_y3 = []
                for w3 in KT_y_prob_3:  # 0 1
                    if w3 > 0.5:
                        light_y3.append(1)
                    else:
                        light_y3.append(0)
                # acc3 += accuracy_score(y_ture, light_y3)
                # precision3 += precision_score(y_ture, light_y3)
                # recall3 += recall_score(y_ture, light_y3)
                # f13 += f1_score(y_ture, light_y3)
                #
                # fpr3, tpr3, thresholds3 = roc_curve(y_ture, KT_y_prob_3)
                # prec3, rec3, thr3 = precision_recall_curve(y_ture, KT_y_prob_3)
                # roc_curves.append((fpr3, tpr3))
                # pr_curves.append((rec3, prec3))
                # AUC3 += auc(fpr3, tpr3)
                # AUPR3 += auc(rec3, prec3)
                acc3 = accuracy_score(y_ture, light_y3)
                precision3 = precision_score(y_ture, light_y3)
                recall3 = recall_score(y_ture, light_y3)
                f13 = f1_score(y_ture, light_y3)

                fpr3, tpr3, thresholds3 = roc_curve(y_ture, KT_y_prob_3)
                prec3, rec3, thr3 = precision_recall_curve(y_ture, KT_y_prob_3)
                roc_curves.append((fpr3, tpr3))
                pr_curves.append((rec3, prec3))
                AUC3 = auc(fpr3, tpr3)
                AUPR3 = auc(rec3, prec3)

                n_AUC3.append(AUC3)
                n_AUPR3.append(AUPR3)
                n_acc3.append(acc3)
                n_precision3.append(precision3)
                n_recall3.append(recall3)
                n_f13.append(f13)
            # acc1 = acc1 / 5
            # precision1 = precision1 / 5
            # recall1 = recall1 / 5
            # f11 = f11 / 5
            # AUC1 = AUC1 / 5
            # AUPR1 = AUPR1 / 5
            #
            # acc2 = acc2 / 5
            # precision2 = precision2 / 5
            # recall2 = recall2 / 5
            # f12 = f12 / 5
            # AUC2 = AUC2 / 5
            # AUPR2 = AUPR2 / 5

            # acc3 = acc3 / 5
            # precision3 = precision3 / 5
            # recall3 = recall3 / 5
            # f13 = f13 / 5
            # AUC3 = AUC3 / 5
            # AUPR3 = AUPR3 / 5

            # print('--------------------------------------model1结果---------------------------------------------')
            # print("accuracy:%.4f" % acc1)
            # print("precision:%.4f" % precision1)
            # print("recall:%.4f" % recall1)
            # print("F1 score:%.4f" % f11)
            # print("AUC:%.4f" % AUC1)
            # print("AUPR:%.4f" % AUPR1)
            #
            # print('--------------------------------------model2结果---------------------------------------------')
            # print("accuracy:%.4f" % acc2)
            # print("precision:%.4f" % precision2)
            # print("recall:%.4f" % recall2)
            # print("F1 score:%.4f" % f12)
            # print("AUC:%.4f" % AUC2)
            # print("AUPR:%.4f" % AUPR2)
            #
            # print('--------------------------------------集成结果---------------------------------------------')
            # print("accuracy:%.4f" % acc3)
            # print("precision:%.4f" % precision3)
            # print("recall:%.4f" % recall3)
            # print("F1 score:%.4f" % f13)
            # print("AUC:%.4f" % AUC3)
            # print("AUPR:%.4f" % AUPR3)

            # n_AUC1=np.append(n_AUC1,AUC1)
            # n_AUPR1=np.append(n_AUPR1,AUPR1)
            # n_acc1=np.append(n_acc1,acc1)
            # n_precision1=np.append(n_precision1,precision1)
            # n_recall1=np.append(n_recall1,recall1)
            # n_f11=np.append(n_f11,f11)
            #
            # n_AUC2=np.append(n_AUC2,AUC2)
            # n_AUPR2=np.append(n_AUPR2,AUPR2)
            # n_acc2=np.append(n_acc2,acc2)
            # n_precision2=np.append(n_precision2,precision2)
            # n_recall2=np.append(n_recall2,recall2)
            # n_f12=np.append(n_f12,f12)

            # n_AUC3=np.append(n_AUC3,AUC3)
            # n_AUPR3=np.append(n_AUPR3,AUPR3)
            # n_acc3=np.append(n_acc3,acc3)
            # n_precision3=np.append(n_precision3,precision3)
            # n_recall3=np.append(n_recall3,recall3)
            # n_f13=np.append(n_f13,f13)
            # 假设 roc_curves 是 (fpr, tpr) 的列表
            # 计算平均 ROC 曲线
            mean_fpr = np.linspace(0, 1, 100)

            # 对齐所有 fpr-tpr 曲线，并确保插值过程无误
            mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in roc_curves], axis=0)

            # 确保 TPR 单调递增，以符合 ROC 曲线的特性
            mean_tpr = np.maximum.accumulate(mean_tpr)  # 确保 TPR 单调递增

            # 计算平均 AUC ROC
            mean_auc = auc(mean_fpr, mean_tpr)

            # 保存平均 ROC 曲线
            roc_curve_df = pd.DataFrame({'FPR': mean_fpr, 'TPR': mean_tpr})
            roc_curve_df.to_csv(f'./Result/average_roc_cv{cv}_{mean_auc:.4f}.csv', index=False)

            # 计算平均 PR 曲线
            mean_rec = np.linspace(0, 1, 100)
            # 使用 np.maximum.accumulate 来确保 Precision 单调递减（PR 曲线的特性）
            mean_pre = np.mean([np.interp(mean_rec, rec_[::-1], pre[::-1]) for rec_, pre in pr_curves], axis=0)
            mean_pre = np.maximum.accumulate(mean_pre[::-1])[::-1]  # Ensure precision is monotonically decreasing
            # 计算平均 AUC PR
            mean_auc_pr = auc(mean_rec, mean_pre)
            # 保存 PR 曲线数据
            pr_curve_df = pd.DataFrame({'Recall': mean_rec, 'Precision': mean_pre})
            pr_curve_df.to_csv(f'./Result/average_pr_cv{cv}_{mean_auc_pr:.4f}.csv', index=False)

            print(f"Average AUC for ROC (CV {cv}): {mean_auc}")
            print(f"Average AUC for PR (CV {cv}): {mean_auc_pr}")

            # 重置曲线列表，以便下一个 CV 使用
            roc_curves = []
            pr_curves = []

        # model1输出
        # mean_AUC1 = np.mean(n_AUC1)
        # mean_AUPR1 = np.mean(n_AUPR1)
        # mean_acc1 = np.mean(n_acc1)
        # mean_precision1 = np.mean(n_precision1)
        # mean_recall1 = np.mean(n_recall1)
        # mean_f11 = np.mean(n_f11)
        #
        # std_AUC1 = np.std(n_AUC1)
        # std_AUPR1 = np.std(n_AUPR1)
        # std_acc1 = np.std(n_acc1)
        # std_precision1 = np.std(n_precision1)
        # std_recall1 = np.std(n_recall1)
        # std_f11 = np.std(n_f11)


        # print('--------------------------------------model1平均结果---------------------------------------------')
        # print("accuracy:%.4f" % mean_acc1)
        # print("precision:%.4f" % mean_precision1)
        # print("recall:%.4f" % mean_recall1)
        # print("F1 score:%.4f" % mean_f11)
        # print("AUC:%.4f" % mean_AUC1)
        # print("AUPR:%.4f" % mean_AUPR1)
        #
        # print('--------------------------------------model1平均std---------------------------------------------')
        # print("accuracy:%.4f" % std_acc1)
        # print("precision:%.4f" % std_precision1)
        # print("recall:%.4f" % std_recall1)
        # print("F1 score:%.4f" % std_f11)
        # print("AUC:%.4f" % std_AUC1)
        # print("AUPR:%.4f" % std_AUPR1)
        #
        # with open('./Result/main_DNN.csv', 'a', encoding='utf-8') as f:
        #     f.write(f"{cv},"
        #             f"{mean_AUC1:.4f},{std_AUC1:.4f},{mean_AUPR1:.4f},{std_AUPR1:.4f},"
        #             f"{mean_f11:.4f},{std_f11:.4f},"f"{mean_recall1:.4f},"f"{std_recall1:.4f},"
        #             f"{mean_precision1:.4f},{std_precision1:.4f},{mean_acc1:.4f},{std_acc1:.4f}\n")

        # model2输出
        # mean_AUC2 = np.mean(n_AUC2)
        # mean_AUPR2 = np.mean(n_AUPR2)
        # mean_acc2 = np.mean(n_acc2)
        # mean_precision2 = np.mean(n_precision2)
        # mean_recall2 = np.mean(n_recall2)
        # mean_f12 = np.mean(n_f12)
        #
        # std_AUC2 = np.std(n_AUC2)
        # std_AUPR2 = np.std(n_AUPR2)
        # std_acc2 = np.std(n_acc2)
        # std_precision2 = np.std(n_precision2)
        # std_recall2 = np.std(n_recall2)
        # std_f12 = np.std(n_f12)


        # print('--------------------------------------model2平均结果---------------------------------------------')
        # print("accuracy:%.4f" % mean_acc2)
        # print("precision:%.4f" % mean_precision2)
        # print("recall:%.4f" % mean_recall2)
        # print("F1 score:%.4f" % mean_f12)
        # print("AUC:%.4f" % mean_AUC2)
        # print("AUPR:%.4f" % mean_AUPR2)
        #
        # print('--------------------------------------model2平均std---------------------------------------------')
        # print("accuracy:%.4f" % std_acc2)
        # print("precision:%.4f" % std_precision2)
        # print("recall:%.4f" % std_recall2)
        # print("F1 score:%.4f" % std_f12)
        # print("AUC:%.4f" % std_AUC2)
        # print("AUPR:%.4f" % std_AUPR2)
        #
        # with open('./Result/main_IBM.csv', 'a', encoding='utf-8') as f:
        #     f.write(f"{cv},"
        #             f"{mean_AUC2:.4f},{std_AUC2:.4f},{mean_AUPR2:.4f},{std_AUPR2:.4f},"
        #             f"{mean_f12:.4f},{std_f12:.4f},"f"{mean_recall2:.4f},"f"{std_recall2:.4f},"
        #             f"{mean_precision2:.4f},{std_precision2:.4f},{mean_acc2:.4f},{std_acc2:.4f}\n")

        # 集成输出
        mean_AUC3 = np.mean(n_AUC3)
        mean_AUPR3 = np.mean(n_AUPR3)
        mean_acc3 = np.mean(n_acc3)
        mean_precision3 = np.mean(n_precision3)
        mean_recall3 = np.mean(n_recall3)
        mean_f13 = np.mean(n_f13)

        std_AUC3 = np.std(n_AUC3)
        std_AUPR3 = np.std(n_AUPR3)
        std_acc3 = np.std(n_acc3)
        std_precision3 = np.std(n_precision3)
        std_recall3 = np.std(n_recall3)
        std_f13 = np.std(n_f13)


        print('--------------------------------------集成平均结果---------------------------------------------')
        print("accuracy:%.4f" % mean_acc3)
        print("precision:%.4f" % mean_precision3)
        print("recall:%.4f" % mean_recall3)
        print("F1 score:%.4f" % mean_f13)
        print("AUC:%.4f" % mean_AUC3)
        print("AUPR:%.4f" % mean_AUPR3)

        print('--------------------------------------集成平均std---------------------------------------------')
        print("accuracy:%.4f" % std_acc3)
        print("precision:%.4f" % std_precision3)
        print("recall:%.4f" % std_recall3)
        print("F1 score:%.4f" % std_f13)
        print("AUC:%.4f" % std_AUC3)
        print("AUPR:%.4f" % std_AUPR3)

        with open('./Result/main_ENS.csv', 'a', encoding='utf-8') as f:
            f.write(f"{cv},"
                    f"{mean_AUC3:.4f},{std_AUC3:.4f},{mean_AUPR3:.4f},{std_AUPR3:.4f},"
                    f"{mean_f13:.4f},{std_f13:.4f},"f"{mean_recall3:.4f},"f"{std_recall3:.4f},"
                    f"{mean_precision3:.4f},{std_precision3:.4f},{mean_acc3:.4f},{std_acc3:.4f}\n")