import numpy as np
import time

# ---------- 工具函数 ----------

def new_normalization(w: np.ndarray) -> np.ndarray:
    """与原SNF一致：主对角0.5，非对角按行和分配到剩余0.5。"""
    m = w.shape[0]
    p = np.zeros((m, m), dtype=float)
    row_sums = w.sum(axis=1) - np.diag(w)
    for i in range(m):
        p[i, i] = 0.5
        denom = row_sums[i]
        if denom > 0:
            p[i, :] += 0.5 * (w[i, :] / denom)
            p[i, i] = 0.5  # 确保主对角仍为0.5
    return p


def weighted_knn_matrix(S: np.ndarray, k: int, decay: float = 0.85) -> tuple[np.ndarray, list[set]]:
    """
    WKNN 邻接矩阵 Sh ，并返回每个节点的KNN集合 Ni。
    - 对每一行 i，选取除自身外相似度最高的 k 个邻居 Ni；
    - 给这 k 个邻居施加指数衰减权重 w^ℓ（ℓ=0..k-1），再对k个权重归一化；
    - Sh[i, j in Ni] = ( 归一化权重_j ) * ( S[i, j] / sum_{j in Ni} S[i, j] )
    """
    n = S.shape[0]
    Sh = np.zeros((n, n), dtype=float)
    knn_sets = []

    for i in range(n):
        # 排序（降序），排除自身
        idx = np.argsort(S[i, :])[::-1]
        idx = idx[idx != i]
        Ni = idx[:k]
        knn_sets.append(set(Ni.tolist()))

        # 相似度和（仅在 Ni 上）
        denom_sim = S[i, Ni].sum()
        if denom_sim <= 0:
            continue

        # 衰减权重（w^0, w^1, ..., w^{k-1}），并归一化
        raw_w = decay ** np.arange(len(Ni), dtype=float)
        raw_w_sum = raw_w.sum()
        if raw_w_sum == 0:
            weights = np.ones_like(raw_w) / len(raw_w)
        else:
            weights = raw_w / raw_w_sum

        # 叠乘到比例化的相似度上
        Sh[i, Ni] = weights * (S[i, Ni] / denom_sim)

    return Sh, knn_sets


def mutual_weight_matrix(knn_sets: list[set], k: int) -> np.ndarray:
    """
    构造的权重矩阵 W：
      - 若 i∈Nj 且 j∈Ni ：W_ij = 1
      - 若 i∉Nj 且 j∉Ni：W_ij = 0
      - 否则：W_ij = 0.5
    对角线设为 1（自身与自身）。
    """
    n = len(knn_sets)
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        W[i, i] = 1.0
        for j in range(i + 1, n):
            i_in_Nj = i in knn_sets[j]
            j_in_Ni = j in knn_sets[i]
            if i_in_Nj and j_in_Ni:
                wij = 1.0
            elif (not i_in_Nj) and (not j_in_Ni):
                wij = 0.0
            else:
                wij = 0.5
            W[i, j] = W[j, i] = wij
    return W


def symmetrize(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)

# ---------- SMF：多源相似度融合 ----------

def smf_multisource_fusion(S_list: list[np.ndarray],
                           k: int,
                           decay: float = 0.85,
                           alpha: float = 0.7,
                           max_iter: int = 50,
                           tol: float = 1e-7) -> tuple[np.ndarray, list[set], np.ndarray]:
    """
    输入多源相似度 S_list（长度=3），输出融合后的 Sm（式(4)），以及：
      - 对应的 KNN 集合（用于后续式(7)）
      - 多源平均初始化 P0（调试/复用）
    流程：
      1) 归一化得到 Ph0，并基于 S_h 计算 Sh（WKNN）
      2) 迭代更新 P_h^t
      3) 取平均并对称化
    """
    assert len(S_list) == 3, "当前实现3个相似度来源（可按需扩展）"
    H = len(S_list)
    n = S_list[0].shape[0]

    # 初始：归一化 Ph0；WKNN 得 Sh 与 KNN 集合
    P0 = []
    S_neighbor = []
    knn_sets = None  # 取第一个源的KNN作为集合定义，也可以合并多源KNN（此处保持简单）
    for h, S in enumerate(S_list):
        Ph0_h = new_normalization(S)
        P0.append(Ph0_h)

    # WKNN（每个源都算Sh）
    Sh_list = []
    for h, S in enumerate(S_list):
        Sh_h, knn_h = weighted_knn_matrix(S, k=k, decay=decay)
        Sh_list.append(Sh_h)
        # 记录第一源的KNN集合作为后续互为邻居的参考；也可以用并集/交集策略
        if h == 0:
            knn_sets = knn_h

    # 迭代
    P_curr = [P0[h].copy() for h in range(H)]
    avg_P0_others = []
    for h in range(H):
        others = [P0[r] for r in range(H) if r != h]
        avg_P0_others.append(sum(others) / (H - 1))

    it = 0
    while it < max_iter:
        it += 1
        P_next = []
        for h in range(H):
            others_curr = [P_curr[r] for r in range(H) if r != h]
            avg_P_others = sum(others_curr) / (H - 1)
            term_prop = Sh_list[h] @ avg_P_others @ Sh_list[h].T
            P_h_next = alpha * term_prop + (1.0 - alpha) * avg_P0_others[h]
            # 归一化保持数值稳定
            P_h_next = new_normalization(P_h_next)
            P_next.append(P_h_next)

        delta = np.linalg.norm(sum(P_next)/H - sum(P_curr)/H) / (np.linalg.norm(sum(P_curr)/H) + 1e-12)
        P_curr = P_next
        if delta < tol:
            break

    # 融合并对称
    Sm = symmetrize(sum(P_curr) / H)
    return Sm, knn_sets, sum(P0) / H


# ---------- SMF：跨模态拓扑融合 ----------

def cross_modal_refine(Sm: np.ndarray, Sd: np.ndarray, A: np.ndarray,
                       beta: float = 0.3,
                       max_iter: int = 30,
                       tol: float = 1e-7) -> tuple[np.ndarray, np.ndarray]:
    """
    用 A 强化 Sm 与 Sd（式(5)(6)）。
    默认做固定步或收敛判断。
    """
    m, n = A.shape
    AA_T = A @ A.T   # m×m
    A_TA = A.T @ A   # n×n

    Sm_curr = Sm.copy()
    Sd_curr = Sd.copy()

    it = 0
    while it < max_iter:
        it += 1
        Sm_next = beta * (AA_T @ Sm_curr) + (1.0 - beta) * Sm_curr
        Sd_next = beta * (Sd_curr @ A_TA) + (1.0 - beta) * Sd_curr

        # 对称化与轻度归一
        Sm_next = symmetrize(Sm_next)
        Sd_next = symmetrize(Sd_next)

        d1 = np.linalg.norm(Sm_next - Sm_curr) / (np.linalg.norm(Sm_curr) + 1e-12)
        d2 = np.linalg.norm(Sd_next - Sd_curr) / (np.linalg.norm(Sd_curr) + 1e-12)
        Sm_curr, Sd_curr = Sm_next, Sd_next
        if max(d1, d2) < tol:
            break

    return Sm_curr, Sd_curr


# ---------- 顶层：读取数据并给出最终 M、D ----------

def get_smf_syn_sim(k_microbe: int,
                    k_disease: int,
                    alpha: float = 0.7,
                    beta: float = 0.3,
                    decay: float = 0.85,
                    max_iter_source: int = 50,
                    max_iter_cross: int = 30,
                    tol: float = 1e-7,
                    paths: dict | None = None):
    """
    计算 SMF 融合后的微生物/疾病相似度：
    - k_microbe, k_disease：KNN 邻居数
    - alpha：式(3) 融合权重
    - beta：式(5)(6) 跨模态强化系数
    - decay：WKNN 衰减（0~1）
    - paths：文件路径字典
        {
          "microbe": ["fun_MS.txt", "cos_MS.txt", "Ga_MS.txt"],
          "disease": ["sem_DS.txt", "cos_DS.txt", "Ga_DS.txt"],
          "assoc":   "A.txt"
        }
    返回：M_final, D_final
    """
    if paths is None:
        paths = {
            "microbe": ["fun_MS.txt", "cos_MS.txt", "Ga_MS.txt"],
            "disease": ["sem_DS.txt", "cos_DS.txt", "Ga_DS.txt"],
            "assoc":   "A.txt",
        }

    # 读取相似度与关联矩阵
    microbe_list = [np.loadtxt(p) for p in paths["microbe"]]
    disease_list = [np.loadtxt(p) for p in paths["disease"]]
    A = np.loadtxt(paths["assoc"])

    # 多源融合
    Sm0, knn_m, _ = smf_multisource_fusion(microbe_list, k=k_microbe,
                                           decay=decay, alpha=alpha,
                                           max_iter=max_iter_source, tol=tol)
    Sd0, knn_d, _ = smf_multisource_fusion(disease_list, k=k_disease,
                                           decay=decay, alpha=alpha,
                                           max_iter=max_iter_source, tol=tol)

    # 跨模态拓扑融合
    Sm_ref, Sd_ref = cross_modal_refine(Sm0, Sd0, A, beta=beta,
                                        max_iter=max_iter_cross, tol=tol)

    # 互为邻居权重矩阵
    Wm = mutual_weight_matrix(knn_m, k_microbe)
    Wd = mutual_weight_matrix(knn_d, k_disease)

    # 最终
    M_final = Wm * Sm_ref
    D_final = Wd * Sd_ref

    # 对称化+微幅数值清理
    M_final = symmetrize(M_final)
    D_final = symmetrize(D_final)

    return M_final, D_final


# ---------- 可执行入口 ----------

if __name__ == "__main__":
    total_start = time.time()

    # 按需改这些超参数
    k1 = 1177     # 微生物 KNN
    k2 = 134       # 疾病 KNN
    alpha = 0.7   # 权重
    beta = 0.3    # 迭代系数
    decay = 0.85  # WKNN 衰减（0~1）
    max_iter_source = 50
    max_iter_cross = 30
    tol = 1e-7

    sim_m, sim_d = get_smf_syn_sim(k_microbe=k1,
                                   k_disease=k2,
                                   alpha=alpha,
                                   beta=beta,
                                   decay=decay,
                                   max_iter_source=max_iter_source,
                                   max_iter_cross=max_iter_cross,
                                   tol=tol,
                                   paths=None)

    np.savetxt("m_sim_final_SMF.txt", sim_m)
    np.savetxt("d_sim_final_SMF.txt", sim_d)

    total_end = time.time()
    print(f"Total SMF fusion time: {total_end - total_start:.2f} seconds")
