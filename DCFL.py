# dcfl.py
import numpy as np
from typing import Tuple, Optional


EPS = 1e-12


# --------------------- 基础工具 ---------------------

def relu(X: np.ndarray) -> np.ndarray:
    return np.maximum(X, 0.0)

def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, EPS)
    return X / n

def build_hypergraph_incidence_from_similarity(S: np.ndarray, include_self: bool = True) -> np.ndarray:
    """
    基于相似度矩阵 S 构建超图的关联矩阵 A（n x n）：
      - 先建图 G：S[i,j] > 0 视为邻接；
      - 第 j 列超边包含节点 j 的邻居（以及自身可选）。
    A[i,j] = 1 若 i 属于第 j 条超边，否则 0。
    """
    n = S.shape[0]
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        neigh = np.where(S[j, :] > 0)[0].tolist()
        if include_self and j not in neigh:
            neigh.append(j)
        A[neigh, j] = 1.0
    return A

def compute_hg_propagator(A: np.ndarray, W: Optional[np.ndarray] = None) -> np.ndarray:
    """
    计算超图卷积传播算子：S = D^{-1/2} A W B^{-1} A^T D^{-1/2}
    - A:  n x m（n 个点，m 条超边）
    - W:  m x m 超边权重，默认 I
    - B:  超边度对角阵，B_ee = sum_v A[v,e]
    - D:  节点度对角阵，D_vv = sum_e A[v,e] * W[ee]
    返回：S (n x n)，对称半正定。
    """
    n, m = A.shape
    if W is None:
        W = np.eye(m)

    # 超边度
    B_diag = np.sum(A, axis=0)  # m
    B_inv = np.diag(1.0 / np.maximum(B_diag, EPS))

    # 节点度
    # D = diag(A W 1)
    ones_m = np.ones((m, 1))
    D_diag = (A @ (W @ ones_m)).reshape(-1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(D_diag, EPS)))

    S = D_inv_sqrt @ A @ W @ B_inv @ A.T @ D_inv_sqrt
    return S


# --------------------- HGCN 两层 ---------------------

def hgcn_forward(F_in: np.ndarray,
                 S: np.ndarray,
                 dims: Tuple[int, int],
                 seed: int = 42) -> np.ndarray:
    """
    两层 HGCN：H0=F_in
      H1 = ReLU( S H0 P0 )
      H2 = ReLU( S H1 P1 )
    其中 P0: d_in -> d_hidden, P1: d_hidden -> d_out
    为无监督特征提取，P 可随机初始化；若需要可外部训练。
    """
    n, d_in = F_in.shape
    d_hid, d_out = dims
    rng = np.random.default_rng(seed)

    # He 初始化
    P0 = rng.standard_normal((d_in, d_hid)) * np.sqrt(2.0 / max(1, d_in))
    P1 = rng.standard_normal((d_hid, d_out)) * np.sqrt(2.0 / max(1, d_hid))

    H0 = F_in
    H1 = relu(S @ H0 @ P0)
    H2 = relu(S @ H1 @ P1)
    return H2


# --------------------- MLP 分支（可选对齐训练） ---------------------

class SimpleMLP:
    """
    两层/三层 MLP（默认两层），支持用 HGCN 输出做目标的无监督对齐训练（MSE）。
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        if hidden_dim is None:
            hidden_dim = max(32, min(256, in_dim // 2))

        # He 初始化
        self.W1 = self.rng.standard_normal((in_dim, hidden_dim)) * np.sqrt(2.0 / max(1, in_dim))
        self.b1 = np.zeros((hidden_dim,))
        self.W2 = self.rng.standard_normal((hidden_dim, out_dim)) * np.sqrt(2.0 / max(1, hidden_dim))
        self.b2 = np.zeros((out_dim,))

    def forward(self, X: np.ndarray) -> np.ndarray:
        Z1 = X @ self.W1 + self.b1
        A1 = relu(Z1)
        Z2 = A1 @ self.W2 + self.b2
        A2 = relu(Z2)
        return A2

    def fit_to_target(self, X: np.ndarray, Y: np.ndarray,
                      lr: float = 1e-3, epochs: int = 200, batch_size: int = 256, verbose: bool = True):
        """
        用 MSE 让 MLP(X) ≈ Y。纯 numpy 版小批量 SGD。
        """
        n = X.shape[0]
        for ep in range(1, epochs + 1):
            # 随机打乱
            idx = self.rng.permutation(n)
            Xs = X[idx]
            Ys = Y[idx]

            for start in range(0, n, batch_size):
                end = min(n, start + batch_size)
                xb = Xs[start:end]
                yb = Ys[start:end]

                # 前向
                z1 = xb @ self.W1 + self.b1
                a1 = relu(z1)
                z2 = a1 @ self.W2 + self.b2
                a2 = relu(z2)

                # 损失与梯度 (MSE)
                diff = (a2 - yb)
                # d(ReLU)
                da2 = diff * (1.0 / (end - start))
                dz2 = da2 * (z2 > 0)

                gW2 = a1.T @ dz2
                gb2 = dz2.sum(axis=0)

                da1 = dz2 @ self.W2.T
                dz1 = da1 * (z1 > 0)

                gW1 = xb.T @ dz1
                gb1 = dz1.sum(axis=0)

                # SGD
                self.W2 -= lr * gW2
                self.b2 -= lr * gb2
                self.W1 -= lr * gW1
                self.b1 -= lr * gb1

            if verbose and (ep % 50 == 0 or ep == 1 or ep == epochs):
                pred = self.forward(X)
                mse = np.mean((pred - Y) ** 2)
                print(f"[MLP] epoch={ep:4d}  MSE={mse:.6f}")


# --------------------- DCFL 顶层 ---------------------

def dcfl_features(
    F_in: np.ndarray,
    S: np.ndarray,
    hgc_hidden: int,
    out_dim: Optional[int] = None,
    mlp_hidden: Optional[int] = None,
    align_epochs: int = 200,
    align_lr: float = 1e-3,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    输入：
      - F_in : 初始特征（如 FHM = concat(MFS, MCS, MGS)），shape (n, d_in)
      - S    : 超图传播算子（由相似度 -> A -> S 计算）
    输出：
      - FM2  : 融合后的非线性特征  (n, out_dim)
      - F_hg : HGCN 分支输出       (n, out_dim)
      - F_mlp: MLP  分支输出       (n, out_dim)
    """
    n, d_in = F_in.shape
    if out_dim is None:
        out_dim = d_in  # 与文中一致，H^l 与 FHM 维度一致更易融合

    # HGCN 两层
    F_hg = hgcn_forward(F_in, S, dims=(hgc_hidden, out_dim), seed=seed)
    F_hg = l2_normalize_rows(F_hg)

    # MLP 分支 +（可选）对齐训练
    mlp = SimpleMLP(in_dim=d_in, out_dim=out_dim, hidden_dim=mlp_hidden, seed=seed)
    # 用 HGCN 输出作为软目标，让两通道对齐，提升鲁棒性
    if align_epochs > 0:
        mlp.fit_to_target(F_in, F_hg, lr=align_lr, epochs=align_epochs, verbose=verbose)
    F_mlp = mlp.forward(F_in)
    F_mlp = l2_normalize_rows(F_mlp)

    # 特征融合：F = F_hg + (F_mlp)^2
    FM2 = F_hg + np.square(F_mlp)
    FM2 = l2_normalize_rows(FM2)
    return FM2, F_hg, F_mlp


# --------------------- 便捷封装：微生物 / 疾病 ---------------------

def dcfl_from_similarity_and_features(
    S_sim: np.ndarray,           # 相似度矩阵（微生物或疾病）
    F_concat: np.ndarray,        # 初始特征（FHM 或 FHD）
    use_edge_weight: bool = False,
    hgc_hidden: int = 128,
    out_dim: Optional[int] = None,
    mlp_hidden: Optional[int] = None,
    align_epochs: int = 200,
    align_lr: float = 1e-3,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    1) 由相似度 -> 关联矩阵 A -> 传播算子 S
    2) 跑 DCFL，得到最终非线性特征
    """
    A = build_hypergraph_incidence_from_similarity(S_sim, include_self=True)  # (n x m)
    if use_edge_weight:
        # 可选：以每条超边（每个中心节点 j 的邻居）在 S 中的相似度均值作为权重
        m = A.shape[1]
        W = np.eye(m)
        for j in range(m):
            nodes = np.where(A[:, j] > 0)[0]
            if len(nodes) > 1:
                # 以中心节点 j 对应的相似度均值为权
                wj = np.mean(S_sim[j, nodes])
                W[j, j] = max(wj, EPS)
        S_prop = compute_hg_propagator(A, W=W)
    else:
        S_prop = compute_hg_propagator(A, W=None)

    FM2, F_hg, F_mlp = dcfl_features(
        F_in=F_concat, S=S_prop, hgc_hidden=hgc_hidden, out_dim=out_dim,
        mlp_hidden=mlp_hidden, align_epochs=align_epochs, align_lr=align_lr,
        seed=seed, verbose=verbose
    )
    return FM2, F_hg, F_mlp


# --------------------- 示例入口 ---------------------

def demo_microbe_and_disease(
    microbe_sim_path: str = "m_sim_final_SMF.txt",
    disease_sim_path: str = "d_sim_final_SMF.txt",
    FHM_path: str = "FHM.npy",   # 形如 concat(MFS, MCS, MGS) 的 (nm, 2*nm) 或任意 d
    FHD_path: str = "FHD.npy",
    out_microbe_path: str = "FM2_microbe.npy",
    out_disease_path: str = "FD2_disease.npy",
    hgc_hidden: int = 128,
    out_dim: Optional[int] = None,
    align_epochs: int = 200,
    align_lr: float = 1e-3,
    seed: int = 42,
    verbose: bool = True,
):
    Sm = np.loadtxt(microbe_sim_path)
    Sd = np.loadtxt(disease_sim_path)
    FHM = np.load(FHM_path)  # shape (nm, d_m)
    FHD = np.load(FHD_path)  # shape (nd, d_d)

    FM2, _, _ = dcfl_from_similarity_and_features(
        S_sim=Sm, F_concat=FHM, use_edge_weight=False,
        hgc_hidden=hgc_hidden, out_dim=out_dim,
        align_epochs=align_epochs, align_lr=align_lr,
        seed=seed, verbose=verbose
    )
    FD2, _, _ = dcfl_from_similarity_and_features(
        S_sim=Sd, F_concat=FHD, use_edge_weight=False,
        hgc_hidden=hgc_hidden, out_dim=out_dim,
        align_epochs=align_epochs, align_lr=align_lr,
        seed=seed, verbose=verbose
    )

    np.save(out_microbe_path, FM2)
    np.save(out_disease_path, FD2)
    print(f"[DCFL] Done. FM2: {FM2.shape}, FD2: {FD2.shape}")


if __name__ == "__main__":
    # 仅示例：确保同目录下存在 m_sim_final_SMF.txt / d_sim_final_SMF.txt / FHM.npy / FHD.npy
    demo_microbe_and_disease()
