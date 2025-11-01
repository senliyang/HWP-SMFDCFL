# dpmf.py
import numpy as np
import time
from typing import List, Tuple


EPS = 1e-10


# ------------------------- 工具函数 -------------------------

def clip_nonneg(M: np.ndarray) -> np.ndarray:
    """投影到非负正交锥，避免出现 -0."""
    M[M < 0] = 0.0
    return M


def l21_row_grad(H: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    计算 L_{2,1}(alpha * H) 的梯度项：
    beta * alpha^2 * H_{ij} / ( ||alpha * H_{i:}||_2 + EPS )
    返回与 H 同形状的矩阵（逐元素）。
    """
    row_norm = np.linalg.norm(alpha * H, axis=1, keepdims=True)  # (r,1)
    row_norm = np.maximum(row_norm, EPS)
    G = beta * (alpha ** 2) * (H / row_norm)
    return G


def chain_product_left(W_list: List[np.ndarray], upto: int) -> np.ndarray:
    """
    计算 G_{upto} = W1 W2 ... W_upto
    若 upto == 0，返回单位阵（根据形状推断）。
    """
    if upto == 0:
        # identity with rows = rows of W1 and cols = rows of W1
        m = W_list[0].shape[0]
        return np.eye(m)
    G = W_list[0]
    for i in range(1, upto):
        G = G @ W_list[i]
    return G


def chain_product_right(W_list: List[np.ndarray], start: int, H: np.ndarray) -> np.ndarray:
    """
    计算 F_{start} = W_{start+1} ... W_l H
    若 start == len(W_list)，返回 H
    """
    F = H
    for i in range(len(W_list) - 1, start - 1, -1):
        F = W_list[i] @ F
    return F


def stack_product(W_list: List[np.ndarray], H: np.ndarray) -> np.ndarray:
    """计算 W1 W2 ... Wl H"""
    W = chain_product_left(W_list, len(W_list))
    return W @ H


# ------------------------- DNMF 主体 -------------------------

def dnmf(
    A: np.ndarray,
    X: np.ndarray,
    k: int = 64,
    l: int = 3,
    lambda_s: float = 1.0,
    alpha: float = 1.0,
    beta: float = 1e-2,
    max_iter: int = 300,
    seed: int = 42,
    init_scale: float = 1.0,
    ranks: List[int] | None = None,
    verbose: bool = True,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    深层非负矩阵分解（DNMF）：
      目标：min ||A - W*H||_F^2 + lambda_s * ||X - W*H||_F^2 + beta * ||alpha * H||_{2,1}
      其中 W = W1 W2 ... Wl, H = H_l
    采用分层乘法更新：
      对每层 Wi：Wi <- Wi * (G_{i-1}^T Y F_i^T) / (G_{i-1}^T G_{i-1} Wi F_i F_i^T + EPS)
      对 H：   H  <- H  * (W^T Y) / (W^T W H + beta * l21_grad + EPS)

    返回：
      W_list: [W1,...,Wl]
      H: 最深层系数矩阵
      A_rec: 重构矩阵 W*H
    """
    assert A.shape == X.shape, "A 与 X 的尺寸需一致"
    m, n = A.shape

    rng = np.random.default_rng(seed)

    # 目标融合：以 Y 为联合重构目标（数值更稳）
    Y = (A + lambda_s * X) / (1.0 + lambda_s)

    # 层秩设置
    if ranks is None:
        ranks = [k] * l
    assert len(ranks) == l
    assert all(r > 0 for r in ranks)

    # 参数初始化（非负）
    W_list = []
    r_prev = m
    for r in ranks:
        Wi = clip_nonneg(rng.random((r_prev, r)) * init_scale + 0.1)
        W_list.append(Wi)
        r_prev = r
    H = clip_nonneg(rng.random((r_prev, n)) * init_scale + 0.1)

    # 预分配
    for it in range(1, max_iter + 1):
        # 先更新 H（使用整体 W = W1...Wl）
        W_all = chain_product_left(W_list, len(W_list))  # m x r_l
        WH = W_all @ H

        # H 的乘法更新（含 L21）
        num_H = W_all.T @ Y                      # r_l x n
        den_H = (W_all.T @ W_all) @ H           # r_l x n
        # L21 行稀疏正则
        G_l21 = l21_row_grad(H, alpha=alpha, beta=beta)  # r_l x n
        den_H = den_H + G_l21 + EPS

        H *= num_H / den_H
        H = clip_nonneg(H)

        # 逐层更新 W_i（自左至右或自右至左都可，这里自左至右）
        # 对每层：min ||Y - (G_{i-1} Wi F_i)||_F^2 的标准 NMF 型更新
        for i in range(len(W_list)):
            # G_{i-1}: W1...W_{i-1}
            G_left = chain_product_left(W_list, i)  # m x r_{i-1} (i=0时为 I_m)
            # F_i: W_{i+1}...W_l H
            F_right = chain_product_right(W_list, i + 1, H)  # r_i x n

            Wi = W_list[i]
            # 乘法更新
            num_Wi = G_left.T @ Y @ F_right.T                           # r_{i-1} x r_i
            den_Wi = (G_left.T @ G_left) @ Wi @ (F_right @ F_right.T)   # r_{i-1} x r_i
            den_Wi = den_Wi + EPS

            Wi *= num_Wi / den_Wi
            W_list[i] = clip_nonneg(Wi)

        if verbose and (it % 50 == 0 or it == 1 or it == max_iter):
            W_all = chain_product_left(W_list, len(W_list))
            WH = W_all @ H
            loss_rec = np.linalg.norm(A - WH, "fro") ** 2
            loss_rec2 = np.linalg.norm(X - WH, "fro") ** 2
            # L21
            l21 = np.sum(np.linalg.norm(alpha * H, axis=1))
            obj = loss_rec + lambda_s * loss_rec2 + beta * l21
            print(f"[DNMF] iter={it:4d}  "
                  f"||A-WH||^2={loss_rec:.4e}  ||X-WH||^2={loss_rec2:.4e}  "
                  f"L21={l21:.4e}  Obj={obj:.4e}")

    # 最终重构
    W_all = chain_product_left(W_list, len(W_list))
    A_rec = W_all @ H
    return W_list, H, A_rec


# ------------------------- 加权 NMF（二阶段特征） -------------------------

def weighted_nmf(
    M: np.ndarray,
    k: int,
    Q: np.ndarray | None = None,
    theta1: float = 1e-2,
    theta2: float = 1e-2,
    max_iter: int = 1000,
    seed: int = 42,
    init_scale: float = 1.0,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    求解 min_{U,V>=0} || Q ⊙ (M - U V) ||_F^2 + theta1 ||U||_F^2 + theta2 ||V||_F^2
    乘法更新（式(17)(18)）：
      U <- U ⊙ ( (Q⊙M) V^T ) / ( (Q⊙(UV)) V^T + theta1 U + EPS )
      V <- V ⊙ ( U^T (Q⊙M) ) / ( U^T (Q⊙(UV)) + theta2 V + EPS )
    """
    m, n = M.shape
    rng = np.random.default_rng(seed)

    U = clip_nonneg(rng.random((m, k)) * init_scale + 0.1)
    V = clip_nonneg(rng.random((k, n)) * init_scale + 0.1)

    if Q is None:
        # 与文中一致，初值取与 A 相同；若 M 来自 A_rec，可用 (M>0) 或 np.clip(M,0,1)
        Q = np.clip(M, 0.0, 1.0)

    for it in range(1, max_iter + 1):
        UV = U @ V
        Q_res = Q * (M - UV)

        # U 更新
        num_U = (Q * M) @ V.T                   # m x k
        den_U = (Q * UV) @ V.T + theta1 * U + EPS
        U *= num_U / den_U
        U = clip_nonneg(U)

        # V 更新
        UV = U @ V
        num_V = U.T @ (Q * M)                   # k x n
        den_V = U.T @ (Q * UV) + theta2 * V + EPS
        V *= num_V / den_V
        V = clip_nonneg(V)

        if verbose and (it % 100 == 0 or it == 1 or it == max_iter):
            UV = U @ V
            wloss = np.linalg.norm(Q * (M - UV), "fro") ** 2
            reg = theta1 * np.linalg.norm(U, "fro") ** 2 + theta2 * np.linalg.norm(V, "fro") ** 2
            print(f"[NMF]   iter={it:4d}  weighted loss={wloss:.4e}  reg={reg:.4e}  obj={(wloss+reg):.4e}")

    return U, V


# ------------------------- 顶层封装：DPMF -------------------------

def dpmf_pipeline(
    A: np.ndarray,
    X: np.ndarray,
    k: int = 64,
    dnmf_iter: int = 300,
    nmf_iter: int = 1000,
    lambda_s: float = 1.0,
    alpha: float = 1.0,
    beta: float = 1e-2,
    theta1: float = 1e-2,
    theta2: float = 1e-2,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    双路径矩阵分解总流程：
      1) DNMF: (A, X) -> A_rec
      2) NMF : A_rec -> U (微生物特征 F_M), V (疾病特征 F_D)
    返回：
      A_rec, F_M (U), F_D (V)
    """
    # 1) DNMF
    W_list, H, A_rec = dnmf(
        A=A, X=X, k=k, l=3, lambda_s=lambda_s, alpha=alpha, beta=beta,
        max_iter=dnmf_iter, seed=seed, verbose=verbose
    )

    # 2) NMF
    # Q 初值设为与 A 相同（数值裁剪到[0,1]亦可）
    Q_init = np.clip(A, 0.0, 1.0)
    U, V = weighted_nmf(
        M=A_rec, k=k, Q=Q_init, theta1=theta1, theta2=theta2,
        max_iter=nmf_iter, seed=seed, verbose=verbose
    )
    return A_rec, U, V


# ------------------------- 文件 I/O 入口 -------------------------

def run_dpmf_from_files(
    A_path: str = "MD_A.csv",
    X_path: str = "X_HWP.txt",
    out_Arec_path: str = "A_rec_DNMF.txt",
    out_U_path: str = "FM1_U.txt",
    out_V_path: str = "FD1_V.txt",
    k: int = 64,
    dnmf_iter: int = 300,
    nmf_iter: int = 1000,
    lambda_s: float = 1.0,
    alpha: float = 1.0,
    beta: float = 1e-2,
    theta1: float = 1e-2,
    theta2: float = 1e-1,
    seed: int = 42,
    verbose: bool = True,
):
    A = np.loadtxt(A_path)
    X = np.loadtxt(X_path)

    A_rec, U, V = dpmf_pipeline(
        A=A, X=X, k=k,
        dnmf_iter=dnmf_iter, nmf_iter=nmf_iter,
        lambda_s=lambda_s, alpha=alpha, beta=beta,
        theta1=theta1, theta2=theta2,
        seed=seed, verbose=verbose
    )

    np.savetxt(out_Arec_path, A_rec)
    np.savetxt(out_U_path, U)
    np.savetxt(out_V_path, V)

    return A_rec, U, V


if __name__ == "__main__":
    t0 = time.time()

    # 要按需调参，按需调参
    k = 64
    dnmf_iter = 300
    nmf_iter = 1000
    lambda_s = 1.0
    alpha = 1.0
    beta = 1e-2
    theta1 = 1e-2
    theta2 = 1e-1
    seed = 42
    verbose = True

    A_rec, U, V = run_dpmf_from_files(
        A_path="MD_A.csv",
        X_path="X_HWP.txt",
        out_Arec_path="A_rec_DNMF.txt",
        out_U_path="FM1_U.txt",
        out_V_path="FD1_V.txt",
        k=k,
        dnmf_iter=dnmf_iter,
        nmf_iter=nmf_iter,
        lambda_s=lambda_s,
        alpha=alpha,
        beta=beta,
        theta1=theta1,
        theta2=theta2,
        seed=seed,
        verbose=verbose,
    )

    t1 = time.time()
    print(f"DPMF finished. A_rec shape={A_rec.shape}, U={U.shape}, V={V.shape}, time={t1 - t0:.2f}s")
