import numpy as np
import time


def clip01(M: np.ndarray) -> np.ndarray:
    """Clip matrix to [0, 1]."""
    return np.clip(M, 0.0, 1.0)


def relu(M: np.ndarray) -> np.ndarray:
    """Elementwise ReLU."""
    return np.maximum(M, 0.0)


def mat_power_clip01(Z: np.ndarray, power: int) -> np.ndarray:
    """
    Compute (Z)^power with clipping to [0,1] after each multiplication
    to avoid numeric blow-up; power >= 1.
    """
    assert power >= 1
    R = Z.copy()
    for _ in range(1, power):
        R = clip01(R @ Z)
    return R


def build_connection_matrix(A: np.ndarray) -> np.ndarray:
    """
    Z = min(1, A^T A). A: m x n -> Z: n x n (disease-disease).
    """
    Z = A.T @ A
    Z = clip01(Z)
    return Z


def hwp_construct(
    A: np.ndarray,
    q: int = 3,
    omega: float = 0.5,
    tau: float = 0.1,
    seed: int | None = 42,
) -> np.ndarray:
    """
    Higher-Order Weighted Perturbation (HWP).

    Parameters
    ----------
    A : np.ndarray
        Known microbe-disease association matrix (m x n), values in {0,1} or [0,1].
    q : int
        Maximum order (>=2). For i=2..q we build Assoconnection(A,i).
    omega : float
        Base perturbation weight in (0,1). Actual weight at order i is omega^(i-2).
    tau : float
        Threshold to suppress noise.
    seed : int|None
        Random seed for reproducibility.

    Returns
    -------
    X_thr : np.ndarray
        Higher-order association matrix after aggregation  and thresholding.
    """
    assert q >= 2, "q 必须 ≥ 2"
    assert 0.0 < omega < 1.0, "omega 应在 (0,1) 内"
    assert 0.0 <= tau <= 1.0, "tau 应在 [0,1] 内"

    m, n = A.shape
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Eq.(9) initial connection matrix on disease side
    Z = build_connection_matrix(A)  # n x n

    # Precompute powers Z^(i-1), i=2..q  (i-1 >= 1)
    Z_powers = {1: Z}
    for p in range(2, q):  # need up to power (q-1)
        Z_powers[p] = mat_power_clip01(Z_powers[p - 1], 1)  # multiply by Z once (with clipping)

    # Aggregate higher-order weighted connections
    agg = np.zeros((m, n), dtype=float)

    for i in range(2, q + 1):
        # Z^(i-1)
        Zi_1 = Z_powers[i - 1] if (i - 1) in Z_powers else mat_power_clip01(Z, i - 1)

        # Candidate higher-order signal: A @ Z^(i-1)
        AZ = A @ Zi_1  # m x n

        # Only keep newly suggested or reinforced positions over direct A
        Delta = relu(AZ - A)  # m x n, non-negative

        if np.count_nonzero(Delta) == 0:
            continue

        # Order-aware random weights: W_i = omega^(i-2) * U(0,1) on the same shape
        Wi = (omega ** (i - 2)) * rng.random(size=(m, n))

        # Apply only where Delta > 0
        mask = (Delta > 1e-12).astype(float)

        # Assoconnection(A, i) = min(1, Delta ⊙ Wi)
        Asso_i = clip01(Delta * Wi * mask)

        agg += Asso_i

    #  X = A + min(1, sum_i Assoconnection(A,i))
    X = A + clip01(agg)
    X = clip01(X)

    #  thresholding
    X_thr = np.where(X >= tau, X, 0.0)

    return X_thr


def run_hwp_from_files(
    A_path: str = "MD_A.csv",
    out_path: str = "X_HWP.txt",
    q: int = 3,
    omega: float = 0.5,
    tau: float = 0.1,
    seed: int | None = 42,
):
    """
    I/O 包装：读入 A，运行 HWP，输出 X。
    """
    A = np.loadtxt(A_path)
    X = hwp_construct(A, q=q, omega=omega, tau=tau, seed=seed)
    np.savetxt(out_path, X)
    return X


if __name__ == "__main__":
    t0 = time.time()

    # 你可按需修改的超参数
    q = 3          # 最大阶数（>=2）
    omega = 0.01    # 基础扰动系数 (0,1)
    tau = 0.15     # 阈值抑噪
    seed = 42

    # 文件名沿用你之前的风格：A.txt 位于当前目录
    X = run_hwp_from_files(
        A_path="MD_A.csv",
        out_path="X_HWP.txt",
        q=q,
        omega=omega,
        tau=tau,
        seed=seed,
    )

    t1 = time.time()
    print(f"HWP finished. Shape={X.shape}, density={(X>0).mean():.4f}, time={t1 - t0:.2f}s")
