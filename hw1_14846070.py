import numpy as np
import numpy.linalg as la

def gram_schmidt(S1: np.ndarray):
    m, n = S1.shape
    S2 = np.zeros((m, n), dtype=float)

    for j in range(n):
        v = S1[:, j]
        coeff = S2[:, :j].T @ v        # (e_i^T v_j)
        u = v - S2[:, :j] @ coeff     # Σ (e_i^T v_j) e_i

        norm_u = la.norm(u)
        if norm_u < 1e-12:
            raise ValueError(f"Vector {j} is linearly dependent.")

        S2[:, j] = u / norm_u

    return S2


# ==========================
# 定義輸入矩陣 S1（一定要有）
# ==========================
S1 = np.array([
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1]
], dtype=float)

# 執行 Gram–Schmidt
S2 = gram_schmidt(S1)

# 驗證
np.set_printoptions(precision=2, suppress=True)
print("S2^T @ S2 =")
print(S2.T @ S2)
