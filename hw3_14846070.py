import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# --------------------------------------------------
# 參數設定（對應題目）
pts = 50          # number of samples
n = 5             # Fourier order
l, r = -2, 2      # interval [l, r]

# --------------------------------------------------
# 建立 x 與 y（題目給的測試資料）
x = np.linspace(l, r, pts)
y = np.zeros_like(x)

# square wave (題目指定)
pts2 = pts // 2
y[:pts2] = -1
y[pts2:] = 1

# --------------------------------------------------
# 週期與角頻率
T0 = r - l
f0 = 1.0 / T0
omega0 = 2.0 * np.pi * f0

# --------------------------------------------------
# step1: generate design matrix X = φ(x)
# X = [1 cos(w0x) ... cos(nw0x) sin(w0x) ... sin(nw0x)]
X = np.ones((pts, 2 * n + 1))

for k in range(1, n + 1):
    X[:, k] = np.cos(k * omega0 * x)
    X[:, n + k] = np.sin(k * omega0 * x)

# --------------------------------------------------
# step2: SVD of X
U, S, VT = la.svd(X, full_matrices=False)

# --------------------------------------------------
# step3: solve least squares using short SVD
# a = V Σ^{-1} U^T y
a = VT.T @ (np.diag(1.0 / S) @ (U.T @ y))

# --------------------------------------------------
# step4: reconstructed curve
y_bar = X @ a

# --------------------------------------------------
# Plot result
plt.figure(figsize=(6, 4))
plt.plot(x, y, 'b.', label='True data')
plt.plot(x, y_bar, 'g-', label='Fitted curve')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fourier Least Squares Approximation (SVD)')
plt.legend()
plt.grid(True)
plt.show()
