import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 144
np.random.seed(0)

# --------------------------------------------------
# 1. Generate data (given)
# --------------------------------------------------
mean1 = np.array([0, 5])
sigma1 = np.array([[0.3, 0.2],
                   [0.2, 1]])
N1 = 200
X1 = np.random.multivariate_normal(mean1, sigma1, N1)

mean2 = np.array([3, 4])
sigma2 = np.array([[0.3, 0.2],
                   [0.2, 1]])
N2 = 100
X2 = np.random.multivariate_normal(mean2, sigma2, N2)

# --------------------------------------------------
# 2. LDA projection vector
# --------------------------------------------------
mu1 = X1.mean(axis=0)
mu2 = X2.mean(axis=0)

S1 = np.cov(X1, rowvar=False)
S2 = np.cov(X2, rowvar=False)
Sw = S1 + S2

w = la.inv(Sw) @ (mu1 - mu2)
w = w / la.norm(w)   # normalize for plotting

# --------------------------------------------------
# 3. Project data
# --------------------------------------------------
y1 = X1 @ w
y2 = X2 @ w

# --------------------------------------------------
# 4. Plot original data
# --------------------------------------------------
plt.figure(figsize=(5, 5))
plt.scatter(X1[:, 0], X1[:, 1], c='red', s=8)
plt.scatter(X2[:, 0], X2[:, 1], c='green', s=8)

# --------------------------------------------------
# 5. Draw LDA projection segments (like sample figure)
# --------------------------------------------------
# baseline center (move downward to avoid overlap)
center = (mu1 + mu2) / 2
offset = np.array([0.0, -2.5])   # vertical shift
base = center + offset

# class-wise projection ranges
t1_min, t1_max = y1.min(), y1.max()
t2_min, t2_max = y2.min(), y2.max()

# red segment (class 1)
p1 = base + t1_min * w
p2 = base + t1_max * w
plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=3)

# green segment (class 2)
p3 = base + t2_min * w
p4 = base + t2_max * w
plt.plot([p3[0], p4[0]], [p3[1], p4[1]], 'g-', linewidth=3)

# --------------------------------------------------
# plot settings
# --------------------------------------------------
plt.axis('equal')
plt.grid(True)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('LDA projection result')
plt.show()
