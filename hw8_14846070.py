import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import os

plt.rcParams['figure.dpi'] = 144
np.random.seed(0)

# --------------------------------------------------
# 1. Load data (or generate XOR-style data)
# --------------------------------------------------
CSV_PATH = 'data/hw8.csv'

if os.path.exists(CSV_PATH):
    data = np.loadtxt(CSV_PATH, delimiter=',')
    X = data[:, :2]
    y = data[:, 2].astype(int)
else:
    print('[Warning] hw8.csv not found. Generate XOR-style data.')

    n = 400
    X = np.random.randn(n, 2) * 6
    y = (X[:, 0] * X[:, 1] > 0).astype(int)   # XOR rule

# --------------------------------------------------
# 2. Feature transform (KEY POINT)
# φ(x) = [x1, x2, x1*x2]
# --------------------------------------------------
X_feat = np.column_stack([
    X[:, 0],
    X[:, 1],
    X[:, 0] * X[:, 1]
])

# --------------------------------------------------
# 3. Train linear SVM in transformed space
# --------------------------------------------------
clf = SVC(kernel='linear', C=1.0)
clf.fit(X_feat, y)

# --------------------------------------------------
# 4. Decision region
# --------------------------------------------------
x_min, x_max = X[:, 0].min() - 2, X[:, 0].max() + 2
y_min, y_max = X[:, 1].min() - 2, X[:, 1].max() + 2

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)

Xg = np.column_stack([
    xx.ravel(),
    yy.ravel(),
    xx.ravel() * yy.ravel()
])

Z = clf.predict(Xg)
Z = Z.reshape(xx.shape)

# --------------------------------------------------
# 5. Plot (match reference figure style)
# --------------------------------------------------
plt.figure(figsize=(6, 5))
plt.contourf(xx, yy, Z, alpha=0.35, cmap=plt.cm.Paired)

plt.scatter(X[y == 0, 0], X[y == 0, 1],
            c='red', s=12, label='Class 0')
plt.scatter(X[y == 1, 0], X[y == 1, 1],
            c='blue', s=12, label='Class 1')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Classification result (XOR-style decision regions)')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()
