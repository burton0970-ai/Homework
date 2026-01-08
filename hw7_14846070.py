import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 144
np.random.seed(0)

# --------------------------------------------------
# 1. Generate data (same style as lecture)
# --------------------------------------------------
x = np.linspace(0, 1, 20)
y = 0.3 + 1.0 * np.sin(2 * np.pi * x + 0.2)
y += 0.15 * np.random.randn(len(y))

# --------------------------------------------------
# 2. Model and cost function
# y = w1 + w2 sin(w3 x + w4)
# --------------------------------------------------
def model(x, w):
    return w[0] + w[1] * np.sin(w[2] * x + w[3])

def cost(w):
    e = y - model(x, w)
    return np.sum(e ** 2)

# --------------------------------------------------
# 3. Analytic gradient
# --------------------------------------------------
def grad_analytic(w):
    e = y - model(x, w)
    s = np.sin(w[2] * x + w[3])
    c = np.cos(w[2] * x + w[3])

    g1 = -2 * np.sum(e)
    g2 = -2 * np.sum(e * s)
    g3 = -2 * np.sum(e * w[1] * x * c)
    g4 = -2 * np.sum(e * w[1] * c)

    return np.array([g1, g2, g3, g4])

# --------------------------------------------------
# 4. Numerical gradient (finite difference)
# --------------------------------------------------
def grad_numeric(w, eps=1e-8):
    g = np.zeros_like(w)
    J0 = cost(w)
    for k in range(len(w)):
        w_eps = w.copy()
        w_eps[k] += eps
        g[k] = (cost(w_eps) - J0) / eps
    return g

# --------------------------------------------------
# 5. Gradient descent parameters (IMPORTANT FIX)
# --------------------------------------------------
alpha = 0.005      # smaller learning rate (key fix)
iters = 1200

# initial weights (given in lecture style)
w0 = np.array([-0.2, 1.5, 6.0, -1.0])

# --------------------------------------------------
# 6. Gradient descent using analytic gradient
# --------------------------------------------------
w_a = w0.copy()
for _ in range(iters):
    w_a -= alpha * grad_analytic(w_a)

# --------------------------------------------------
# 7. Gradient descent using numeric gradient
# --------------------------------------------------
w_n = w0.copy()
for _ in range(iters):
    w_n -= alpha * grad_numeric(w_n)

# --------------------------------------------------
# 8. Plot results (match reference figure)
# --------------------------------------------------
xt = np.linspace(0, 1, 300)
yt_a = model(xt, w_a)
yt_n = model(xt, w_n)

plt.figure(figsize=(6, 4))
plt.plot(x, y, 'k.', label='data')
plt.plot(xt, yt_a, 'b--', linewidth=2, label='analytic method')
plt.plot(xt, yt_n, 'r-', linewidth=2, label='numeric method')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gradient descent result')
plt.legend()
plt.grid(True)
plt.show()

# --------------------------------------------------
# 9. Print final weights
# --------------------------------------------------
print('Final weights:')
print('Analytic method:', w_a)
print('Numeric method :', w_n)
