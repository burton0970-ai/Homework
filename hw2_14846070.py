# If this script is not run under spyder IDE, comment the following two lines.
from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import cv2

plt.rcParams['figure.dpi'] = 144


# --------------------------------------------------
# calculate the eigenvalues and eigenvectors of a squared matrix
# the eigenvalues are decreasing ordered
def myeig(A, symmetric=False):
    if symmetric:
        lambdas, V = la.eigh(A)
    else:
        lambdas, V = la.eig(A)

    lambdas = np.real(lambdas)
    idx = np.argsort(lambdas)[::-1]
    return lambdas[idx], V[:, idx]


# --------------------------------------------------
# SVD: A = U * Sigma * V^T
# V: eigenvector matrix of A^T * A
def mysvd(A):
    lambdas, V = myeig(A.T @ A, symmetric=True)

    # avoid numerical issue
    lambdas = np.append(lambdas, 1e-12)
    rank = np.argwhere(lambdas < 1e-6).min()

    lambdas = lambdas[:rank]
    V = V[:, :rank]

    U = A @ V / np.sqrt(lambdas)
    Sigma = np.diag(np.sqrt(lambdas))

    return U, Sigma, V


# --------------------------------------------------
# Energy of a 2D signal (Frobenius norm squared)
def compute_energy(X):
    return np.sum(X ** 2)


# --------------------------------------------------
# Read image (or generate synthetic signal if not found)
img = cv2.imread('data/svd_demo1.jpg', cv2.IMREAD_GRAYSCALE)

if img is None:
    print('[Warning] Image not found. Use synthetic 2D signal instead.')

    h, w = 256, 256
    x = np.linspace(0, 255, w)
    A = np.tile(x, (h, 1)).astype(np.float64)

else:
    A = img.astype(np.float64)


# --------------------------------------------------
# SVD of A
U, Sigma, V = mysvd(A)
VT = V.T


# --------------------------------------------------
# Parameters
keep_r = 201
rs = np.arange(1, keep_r)

# --------------------------------------------------
# Energy of original signal
energy_A = compute_energy(A)

# --------------------------------------------------
# Energy of noise vs r
energy_N = np.zeros(keep_r)

for r in rs:
    A_bar = U[:, :r] @ Sigma[:r, :r] @ VT[:r, :]
    Noise = A - A_bar
    energy_N[r] = compute_energy(Noise)

# --------------------------------------------------
# Compute SNR
snr = np.zeros(keep_r)

for r in rs:
    snr[r] = 10 * np.log10(energy_A / energy_N[r])

# --------------------------------------------------
# Plot SNR vs r
plt.figure()
plt.plot(rs, snr[rs])
plt.xlabel('r')
plt.ylabel('SNR (dB)')
plt.title('SNR vs r using SVD')
plt.grid(True)
plt.show()


# --------------------------------------------------
# Verify theory:
# ||A - A_r||_F^2 = sum_{i=r+1} lambda_i
lambdas, _ = myeig(A.T @ A, symmetric=True)

energy_N_theory = np.zeros(keep_r)
for r in rs:
    energy_N_theory[r] = np.sum(lambdas[r:])

# --------------------------------------------------
# Plot energy verification
plt.figure()
plt.plot(rs, energy_N[rs], label='Computed Noise Energy')
plt.plot(rs, energy_N_theory[rs], '--', label='Sum of Eigenvalues')
plt.xlabel('r')
plt.ylabel('Energy')
plt.title('Energy Verification')
plt.legend()
plt.grid(True)
plt.show()
