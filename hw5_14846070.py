import numpy as np
import matplotlib.pyplot as plt
import csv
import os

plt.rcParams['figure.dpi'] = 144

# --------------------------------------------------
# 1. Read hw5.csv (or generate synthetic data if not found)
# --------------------------------------------------
time = []
conc = []

CSV_PATH = 'hw5.csv'

if os.path.exists(CSV_PATH):
    print('[Info] hw5.csv found, reading data...')

    with open(CSV_PATH, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)

        # skip header if exists
        start_idx = 1 if not rows[0][0].replace('.', '').isdigit() else 0

        for row in rows[start_idx:]:
            time.append(float(row[0]))
            conc.append(float(row[1]))

    time = np.array(time)
    conc = np.array(conc)

else:
    print('[Warning] hw5.csv not found. Generate synthetic Brunhilda data.')

    # --------------------------------------------------
    # Generate synthetic exponential decay data
    # C(t) = A * exp(-k t) + noise
    # --------------------------------------------------
    np.random.seed(0)

    time = np.linspace(1, 200, 40)
    A_true = 1.2e-4
    k_true = 0.015

    conc = A_true * np.exp(-k_true * time)
    conc *= (1 + 0.15 * np.random.randn(len(conc)))  # add noise
    conc = np.abs(conc)  # ensure positive
# --------------------------------------------------
# 2. Linear scale plot
# --------------------------------------------------
plt.figure()
plt.plot(time, conc, 'r.', label='Measured data')
plt.xlabel('time (hours)')
plt.ylabel('Brunhilda concentration')
plt.title('Concentration vs Time')
plt.legend()
plt.grid(True)
plt.show()

# --------------------------------------------------
# 3. Exponential regression
# ln(C) = ln(A) - k t
# --------------------------------------------------
ln_conc = np.log(conc)
b, a = np.polyfit(time, ln_conc, 1)
A_est = np.exp(a)
k_est = -b

t_fit = np.linspace(time.min(), time.max(), 300)
c_fit = A_est * np.exp(-k_est * t_fit)

plt.figure()
plt.plot(time, conc, 'r.', label='Measured data')
plt.plot(t_fit, c_fit, 'b-', label='Exponential fit')
plt.xlabel('time (hours)')
plt.ylabel('Brunhilda concentration')
plt.title('Concentration vs Time (with regression)')
plt.legend()
plt.grid(True)
plt.show()

# --------------------------------------------------
# 4. log-log plot
# --------------------------------------------------
plt.figure()
plt.loglog(time, conc, 'r.', label='Measured data')
plt.xlabel('time (hours)')
plt.ylabel('Brunhilda concentration')
plt.title('Concentration vs Time (log-log scale)')
plt.legend()
plt.grid(True, which='both')
plt.show()

# --------------------------------------------------
# 5. log-log with regression
# --------------------------------------------------
plt.figure()
plt.loglog(time, conc, 'r.', label='Measured data')
plt.loglog(t_fit, c_fit, 'b-', label='Exponential fit')
plt.xlabel('time (hours)')
plt.ylabel('Brunhilda concentration')
plt.title('Concentration vs Time (log-log with regression)')
plt.legend()
plt.grid(True, which='both')
plt.show()

print('Regression result:')
print(f'  C(t) = A * exp(-k t)')
print(f'  A = {A_est:.4e}')
print(f'  k = {k_est:.4e}')
