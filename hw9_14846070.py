import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['figure.dpi'] = 144
np.random.seed(0)

# --------------------------------------------------
# 1. Load data or generate synthetic velocity
# --------------------------------------------------
CSV_PATH = 'data/hw9.csv'

if os.path.exists(CSV_PATH):
    data = np.loadtxt(CSV_PATH, delimiter=',')
    t = data[:, 0]
    v = data[:, 1]
else:
    print('[Warning] hw9.csv not found. Generate synthetic flow velocity.')

    t = np.linspace(0, 30, 600)
    v = 2.0 * np.sin(2 * np.pi * 0.7 * t)
    v += 0.4 * np.random.randn(len(t))   # noise
    v += 0.3                              # DC offset (important!)

dt = t[1] - t[0]

# --------------------------------------------------
# 2. Remove DC offset (KEY FIX #1)
# --------------------------------------------------
v0 = v - np.mean(v)

# --------------------------------------------------
# 3. Plot raw velocity
# --------------------------------------------------
plt.figure()
plt.plot(t, v0, 'r')
plt.title('Gas Flow Velocity')
plt.xlabel('time (in seconds)')
plt.ylabel('velocity')
plt.grid(True)
plt.show()

# --------------------------------------------------
# 4. Trapezoidal integration (KEY FIX #2)
# --------------------------------------------------
flow_raw = np.zeros_like(v0)
for i in range(1, len(v0)):
    flow_raw[i] = flow_raw[i-1] + 0.5 * (v0[i] + v0[i-1]) * dt

plt.figure()
plt.plot(t, flow_raw, 'r')
plt.title('Gas Net Flow (Raw Integration)')
plt.xlabel('time (in seconds)')
plt.ylabel('net flow')
plt.grid(True)
plt.show()

# --------------------------------------------------
# 5. Moving average filter (KEY FIX #3)
# --------------------------------------------------
def moving_average(x, window):
    return np.convolve(x, np.ones(window) / window, mode='same')

# window ≈ one oscillation period
window = 30
v_filt = moving_average(v0, window)

plt.figure()
plt.plot(t, v_filt, 'r')
plt.title('Gas Flow Velocity (Filtered)')
plt.xlabel('time (in seconds)')
plt.ylabel('velocity')
plt.grid(True)
plt.show()

# --------------------------------------------------
# 6. Integration after filtering
# --------------------------------------------------
flow_filt = np.zeros_like(v_filt)
for i in range(1, len(v_filt)):
    flow_filt[i] = flow_filt[i-1] + 0.5 * (v_filt[i] + v_filt[i-1]) * dt

plt.figure()
plt.plot(t, flow_filt, 'r')
plt.title('Gas Net Flow (Filtered Integration)')
plt.xlabel('time (in seconds)')
plt.ylabel('net flow')
plt.grid(True)
plt.show()
