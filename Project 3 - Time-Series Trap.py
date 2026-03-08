#Project 3: Time-Series Trap

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

np.random.seed(42)

# ── Simulate time-series data ───────────────────────────────────
n_samples = 200
time = np.arange(n_samples)  # time index (0 to 199)
# Trend + seasonality + noise
true_trend = 0.05 * time
seasonality = 2 * np.sin(2 * np.pi * time / 30)  # monthly-ish cycle
noise = np.random.normal(0, 0.8, n_samples)
y_ts = true_trend + seasonality + noise

# X = time (for regression)
X_ts = time.reshape(-1, 1)

# Model (moderate complexity)
degree = 6
poly = PolynomialFeatures(degree)
model = make_pipeline(poly, LinearRegression())

# ── Bad split: random shuffle (leakage) ─────────────────────────
X_train_rand, X_test_rand, y_train_rand, y_test_rand = train_test_split(
    X_ts, y_ts, test_size=0.3, random_state=42
)

model.fit(X_train_rand, y_train_rand.ravel())
y_pred_rand = model.predict(X_test_rand)
mse_rand = mean_squared_error(y_test_rand, y_pred_rand)

print(f"Bad (random shuffle) split → Test MSE: {mse_rand:.3f}")

# ── Good split: chronological (past → train, future → test) ─────
split_idx = int(0.7 * n_samples)
X_train_chrono = X_ts[:split_idx]
y_train_chrono = y_ts[:split_idx]
X_test_chrono  = X_ts[split_idx:]
y_test_chrono  = y_ts[split_idx:]

model.fit(X_train_chrono, y_train_chrono.ravel())
y_pred_chrono = model.predict(X_test_chrono)
mse_chrono = mean_squared_error(y_test_chrono, y_pred_chrono)

print(f"Good (chronological) split → Test MSE: {mse_chrono:.3f}")

# ── Plot both for comparison ────────────────────────────────────
plt.figure(figsize=(14, 6))

# Bad (random) split
plt.subplot(1, 2, 1)
plt.scatter(X_train_rand, y_train_rand, color='blue', alpha=0.6, s=30, label='Train (random)')
plt.scatter(X_test_rand, y_test_rand, color='orange', alpha=0.6, s=30, label='Test (random)')
plt.plot(time, true_trend + seasonality, 'g--', linewidth=2, label='True trend + seasonality')
plt.title(f"Bad: Random Shuffle Split\nTest MSE: {mse_rand:.3f}")
plt.xlabel('Time index')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)

# Good (chronological) split
plt.subplot(1, 2, 2)
plt.scatter(X_train_chrono, y_train_chrono, color='blue', alpha=0.6, s=30, label='Train (past)')
plt.scatter(X_test_chrono, y_test_chrono, color='orange', alpha=0.6, s=30, label='Test (future)')
plt.plot(time, true_trend + seasonality, 'g--', linewidth=2, label='True trend + seasonality')
plt.title(f"Good: Chronological Split\nTest MSE: {mse_chrono:.3f}")
plt.xlabel('Time index')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()