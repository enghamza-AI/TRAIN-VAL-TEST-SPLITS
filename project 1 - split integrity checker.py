#Project 1: Split Integrity Checker

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

np.random.seed(42)

# ── Data ────────────────────────────────────────────────────────
n_samples = 200
X = np.random.uniform(-3, 3, n_samples).reshape(-1, 1)
true_fn = lambda x: 1.5 * x**2 - 2 * x + 0.8
y = true_fn(X) + np.random.normal(0, 0.8, n_samples).reshape(-1, 1)

# Model (moderate complexity so leakage is obvious)
degree = 6
poly = PolynomialFeatures(degree)
model = make_pipeline(poly, LinearRegression())

# ── Good split: completely random ───────────────────────────────
X_train_good, X_test_good, y_train_good, y_test_good = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model.fit(X_train_good, y_train_good.ravel())
y_pred_good = model.predict(X_test_good)
mse_good = mean_squared_error(y_test_good, y_pred_good)

print(f"Good random split → Test MSE: {mse_good:.3f}")

# ── Bad split: sort by X → massive leakage ──────────────────────
sorted_idx = np.argsort(X.ravel())
X_sorted = X[sorted_idx]
y_sorted = y[sorted_idx]

split_idx = int(0.7 * n_samples)
X_train_bad = X_sorted[:split_idx]
y_train_bad = y_sorted[:split_idx]
X_test_bad  = X_sorted[split_idx:]
y_test_bad  = y_sorted[split_idx:]

model.fit(X_train_bad, y_train_bad.ravel())
y_pred_bad = model.predict(X_test_bad)
mse_bad = mean_squared_error(y_test_bad, y_pred_bad)

print(f"Bad sorted split (leakage) → Test MSE: {mse_bad:.3f}")

# ── Visual comparison ───────────────────────────────────────────
X_plot = np.linspace(-3, 3, 300).reshape(-1, 1)
y_true_plot = true_fn(X_plot)

plt.figure(figsize=(14, 6))

# Good split plot
plt.subplot(1, 2, 1)
plt.scatter(X_train_good, y_train_good, color='blue', alpha=0.6, s=30, label='Train (good)')
plt.scatter(X_test_good, y_test_good, color='orange', alpha=0.6, s=30, label='Test (good)')
plt.plot(X_plot, y_true_plot, 'g--', linewidth=2, label='True function')
plt.plot(X_plot, model.predict(X_plot), 'r-', linewidth=2.5, label='Fit (good split)')
plt.title(f"Good Random Split\nTest MSE: {mse_good:.3f}")
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)

# Bad split plot (re-fit for consistent visualization)
model.fit(X_train_bad, y_train_bad.ravel())
plt.subplot(1, 2, 2)
plt.scatter(X_train_bad, y_train_bad, color='blue', alpha=0.6, s=30, label='Train (bad)')
plt.scatter(X_test_bad, y_test_bad, color='orange', alpha=0.6, s=30, label='Test (bad)')
plt.plot(X_plot, y_true_plot, 'g--', linewidth=2, label='True function')
plt.plot(X_plot, model.predict(X_plot), 'r-', linewidth=2.5, label='Fit (bad split)')
plt.title(f"Bad Sorted Split (leakage)\nTest MSE: {mse_bad:.3f}")
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()