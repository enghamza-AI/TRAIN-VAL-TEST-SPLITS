#Project 2: Leakage via Preprocessing

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

np.random.seed(42)

# ── Data ────────────────────────────────────────────────────────
n_samples = 200
X = np.random.uniform(-3, 3, n_samples).reshape(-1, 1)
true_fn = lambda x: 1.5 * x**2 - 2 * x + 0.8
y = true_fn(X) + np.random.normal(0, 0.8, n_samples).reshape(-1, 1)

# Model pipeline (without scaling yet)
degree = 6
model_base = make_pipeline(PolynomialFeatures(degree), LinearRegression())

# ── Wrong way: scale BEFORE split (leakage) ─────────────────────
scaler_wrong = StandardScaler()
X_scaled_wrong = scaler_wrong.fit_transform(X)  # ← fits on ALL data (train + val/test)

X_train_wrong, X_test_wrong, y_train_wrong, y_test_wrong = train_test_split(
    X_scaled_wrong, y, test_size=0.3, random_state=42
)

model_wrong = LinearRegression()
model_wrong.fit(X_train_wrong, y_train_wrong.ravel())
y_pred_wrong = model_wrong.predict(X_test_wrong)
mse_wrong = mean_squared_error(y_test_wrong, y_pred_wrong)

print(f"Wrong way (scale before split) → Test MSE: {mse_wrong:.3f}")

# ── Right way: split first, then scale using train stats only ───
X_train_right, X_test_right, y_train_right, y_test_right = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler_right = StandardScaler()
X_train_scaled = scaler_right.fit_transform(X_train_right)  # ← fit ONLY on train
X_test_scaled  = scaler_right.transform(X_test_right)      # ← apply to test (no refit!)

model_right = LinearRegression()
model_right.fit(X_train_scaled, y_train_right.ravel())
y_pred_right = model_right.predict(X_test_scaled)
mse_right = mean_squared_error(y_test_right, y_pred_right)

print(f"Right way (split first, scale after) → Test MSE: {mse_right:.3f}")

# ── Visual comparison (predictions on original scale) ───────────
X_plot = np.linspace(-3, 3, 300).reshape(-1, 1)
y_true_plot = true_fn(X_plot)

# For wrong way (scaled input)
X_plot_scaled_wrong = scaler_wrong.transform(X_plot)
y_pred_wrong_plot = model_wrong.predict(X_plot_scaled_wrong)

# For right way (scaled input)
X_plot_scaled_right = scaler_right.transform(X_plot)
y_pred_right_plot = model_right.predict(X_plot_scaled_right)

plt.figure(figsize=(14, 6))

# Wrong way
plt.subplot(1, 2, 1)
plt.scatter(X_train_wrong[:, 0], y_train_wrong, color='blue', alpha=0.6, s=30, label='Train (wrong)')
plt.scatter(X_test_wrong[:, 0], y_test_wrong, color='orange', alpha=0.6, s=30, label='Test (wrong)')
plt.plot(X_plot[:, 0], y_true_plot, 'g--', linewidth=2, label='True function')
plt.plot(X_plot[:, 0], y_pred_wrong_plot, 'r-', linewidth=2.5, label='Fit (wrong way)')
plt.title(f"Wrong: Scale Before Split\nTest MSE: {mse_wrong:.3f}")
plt.xlabel('X (scaled)')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)

# Right way
plt.subplot(1, 2, 2)
plt.scatter(X_train_right, y_train_right, color='blue', alpha=0.6, s=30, label='Train (right)')
plt.scatter(X_test_right, y_test_right, color='orange', alpha=0.6, s=30, label='Test (right)')
plt.plot(X_plot, y_true_plot, 'g--', linewidth=2, label='True function')
plt.plot(X_plot, y_pred_right_plot, 'r-', linewidth=2.5, label='Fit (right way)')
plt.title(f"Right: Split First, Scale After\nTest MSE: {mse_right:.3f}")
plt.xlabel('X (original)')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()