#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Generate plots for the linear regression blog post."""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)


def ols_qr(X, y):
    """Closed-form least squares via QR, never forming X^T X."""
    Q, R = np.linalg.qr(X)
    return np.linalg.solve(R, Q.T @ y)


def soft_threshold(z, lam):
    """Element-wise argmin_x 0.5 (x - z)^2 + lam |x|."""
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0.0)


def lasso_coord_descent(X, y, lam, n_iter=500):
    """Coordinate descent on the standardised LASSO."""
    n, d = X.shape
    theta = np.zeros(d)
    col_norms = (X ** 2).sum(axis=0)
    for _ in range(n_iter):
        for j in range(d):
            r = y - X @ theta + X[:, j] * theta[j]
            rho = X[:, j] @ r
            theta[j] = soft_threshold(rho, lam * n) / col_norms[j]
    return theta


def ridge_closed_form(X, y, lam):
    n, d = X.shape
    return np.linalg.solve(X.T @ X + n * lam * np.eye(d), X.T @ y)


# 1. OLS fit with residuals ----------------------------------------------------
n = 25
x = np.linspace(0, 10, n)
y_true = 0.7 * x + 1.0
y = y_true + rng.normal(0, 1.0, n)
X = np.column_stack([np.ones(n), x])
theta = ols_qr(X, y)
y_hat = X @ theta

fig, ax = plt.subplots(figsize=(6, 4))
for xi, yi, yhi in zip(x, y, y_hat):
    ax.plot([xi, xi], [yi, yhi], color="gray", lw=0.8, alpha=0.7)
ax.scatter(x, y, s=25, color="C0", zorder=3, label="data")
xs = np.linspace(0, 10, 100)
ax.plot(xs, theta[0] + theta[1] * xs, color="C3", lw=2, label="OLS fit")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Least squares fit and residuals")
ax.legend()
fig.tight_layout()
fig.savefig("ols_fit.svg", bbox_inches="tight")
plt.close(fig)

# 2. Cost landscape ------------------------------------------------------------
t0 = np.linspace(-1, 3, 200)
t1 = np.linspace(-0.2, 1.6, 200)
T0, T1 = np.meshgrid(t0, t1)
J = np.zeros_like(T0)
for i in range(T0.shape[0]):
    for j in range(T0.shape[1]):
        J[i, j] = np.mean((y - T0[i, j] - T1[i, j] * x) ** 2)

fig, ax = plt.subplots(figsize=(6, 4))
levels = np.linspace(J.min(), J.min() + 30, 20)
cs = ax.contour(T0, T1, J, levels=levels, cmap="viridis")
ax.scatter([theta[0]], [theta[1]], color="C3", s=70, zorder=5, label="optimum")
ax.set_xlabel(r"$\theta_0$ (intercept)")
ax.set_ylabel(r"$\theta_1$ (slope)")
ax.set_title("Cost surface $J(\\theta_0, \\theta_1)$")
ax.legend()
fig.tight_layout()
fig.savefig("cost_landscape.svg", bbox_inches="tight")
plt.close(fig)

# 3. Outlier sensitivity -------------------------------------------------------
y_out = y.copy()
y_out[5] = 18.0  # one wild point
theta_out = ols_qr(X, y_out)

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(x, y, s=25, color="C0", label="clean data")
ax.scatter([x[5]], [y_out[5]], s=80, color="C1", zorder=4, label="outlier")
ax.plot(xs, theta[0] + theta[1] * xs, color="C0", lw=2, label="fit (clean)")
ax.plot(xs, theta_out[0] + theta_out[1] * xs, color="C1", lw=2, ls="--", label="fit (with outlier)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("OLS is moved by a single outlier")
ax.legend()
fig.tight_layout()
fig.savefig("outlier_sensitivity.svg", bbox_inches="tight")
plt.close(fig)

# 4. L1 vs L2 unit balls + loss ellipses --------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharex=True, sharey=True)
center = np.array([1.4, 0.6])
A = np.array([[1.0, 0.7], [0.7, 1.0]])
g = np.linspace(-2, 2, 400)
G0, G1 = np.meshgrid(g, g)
diff0 = G0 - center[0]
diff1 = G1 - center[1]
quad = A[0, 0] * diff0 ** 2 + 2 * A[0, 1] * diff0 * diff1 + A[1, 1] * diff1 ** 2

for ax, ball, title in zip(
    axes,
    ["l2", "l1"],
    ["Ridge: smooth ball, optimum off-axis", "LASSO: sharp corner, optimum on axis"],
):
    ax.contour(G0, G1, quad, levels=8, colors="C0", linewidths=0.7, alpha=0.7)
    if ball == "l2":
        circ = plt.Circle((0, 0), 0.9, fill=False, color="C3", lw=2)
        ax.add_patch(circ)
        opt = center / np.linalg.norm(center) * 0.9
    else:
        ax.plot([0.9, 0, -0.9, 0, 0.9], [0, 0.9, 0, -0.9, 0], color="C3", lw=2)
        opt = np.array([0.9, 0.0])
    ax.scatter([opt[0]], [opt[1]], color="C3", s=70, zorder=5)
    ax.scatter([center[0]], [center[1]], color="C0", s=70, marker="x", zorder=5)
    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(r"$\theta_1$")
axes[0].set_ylabel(r"$\theta_2$")
fig.suptitle("Why LASSO produces sparsity: corners attract the optimum")
fig.tight_layout()
fig.savefig("regularisation_balls.svg", bbox_inches="tight")
plt.close(fig)

# 5. Ridge vs LASSO coefficient paths -----------------------------------------
n_path, d_path = 60, 8
X_path = rng.normal(size=(n_path, d_path))
X_path /= np.linalg.norm(X_path, axis=0, keepdims=True) * np.sqrt(n_path)
true_theta = np.array([2.5, -2.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0])
y_path = X_path @ true_theta + rng.normal(0, 0.05, n_path)

lams = np.logspace(-3, 1, 50)
ridge_paths = np.array([ridge_closed_form(X_path, y_path, lam) for lam in lams])
lasso_paths = np.array([lasso_coord_descent(X_path, y_path, lam) for lam in lams])

fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
for j in range(d_path):
    axes[0].plot(lams, ridge_paths[:, j], lw=1.5)
    axes[1].plot(lams, lasso_paths[:, j], lw=1.5)
for ax, name in zip(axes, ["Ridge ($\\ell_2$)", "LASSO ($\\ell_1$)"]):
    ax.set_xscale("log")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlabel(r"$\lambda$")
    ax.set_title(name)
axes[0].set_ylabel(r"coefficient $\theta_j$")
fig.suptitle("Coefficient paths: ridge shrinks smoothly, LASSO zeroes out features")
fig.tight_layout()
fig.savefig("coef_paths.svg", bbox_inches="tight")
plt.close(fig)

# 6. Ridge SVD filter ----------------------------------------------------------
sig = np.linspace(0.01, 3.0, 400)
N_lam_values = [0.0, 0.05, 0.3, 1.0]
fig, ax = plt.subplots(figsize=(6, 4))
for nl in N_lam_values:
    if nl == 0:
        ax.plot(sig, 1 / sig, lw=2, label=r"OLS: $1/\sigma$")
    else:
        ax.plot(sig, sig / (sig ** 2 + nl), lw=2, label=fr"Ridge ($N\lambda={nl}$)")
ax.set_ylim(0, 6)
ax.set_xlabel(r"singular value $\sigma$")
ax.set_ylabel("filter factor")
ax.set_title("Ridge gently damps small singular values")
ax.legend()
fig.tight_layout()
fig.savefig("ridge_filter.svg", bbox_inches="tight")
plt.close(fig)
