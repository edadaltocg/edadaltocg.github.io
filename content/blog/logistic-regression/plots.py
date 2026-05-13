#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Generate plots for the logistic regression blog post."""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)


def bce_with_logits(z, y):
    """Numerically stable -log Bernoulli(y | sigmoid(z)) for y in {0, 1}."""
    return np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))


def irls_step(X, y, theta):
    """One Newton step on the BCE cost = one weighted least-squares solve."""
    z = X @ theta
    p = 1.0 / (1.0 + np.exp(-z))
    s = p * (1.0 - p)
    z_work = z + (y - p) / np.clip(s, 1e-8, None)
    XtS = X.T * s
    return np.linalg.solve(XtS @ X + 1e-8 * np.eye(X.shape[1]), XtS @ z_work)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# 1. Sigmoid family ------------------------------------------------------------
xs = np.linspace(-8, 8, 400)
fig, ax = plt.subplots(figsize=(6, 4))
for slope, bias, label in [
    (1.0, 0.0, r"$\theta_0=0,\ \theta_1=1$"),
    (2.5, 0.0, r"$\theta_0=0,\ \theta_1=2.5$"),
    (1.0, -3.0, r"$\theta_0=-3,\ \theta_1=1$"),
    (-1.5, 0.0, r"$\theta_0=0,\ \theta_1=-1.5$"),
]:
    ax.plot(xs, sigmoid(bias + slope * xs), lw=2, label=label)
ax.axhline(0.5, color="k", lw=0.4, ls="--")
ax.axhline(0, color="k", lw=0.4)
ax.axhline(1, color="k", lw=0.4)
ax.set_xlabel("x")
ax.set_ylabel(r"$\sigma(\theta_0 + \theta_1 x)$")
ax.set_title("The sigmoid family: slope sets sharpness, bias sets midpoint")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("sigmoid.svg", bbox_inches="tight")
plt.close(fig)

# 2. BCE per-sample loss -------------------------------------------------------
ps = np.linspace(0.001, 0.999, 400)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(ps, -np.log(ps), lw=2, label=r"$y=1$: $-\log\hat{p}$")
ax.plot(ps, -np.log(1 - ps), lw=2, label=r"$y=0$: $-\log(1-\hat{p})$")
ax.set_xlabel(r"predicted $\hat{p}$")
ax.set_ylabel("loss")
ax.set_title("Binary cross-entropy: confidence in the wrong answer is unbounded")
ax.set_ylim(0, 6)
ax.legend()
fig.tight_layout()
fig.savefig("bce_loss.svg", bbox_inches="tight")
plt.close(fig)

# 3. 2D decision boundary ------------------------------------------------------
n = 200
mu1, mu2 = np.array([-1.5, -0.5]), np.array([1.5, 0.8])
X1 = rng.normal(size=(n // 2, 2)) + mu1
X2 = rng.normal(size=(n // 2, 2)) + mu2
Xd = np.vstack([X1, X2])
yd = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
Xd_aug = np.column_stack([np.ones(n), Xd])

theta = np.zeros(3)
for _ in range(15):
    theta = irls_step(Xd_aug, yd, theta)

g = np.linspace(-5, 5, 200)
G0, G1 = np.meshgrid(g, g)
grid = np.column_stack([np.ones(G0.size), G0.ravel(), G1.ravel()])
P = sigmoid(grid @ theta).reshape(G0.shape)

fig, ax = plt.subplots(figsize=(6, 4.5))
cf = ax.contourf(G0, G1, P, levels=20, cmap="RdBu_r", alpha=0.6)
ax.contour(G0, G1, P, levels=[0.5], colors="k", linewidths=2)
ax.scatter(X1[:, 0], X1[:, 1], s=18, color="C0", edgecolor="k", linewidth=0.3, label=r"$y=0$")
ax.scatter(X2[:, 0], X2[:, 1], s=18, color="C3", edgecolor="k", linewidth=0.3, label=r"$y=1$")
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_title("Logistic regression: linear boundary + smooth probability field")
ax.legend(loc="lower right")
fig.colorbar(cf, ax=ax, label=r"$\hat{p}$")
fig.tight_layout()
fig.savefig("decision_boundary.svg", bbox_inches="tight")
plt.close(fig)

# 4. Optimisation convergence: GD vs IRLS --------------------------------------
def loss(theta):
    return float(np.mean(bce_with_logits(Xd_aug @ theta, yd)))


theta_gd = np.zeros(3)
gd_curve = [loss(theta_gd)]
eta = 0.5
for _ in range(60):
    z = Xd_aug @ theta_gd
    p = sigmoid(z)
    grad = Xd_aug.T @ (p - yd) / n
    theta_gd = theta_gd - eta * grad
    gd_curve.append(loss(theta_gd))

theta_irls = np.zeros(3)
irls_curve = [loss(theta_irls)]
for _ in range(15):
    theta_irls = irls_step(Xd_aug, yd, theta_irls)
    irls_curve.append(loss(theta_irls))

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(range(len(gd_curve)), gd_curve, lw=2, label="gradient descent")
ax.plot(range(len(irls_curve)), irls_curve, lw=2, marker="o", ms=4, label="IRLS (Newton)")
ax.set_xlabel("iteration")
ax.set_ylabel("BCE loss")
ax.set_title("IRLS converges in a handful of steps; GD takes many more")
ax.set_yscale("log")
ax.legend()
fig.tight_layout()
fig.savefig("optim_convergence.svg", bbox_inches="tight")
plt.close(fig)

# 5. Regularisation paths ------------------------------------------------------
n_p, d_p = 200, 8
Xp = rng.normal(size=(n_p, d_p))
true_w = np.array([2.5, -2.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0])
yp = (rng.uniform(size=n_p) < sigmoid(Xp @ true_w)).astype(float)


def fit_ridge_logistic(X, y, lam, n_iter=80):
    n_, d_ = X.shape
    th = np.zeros(d_)
    for _ in range(n_iter):
        p = sigmoid(X @ th)
        s = p * (1 - p)
        H = (X.T * s) @ X / n_ + lam * np.eye(d_)
        g = X.T @ (p - y) / n_ + lam * th
        th = th - np.linalg.solve(H, g)
    return th


def fit_lasso_logistic(X, y, lam, n_iter=200, eta=0.5):
    n_, d_ = X.shape
    th = np.zeros(d_)
    for _ in range(n_iter):
        p = sigmoid(X @ th)
        g = X.T @ (p - y) / n_
        th = np.sign(th - eta * g) * np.maximum(np.abs(th - eta * g) - eta * lam, 0.0)
    return th


lams = np.logspace(-3, 1, 40)
ridge = np.array([fit_ridge_logistic(Xp, yp, lam) for lam in lams])
lasso = np.array([fit_lasso_logistic(Xp, yp, lam) for lam in lams])

fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
for j in range(d_p):
    axes[0].plot(lams, ridge[:, j], lw=1.5)
    axes[1].plot(lams, lasso[:, j], lw=1.5)
for ax, name in zip(axes, ["Ridge", "LASSO"]):
    ax.set_xscale("log")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlabel(r"$\lambda$")
    ax.set_title(name)
axes[0].set_ylabel("coefficient")
fig.suptitle("Regularisation paths for logistic regression")
fig.tight_layout()
fig.savefig("reg_paths.svg", bbox_inches="tight")
plt.close(fig)

# 6. Separable-data divergence -------------------------------------------------
n_s = 80
X_sep = np.vstack([rng.normal(loc=-2, size=(n_s // 2, 2)), rng.normal(loc=2, size=(n_s // 2, 2))])
y_sep = np.concatenate([np.zeros(n_s // 2), np.ones(n_s // 2)])
X_sep_aug = np.column_stack([np.ones(n_s), X_sep])

theta_unreg = np.zeros(3)
theta_ridge = np.zeros(3)
norm_unreg, norm_ridge = [], []
for _ in range(150):
    p = sigmoid(X_sep_aug @ theta_unreg)
    theta_unreg = theta_unreg - 0.5 * X_sep_aug.T @ (p - y_sep) / n_s
    norm_unreg.append(np.linalg.norm(theta_unreg))
    p2 = sigmoid(X_sep_aug @ theta_ridge)
    theta_ridge = theta_ridge - 0.5 * (X_sep_aug.T @ (p2 - y_sep) / n_s + 0.05 * theta_ridge)
    norm_ridge.append(np.linalg.norm(theta_ridge))

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(norm_unreg, lw=2, label=r"unregularised: $\|\theta\|$ runs to infinity")
ax.plot(norm_ridge, lw=2, label=r"with $\ell_2$ penalty: bounded")
ax.set_xlabel("iteration")
ax.set_ylabel(r"$\|\theta_t\|_2$")
ax.set_title("Separable data: the MLE diverges, the ridge MAP does not")
ax.legend()
fig.tight_layout()
fig.savefig("separable_divergence.svg", bbox_inches="tight")
plt.close(fig)
