#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Generate plots for the support vector machine blog post."""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)


def smo_pair_update(alpha, K, y, C, i, j, b):
    """Optimise (alpha_i, alpha_j) holding the equality constraint fixed."""
    if i == j:
        return alpha, b
    a_i_old, a_j_old = alpha[i], alpha[j]
    E_i = (alpha * y) @ K[:, i] + b - y[i]
    E_j = (alpha * y) @ K[:, j] + b - y[j]
    if y[i] != y[j]:
        L, H = max(0.0, a_j_old - a_i_old), min(C, C + a_j_old - a_i_old)
    else:
        L, H = max(0.0, a_i_old + a_j_old - C), min(C, a_i_old + a_j_old)
    if L == H:
        return alpha, b
    eta = 2 * K[i, j] - K[i, i] - K[j, j]
    if eta >= 0:
        return alpha, b
    alpha[j] = np.clip(a_j_old - y[j] * (E_i - E_j) / eta, L, H)
    alpha[i] = a_i_old + y[i] * y[j] * (a_j_old - alpha[j])
    return alpha, b


def predict_kernel_svm(alpha, y, K_query, b):
    """Decision function f(x) = sum_i alpha_i y_i k(x_i, x) + b."""
    return (alpha * y) @ K_query + b


def fit_smo(K, y, C, n_iter=200, tol=1e-3):
    n = len(y)
    alpha = np.zeros(n)
    b = 0.0
    for _ in range(n_iter):
        for i in range(n):
            f_i = (alpha * y) @ K[:, i] + b
            E_i = f_i - y[i]
            kkt_violated = (y[i] * E_i < -tol and alpha[i] < C) or (
                y[i] * E_i > tol and alpha[i] > 0
            )
            if not kkt_violated:
                continue
            j = int(rng.integers(n))
            while j == i:
                j = int(rng.integers(n))
            alpha, b = smo_pair_update(alpha, K, y, C, i, j, b)
            # bias update from KKT on a moved variable
            f_i = (alpha * y) @ K[:, i] + b
            f_j = (alpha * y) @ K[:, j] + b
            b_i = b - (f_i - y[i])
            b_j = b - (f_j - y[j])
            if 0 < alpha[i] < C:
                b = b_i
            elif 0 < alpha[j] < C:
                b = b_j
            else:
                b = 0.5 * (b_i + b_j)
    return alpha, b


def linear_kernel(A, B):
    return A @ B.T


def rbf_kernel(A, B, gamma=1.0):
    sq = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * A @ B.T
    return np.exp(-gamma * sq)


# 1. Hard margin: separating hyperplane, margin lines, support vectors --------
n_h = 40
A1 = rng.normal(size=(n_h // 2, 2)) * 0.5 + np.array([-2.5, -1.5])
A2 = rng.normal(size=(n_h // 2, 2)) * 0.5 + np.array([2.5, 1.5])
Xh = np.vstack([A1, A2])
yh = np.concatenate([-np.ones(n_h // 2), np.ones(n_h // 2)])
Kh = linear_kernel(Xh, Xh)
alpha_h, b_h = fit_smo(Kh, yh, C=1e3, n_iter=200)
w_h = (alpha_h * yh) @ Xh

xs = np.linspace(-5, 5, 50)
y_line = -(w_h[0] * xs + b_h) / w_h[1]
y_marg_pos = -(w_h[0] * xs + b_h - 1) / w_h[1]
y_marg_neg = -(w_h[0] * xs + b_h + 1) / w_h[1]

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.scatter(A1[:, 0], A1[:, 1], s=20, color="C0", label=r"$y=-1$")
ax.scatter(A2[:, 0], A2[:, 1], s=20, color="C3", label=r"$y=+1$")
ax.plot(xs, y_line, color="k", lw=2, label="boundary")
ax.plot(xs, y_marg_pos, color="k", ls="--", lw=1)
ax.plot(xs, y_marg_neg, color="k", ls="--", lw=1)
sv_mask = alpha_h > 1e-3
ax.scatter(
    Xh[sv_mask, 0],
    Xh[sv_mask, 1],
    s=140,
    facecolor="none",
    edgecolor="C1",
    lw=2,
    label="support vectors",
)
ax.set_xlim(-5, 5)
ax.set_ylim(-4, 4)
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_title("Maximum-margin separator")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig("hard_margin.svg", bbox_inches="tight")
plt.close(fig)

# 2. Loss comparison -----------------------------------------------------------
zs = np.linspace(-3, 3, 400)
hinge = np.maximum(0, 1 - zs)
bce = np.log1p(np.exp(-zs))
zero_one = (zs < 0).astype(float)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(zs, hinge, lw=2, label="hinge: $\\max(0, 1-yz)$")
ax.plot(zs, bce, lw=2, label="BCE: $\\log(1 + e^{-yz})$")
ax.plot(zs, zero_one, lw=2, ls="--", label="0-1: $\\mathbb{1}[yz < 0]$")
ax.set_xlabel(r"margin $yz$")
ax.set_ylabel("loss")
ax.set_title("Hinge sets a hard zero past the margin; BCE never quite stops pulling")
ax.legend()
fig.tight_layout()
fig.savefig("loss_comparison.svg", bbox_inches="tight")
plt.close(fig)

# 3. Soft margin: effect of C --------------------------------------------------
n_o = 60
B1 = rng.normal(size=(n_o // 2, 2)) * 0.9 + np.array([-1.0, -0.5])
B2 = rng.normal(size=(n_o // 2, 2)) * 0.9 + np.array([1.0, 0.5])
Xo = np.vstack([B1, B2])
yo = np.concatenate([-np.ones(n_o // 2), np.ones(n_o // 2)])
Ko = linear_kernel(Xo, Xo)

fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharex=True, sharey=True)
for ax, C in zip(axes, [0.05, 1.0, 100.0]):
    alpha_o, b_o = fit_smo(Ko, yo, C=C, n_iter=200)
    w_o = (alpha_o * yo) @ Xo
    line = -(w_o[0] * xs + b_o) / w_o[1] if abs(w_o[1]) > 1e-6 else np.zeros_like(xs)
    ax.scatter(B1[:, 0], B1[:, 1], s=18, color="C0")
    ax.scatter(B2[:, 0], B2[:, 1], s=18, color="C3")
    ax.plot(xs, line, color="k", lw=2)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_title(f"C = {C}")
    ax.set_xlabel(r"$x_1$")
axes[0].set_ylabel(r"$x_2$")
fig.suptitle("Soft margin: small C tolerates violations, large C insists on the data")
fig.tight_layout()
fig.savefig("soft_margin_C.svg", bbox_inches="tight")
plt.close(fig)


# 4. Linear vs RBF kernel on moons --------------------------------------------
def make_moons(n=200, noise=0.18):
    n_a = n // 2
    th_a = np.linspace(0, np.pi, n_a)
    A = np.column_stack([np.cos(th_a), np.sin(th_a)])
    B = np.column_stack([1 - np.cos(th_a), 1 - np.sin(th_a) - 0.5])
    X = np.vstack([A, B]) + rng.normal(scale=noise, size=(n, 2))
    y = np.concatenate([-np.ones(n_a), np.ones(n_a)])
    return X, y


Xm, ym = make_moons(n=200)
g = np.linspace(-2, 3, 80)
G0, G1 = np.meshgrid(g, g)
grid = np.column_stack([G0.ravel(), G1.ravel()])

fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True, sharey=True)
for ax, kname in zip(axes, ["linear", "rbf"]):
    if kname == "linear":
        K_train = linear_kernel(Xm, Xm)
        K_grid = linear_kernel(Xm, grid)
    else:
        K_train = rbf_kernel(Xm, Xm, gamma=1.5)
        K_grid = rbf_kernel(Xm, grid, gamma=1.5)
    alpha_m, b_m = fit_smo(K_train, ym, C=2.0, n_iter=200)
    f_grid = predict_kernel_svm(alpha_m, ym, K_grid, b_m).reshape(G0.shape)
    ax.contourf(
        G0, G1, f_grid, levels=[-1e9, 0, 1e9], colors=["#cdd6f4", "#f5c2c7"], alpha=0.7
    )
    ax.contour(G0, G1, f_grid, levels=[0], colors="k", linewidths=2)
    ax.scatter(Xm[ym == -1, 0], Xm[ym == -1, 1], s=15, color="C0")
    ax.scatter(Xm[ym == 1, 0], Xm[ym == 1, 1], s=15, color="C3")
    ax.set_title(f"{kname.upper()} kernel")
    ax.set_xlabel(r"$x_1$")
axes[0].set_ylabel(r"$x_2$")
fig.suptitle("Same data, two kernels: linear cannot separate the moons; RBF can")
fig.tight_layout()
fig.savefig("kernel_compare.svg", bbox_inches="tight")
plt.close(fig)

# 5. Dual coefficients (alphas) ------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
colors = ["C1" if a > 1e-3 else "C7" for a in alpha_h]
ax.bar(np.arange(len(alpha_h)), alpha_h, color=colors)
ax.set_xlabel("training sample index $i$")
ax.set_ylabel(r"$\alpha_i$")
ax.set_title("Dual coefficients on the hard-margin example: most are zero")
fig.tight_layout()
fig.savefig("dual_alphas.svg", bbox_inches="tight")
plt.close(fig)

# 6. Kernel cache memory growth ------------------------------------------------
Ns = np.logspace(2, 5, 50)
gram = Ns**2 * 8 / 1e9  # bytes for float64 -> GB
nystrom = Ns * 200 * 8 / 1e9  # R = 200 landmarks

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(Ns, gram, lw=2, label=r"full Gram matrix: $O(N^2)$")
ax.plot(Ns, nystrom, lw=2, label=r"Nystr\"om approx ($R{=}200$): $O(NR)$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("number of training samples $N$")
ax.set_ylabel("kernel-matrix memory (GB, float64)")
ax.set_title("Why kernel SVMs hit a wall, and why Nyström rescues them")
ax.axhline(8, color="k", ls=":", lw=1)
ax.text(2e2, 9, "8 GB RAM", fontsize=8)
ax.legend()
fig.tight_layout()
fig.savefig("kernel_cache_growth.svg", bbox_inches="tight")
plt.close(fig)
