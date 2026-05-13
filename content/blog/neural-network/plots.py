#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "matplotlib", "jax"]
# ///
"""Generate plots for the neural network blog post."""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)


def forward_backward_mlp(X, y, W1, b1, W2, b2):
    """Forward + backward pass for a 2-layer ReLU regressor with MSE loss."""
    z1 = X @ W1.T + b1  # pre-activations
    a1 = np.maximum(0.0, z1)  # ReLU
    y_hat = a1 @ W2.T + b2  # linear head
    loss = float(np.mean((y_hat - y) ** 2))
    delta2 = 2.0 * (y_hat - y) / len(y)  # output error
    grad_W2, grad_b2 = delta2.T @ a1, delta2.sum(0)
    delta1 = (delta2 @ W2) * (z1 > 0)  # backprop through ReLU
    grad_W1, grad_b1 = delta1.T @ X, delta1.sum(0)
    return loss, (grad_W1, grad_b1, grad_W2, grad_b2)


def adam_step(theta, grad, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    """One Adam update with bias correction. Inputs and outputs are flat arrays."""
    m = b1 * m + (1 - b1) * grad
    v = b2 * v + (1 - b2) * grad**2
    m_hat = m / (1 - b1**t)
    v_hat = v / (1 - b2**t)
    return theta - lr * m_hat / (np.sqrt(v_hat) + eps), m, v


def init_mlp(d_in, d_hid, d_out, scale=None):
    s1 = scale if scale is not None else np.sqrt(2.0 / d_in)
    s2 = scale if scale is not None else np.sqrt(2.0 / d_hid)
    W1 = rng.normal(scale=s1, size=(d_hid, d_in))
    b1 = np.zeros(d_hid)
    W2 = rng.normal(scale=s2, size=(d_out, d_hid))
    b2 = np.zeros(d_out)
    return W1, b1, W2, b2


# 1. XOR: linear failure vs MLP success ---------------------------------------
X_xor = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y_xor = np.array([[0.0], [1.0], [1.0], [0.0]])

W1, b1, W2, b2 = init_mlp(2, 16, 1)
for _ in range(8000):
    loss, (gW1, gb1, gW2, gb2) = forward_backward_mlp(X_xor, y_xor, W1, b1, W2, b2)
    W1 -= 0.1 * gW1
    b1 -= 0.1 * gb1
    W2 -= 0.1 * gW2
    b2 -= 0.1 * gb2

g = np.linspace(-0.3, 1.3, 200)
G0, G1 = np.meshgrid(g, g)
grid = np.column_stack([G0.ravel(), G1.ravel()])
z1 = grid @ W1.T + b1
y_hat = np.maximum(0, z1) @ W2.T + b2
P = y_hat.reshape(G0.shape)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
for ax, title, panel in zip(
    axes,
    ["Logistic regression: no line works", "MLP: carves out the right region"],
    ["lin", "mlp"],
):
    if panel == "mlp":
        cf = ax.contourf(G0, G1, P, levels=20, cmap="RdBu_r", alpha=0.7, vmin=0, vmax=1)
        ax.contour(G0, G1, P, levels=[0.5], colors="k", linewidths=2)
    else:
        ax.contourf(
            G0,
            G1,
            np.zeros_like(P),
            levels=[-1, 0.5, 1],
            colors=["#cdd6f4", "#cdd6f4"],
            alpha=0.4,
        )
        ax.plot(
            [-0.3, 1.3],
            [1.6, -0.0],
            color="k",
            lw=2,
            ls="--",
            label="any straight line fails",
        )
        ax.legend(loc="upper right", fontsize=8)
    for i in range(4):
        c = "C3" if y_xor[i] == 1 else "C0"
        ax.scatter(
            X_xor[i, 0], X_xor[i, 1], s=160, color=c, edgecolor="k", lw=1.2, zorder=5
        )
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_xlabel(r"$x_1$")
    ax.set_title(title, fontsize=10)
axes[0].set_ylabel(r"$x_2$")
fig.suptitle("XOR: a single hidden layer turns an unsolvable problem into a solved one")
fig.tight_layout()
fig.savefig("xor_failure.svg", bbox_inches="tight")
plt.close(fig)

# 2. Activations + their derivatives ------------------------------------------
zs = np.linspace(-4, 4, 400)
sig = 1 / (1 + np.exp(-zs))
sig_d = sig * (1 - sig)
th = np.tanh(zs)
th_d = 1 - th**2
relu = np.maximum(0, zs)
relu_d = (zs > 0).astype(float)
gelu = 0.5 * zs * (1 + np.tanh(np.sqrt(2 / np.pi) * (zs + 0.044715 * zs**3)))
phi_z = np.exp(-(zs**2) / 2) / np.sqrt(2 * np.pi)
Phi_z = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (zs + 0.044715 * zs**3)))
gelu_d = Phi_z + zs * phi_z

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for name, vals in [("sigmoid", sig), ("tanh", th), ("ReLU", relu), ("GELU", gelu)]:
    axes[0].plot(zs, vals, lw=2, label=name)
for name, vals in [
    ("sigmoid", sig_d),
    ("tanh", th_d),
    ("ReLU", relu_d),
    ("GELU", gelu_d),
]:
    axes[1].plot(zs, vals, lw=2, label=name)
for ax, title in zip(axes, ["Activation $\\phi(z)$", "Derivative $\\phi'(z)$"]):
    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlabel("z")
    ax.set_title(title)
    ax.legend(fontsize=8)
fig.suptitle(
    "Sigmoid and tanh saturate; ReLU and GELU keep their derivative away from zero on the active side"
)
fig.tight_layout()
fig.savefig("activations.svg", bbox_inches="tight")
plt.close(fig)


# 3. Vanishing vs exploding gradient ------------------------------------------
def signal_norm(L, d, init):
    a = rng.normal(size=d)
    norms = [np.linalg.norm(a)]
    for _ in range(L):
        if init == "small":
            W = rng.normal(scale=0.5, size=(d, d))
        elif init == "xavier":
            W = rng.normal(scale=np.sqrt(2.0 / (d + d)), size=(d, d))
        elif init == "he":
            W = rng.normal(scale=np.sqrt(2.0 / d), size=(d, d))
        else:
            raise ValueError
        a = np.maximum(0, W @ a)
        norms.append(np.linalg.norm(a))
    return np.array(norms)


L, d = 50, 100
fig, ax = plt.subplots(figsize=(6, 4))
for init, label in [
    ("small", "small Gaussian: vanishes"),
    ("xavier", "Xavier (sigmoid-tuned, on ReLU)"),
    ("he", "He (right scale for ReLU)"),
]:
    avg = np.mean([signal_norm(L, d, init) for _ in range(15)], axis=0)
    ax.plot(avg, lw=2, label=label)
ax.set_yscale("log")
ax.set_xlabel("layer depth")
ax.set_ylabel(r"$\|a^{(\ell)}\|$  (averaged over runs)")
ax.set_title("Why initialisation matters: signal norm through 50 ReLU layers")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("vanish_explode.svg", bbox_inches="tight")
plt.close(fig)

# 4. Optimiser comparison ------------------------------------------------------
n_o = 200
X_o = rng.normal(size=(n_o, 5))
true_w = rng.normal(size=5) * 2
y_o = (X_o @ true_w + rng.normal(scale=0.1, size=n_o)).reshape(-1, 1)


def run_optim(name, lr, n_steps=200):
    w = np.zeros(5)
    m = np.zeros(5)
    v = np.zeros(5)
    vel = np.zeros(5)
    losses = []
    for t in range(1, n_steps + 1):
        idx = rng.integers(n_o, size=32)
        Xb, yb = X_o[idx], y_o[idx, 0]
        pred = Xb @ w
        loss = float(np.mean((pred - yb) ** 2))
        grad = 2 * Xb.T @ (pred - yb) / len(yb)
        if name == "sgd":
            w = w - lr * grad
        elif name == "momentum":
            vel = 0.9 * vel + grad
            w = w - lr * vel
        elif name == "adam":
            w, m, v = adam_step(w, grad, m, v, t, lr=lr)
        losses.append(loss)
    return losses


fig, ax = plt.subplots(figsize=(6, 4))
for name, lr in [("sgd", 0.05), ("momentum", 0.05), ("adam", 0.1)]:
    ax.plot(run_optim(name, lr), lw=2, label=name)
ax.set_yscale("log")
ax.set_xlabel("step")
ax.set_ylabel("training loss (MSE)")
ax.set_title("SGD, momentum, and Adam on a small regression")
ax.legend()
fig.tight_layout()
fig.savefig("optim_curves.svg", bbox_inches="tight")
plt.close(fig)

# 5. Universal approximation: ReLU MLP fits sin(x) on [-pi, pi], fails outside
n_u = 200
x_u = np.linspace(-np.pi, np.pi, n_u).reshape(-1, 1)
y_u = np.sin(x_u)
W1, b1, W2, b2 = init_mlp(1, 32, 1)
for t in range(1, 6001):
    loss, (gW1, gb1, gW2, gb2) = forward_backward_mlp(x_u, y_u, W1, b1, W2, b2)
    lr = 0.005
    W1 -= lr * gW1
    b1 -= lr * gb1
    W2 -= lr * gW2
    b2 -= lr * gb2

x_dense = np.linspace(-3 * np.pi, 3 * np.pi, 1200).reshape(-1, 1)
z1 = x_dense @ W1.T + b1
a1 = np.maximum(0, z1)
y_hat = a1 @ W2.T + b2

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.axvspan(-np.pi, np.pi, color="C2", alpha=0.08, label="training range")
ax.plot(x_dense, np.sin(x_dense), color="C0", lw=2, label="target $\\sin x$")
ax.plot(x_dense, y_hat, color="C3", lw=2, ls="--", label="MLP prediction")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_ylim(-3.5, 3.5)
ax.set_title(
    "Wide ReLU MLP fits sin(x) inside the training range; extrapolates linearly outside"
)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig("uat_fit.svg", bbox_inches="tight")
plt.close(fig)

# 6. Dropout vs no dropout overfitting ----------------------------------------
n_tr, n_val, d_in, d_h = 50, 200, 8, 64
X_tr = rng.normal(size=(n_tr, d_in))
y_tr = (X_tr[:, 0] + 0.5 * X_tr[:, 1] + rng.normal(scale=0.5, size=n_tr)).reshape(-1, 1)
X_val = rng.normal(size=(n_val, d_in))
y_val = (X_val[:, 0] + 0.5 * X_val[:, 1] + rng.normal(scale=0.5, size=n_val)).reshape(
    -1, 1
)


def train_one(use_dropout, n_steps=400, p=0.3):
    W1, b1, W2, b2 = init_mlp(d_in, d_h, 1)
    tr_curve, val_curve = [], []
    for _ in range(n_steps):
        z1 = X_tr @ W1.T + b1
        a1 = np.maximum(0, z1)
        if use_dropout:
            mask = (rng.uniform(size=a1.shape) > p).astype(float) / (1 - p)
            a1 = a1 * mask
        y_hat = a1 @ W2.T + b2
        delta2 = 2 * (y_hat - y_tr) / n_tr
        gW2 = delta2.T @ a1
        gb2 = delta2.sum(0)
        delta1 = (delta2 @ W2) * (z1 > 0)
        if use_dropout:
            delta1 = delta1 * mask
        gW1 = delta1.T @ X_tr
        gb1 = delta1.sum(0)
        W1 -= 0.05 * gW1
        b1 -= 0.05 * gb1
        W2 -= 0.05 * gW2
        b2 -= 0.05 * gb2
        # eval (no dropout at inference)
        a1_t = np.maximum(0, X_tr @ W1.T + b1)
        a1_v = np.maximum(0, X_val @ W1.T + b1)
        tr_curve.append(float(np.mean((a1_t @ W2.T + b2 - y_tr) ** 2)))
        val_curve.append(float(np.mean((a1_v @ W2.T + b2 - y_val) ** 2)))
    return tr_curve, val_curve


tr0, val0 = train_one(use_dropout=False)
tr1, val1 = train_one(use_dropout=True)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(tr0, color="C0", lw=2, label="no dropout: train")
ax.plot(val0, color="C0", lw=2, ls="--", label="no dropout: val")
ax.plot(tr1, color="C3", lw=2, label="dropout p=0.3: train")
ax.plot(val1, color="C3", lw=2, ls="--", label="dropout p=0.3: val")
ax.set_xlabel("step")
ax.set_ylabel("MSE")
ax.set_yscale("log")
ax.set_title("Dropout closes the train/val gap on a small dataset")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("dropout_overfit.svg", bbox_inches="tight")
plt.close(fig)


# 7. PINN: noisy sin(x) data on [-pi, pi] + ODE y'' + y = 0 on [-3pi, 3pi] -----
jax.config.update("jax_enable_x64", True)
jax_key = jax.random.PRNGKey(0)
H1, H2 = 64, 64
k1, k2, k3, k4, k5 = jax.random.split(jax_key, 5)
params = (
    jax.random.normal(k1, (H1,))
    * 1.5,  # W1: wide spread allows multiple wavelengths in basis
    jax.random.uniform(
        k2, (H1,), minval=-jnp.pi, maxval=jnp.pi
    ),  # b1: shifts span the domain
    jax.random.normal(k3, (H2, H1)) * jnp.sqrt(1.0 / H1),  # W2
    jnp.zeros(H2),  # b2
    jax.random.normal(k4, (H2,)) * jnp.sqrt(1.0 / H2),  # W3 (output)
    jnp.array(0.0),  # b3
)


def pinn_y(params, x):
    W1, b1, W2, b2, W3, b3 = params
    a1 = jnp.tanh(W1 * x + b1)
    a2 = jnp.tanh(W2 @ a1 + b2)
    return jnp.sum(W3 * a2) + b3


pinn_dy = jax.grad(pinn_y, argnums=1)
pinn_ddy = jax.grad(pinn_dy, argnums=1)
pinn_y_v = jax.vmap(pinn_y, in_axes=(None, 0))
pinn_ddy_v = jax.vmap(pinn_ddy, in_axes=(None, 0))

# noisy observations of sin(x) on the inner range, used as the data term
n_data = 24
sigma = 0.15
x_data_np = np.linspace(-np.pi, np.pi, n_data)
y_data_np = np.sin(x_data_np) + sigma * rng.normal(size=n_data)
x_data = jnp.asarray(x_data_np)
y_data = jnp.asarray(y_data_np)


def pinn_loss(params, x_col, x_d, y_d, lam_pde=1.0):
    res = pinn_ddy_v(params, x_col) + pinn_y_v(params, x_col)
    pde = jnp.mean(res**2)
    pred = pinn_y_v(params, x_d)
    data = jnp.mean((pred - y_d) ** 2)
    return data + lam_pde * pde


pinn_grad = jax.jit(jax.value_and_grad(pinn_loss))

x_col = jnp.linspace(-3 * jnp.pi, 3 * jnp.pi, 512)
m_state = jax.tree_util.tree_map(jnp.zeros_like, params)
v_state = jax.tree_util.tree_map(jnp.zeros_like, params)
b1_a, b2_a, eps = 0.9, 0.999, 1e-8
for t in range(1, 20001):
    lr = 2e-3 if t <= 10000 else 5e-4
    loss_val, grads = pinn_grad(params, x_col, x_data, y_data)
    m_state = jax.tree_util.tree_map(
        lambda m, g: b1_a * m + (1 - b1_a) * g, m_state, grads
    )
    v_state = jax.tree_util.tree_map(
        lambda v, g: b2_a * v + (1 - b2_a) * g**2, v_state, grads
    )
    bc1, bc2 = 1 - b1_a**t, 1 - b2_a**t
    params = jax.tree_util.tree_map(
        lambda p, m, v: p - lr * (m / bc1) / (jnp.sqrt(v / bc2) + eps),
        params,
        m_state,
        v_state,
    )

# also train a data-only baseline (same architecture and budget, lam_pde = 0)
params_baseline = (
    jax.random.normal(k1, (H1,)) * 1.5,
    jax.random.uniform(k2, (H1,), minval=-jnp.pi, maxval=jnp.pi),
    jax.random.normal(k3, (H2, H1)) * jnp.sqrt(1.0 / H1),
    jnp.zeros(H2),
    jax.random.normal(k4, (H2,)) * jnp.sqrt(1.0 / H2),
    jnp.array(0.0),
)
m_state = jax.tree_util.tree_map(jnp.zeros_like, params_baseline)
v_state = jax.tree_util.tree_map(jnp.zeros_like, params_baseline)
for t in range(1, 20001):
    lr = 2e-3 if t <= 10000 else 5e-4
    loss_val, grads = pinn_grad(params_baseline, x_col, x_data, y_data, 0.0)
    m_state = jax.tree_util.tree_map(
        lambda m, g: b1_a * m + (1 - b1_a) * g, m_state, grads
    )
    v_state = jax.tree_util.tree_map(
        lambda v, g: b2_a * v + (1 - b2_a) * g**2, v_state, grads
    )
    bc1, bc2 = 1 - b1_a**t, 1 - b2_a**t
    params_baseline = jax.tree_util.tree_map(
        lambda p, m, v: p - lr * (m / bc1) / (jnp.sqrt(v / bc2) + eps),
        params_baseline,
        m_state,
        v_state,
    )

x_eval = np.linspace(-3 * np.pi, 3 * np.pi, 1200)
y_pinn = np.array(pinn_y_v(params, jnp.asarray(x_eval)))
y_baseline = np.array(pinn_y_v(params_baseline, jnp.asarray(x_eval)))

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.axvspan(-np.pi, np.pi, color="C2", alpha=0.08, label="data range")
ax.plot(x_eval, np.sin(x_eval), color="C0", lw=2, label="target $\\sin x$")
ax.plot(
    x_eval, y_baseline, color="gray", lw=1.5, ls=":", label="data-only MLP (no physics)"
)
ax.plot(x_eval, y_pinn, color="C3", lw=2, ls="--", label="PINN (data + ODE)")
ax.scatter(
    x_data_np,
    y_data_np,
    color="k",
    s=18,
    zorder=5,
    label=f"noisy data ($\\sigma = {sigma}$)",
)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_ylim(-2.0, 2.0)
ax.set_title(
    "PINN denoises inside the data range and extrapolates outside via $y'' + y = 0$"
)
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig("pinn_sin.svg", bbox_inches="tight")
plt.close(fig)
