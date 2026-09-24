#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Generate plots for the recurrent neural network blog post."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

rng = np.random.default_rng(0)


def rnn_forward(xs, Wx, Wh, b):
    """Elman RNN forward pass over a sequence. xs: (T, d_in). Returns hs: (T+1, d_hid)."""
    T, _ = xs.shape
    d_hid = b.shape[0]
    hs = np.zeros((T + 1, d_hid))
    for t in range(T):
        hs[t + 1] = np.tanh(Wx @ xs[t] + Wh @ hs[t] + b)
    return hs


def rnn_bptt(xs, hs, dys, Wx, Wh, Wy):
    """BPTT for an Elman RNN with linear output and per-step output gradients dys.

    Returns dWx, dWh, dWy, db.
    """
    T = xs.shape[0]
    d_hid = hs.shape[1]
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    dWy = np.zeros_like(Wy)
    db = np.zeros(d_hid)
    dh_next = np.zeros(d_hid)
    for t in reversed(range(T)):
        dy = dys[t]
        dWy += np.outer(dy, hs[t + 1])
        dh = Wy.T @ dy + dh_next
        dz = dh * (1.0 - hs[t + 1] ** 2)  # tanh prime
        db += dz
        dWx += np.outer(dz, xs[t])
        dWh += np.outer(dz, hs[t])
        dh_next = Wh.T @ dz
    return dWx, dWh, dWy, db


# 1. Unrolled computation graph ----------------------------------------------
def draw_unroll():
    fig, ax = plt.subplots(figsize=(9, 3.6))
    T = 4
    y_h = 0.5
    y_x = -0.4
    y_y = 1.4
    for t in range(T):
        x = t * 1.6
        ax.add_patch(Circle((x, y_h), 0.22, facecolor="#bcd2ff", edgecolor="k", lw=1.2))
        ax.text(x, y_h, f"$h_{t+1}$", ha="center", va="center", fontsize=11)
        ax.add_patch(Circle((x, y_x), 0.18, facecolor="#ffe2b0", edgecolor="k", lw=1.2))
        ax.text(x, y_x, f"$x_{t+1}$", ha="center", va="center", fontsize=10)
        ax.add_patch(Circle((x, y_y), 0.18, facecolor="#c5f0c5", edgecolor="k", lw=1.2))
        ax.text(x, y_y, f"$y_{t+1}$", ha="center", va="center", fontsize=10)
        ax.add_patch(
            FancyArrowPatch(
                (x, y_x + 0.22), (x, y_h - 0.22), arrowstyle="->", mutation_scale=12
            )
        )
        ax.add_patch(
            FancyArrowPatch(
                (x, y_h + 0.22), (x, y_y - 0.22), arrowstyle="->", mutation_scale=12
            )
        )
        if t > 0:
            ax.add_patch(
                FancyArrowPatch(
                    ((t - 1) * 1.6 + 0.22, y_h),
                    (x - 0.22, y_h),
                    arrowstyle="->",
                    mutation_scale=12,
                )
            )
            ax.text((t - 1) * 1.6 + 0.8, y_h + 0.13, "$W_h$", ha="center", fontsize=9)
    ax.text(-1.0, y_h, "$h_0$", ha="center", va="center", fontsize=11)
    ax.add_patch(
        FancyArrowPatch((-0.85, y_h), (-0.22, y_h), arrowstyle="->", mutation_scale=12)
    )
    ax.text(-0.55, y_h + 0.13, "$W_h$", ha="center", fontsize=9)
    ax.set_xlim(-1.5, T * 1.6)
    ax.set_ylim(-1.0, 2.0)
    ax.set_axis_off()
    ax.set_title("Unrolled Elman RNN over $T$ time steps")
    fig.tight_layout()
    fig.savefig("rnn_unroll.svg", bbox_inches="tight")
    plt.close(fig)


# 2. Vanishing/exploding gradient norm vs sequence length ---------------------
def draw_vanish():
    d = 16
    Ts = list(range(2, 81, 2))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, scale in [
        ("$\\rho(W_h) \\approx 0.7$", 0.7),
        ("$\\rho(W_h) \\approx 1.0$", 1.0),
        ("$\\rho(W_h) \\approx 1.3$", 1.3),
    ]:
        norms = []
        # construct a matrix with target spectral radius
        A = rng.normal(size=(d, d))
        eigs = np.linalg.eigvals(A)
        A = A * (scale / max(abs(eigs)))
        for T in Ts:
            P = np.eye(d)
            for _ in range(T):
                P = A @ P
            norms.append(float(np.linalg.norm(P, 2)))
        ax.plot(Ts, norms, "-", label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Sequence length $T$")
    ax.set_ylabel("$\\|J^T\\|_2$")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig("rnn_vanish.svg", bbox_inches="tight")
    plt.close(fig)


# 3. BPTT memory cost vs T ---------------------------------------------------
def draw_compute():
    Ts = list(range(10, 201, 10))
    d_hid = 256
    full_mem = [T * d_hid * 4 / 1024 for T in Ts]  # KB, float32
    trunc_64 = [min(T, 64) * d_hid * 4 / 1024 for T in Ts]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(Ts, full_mem, "o-", label="Full BPTT")
    ax.plot(Ts, trunc_64, "s-", label="Truncated BPTT, $k = 64$")
    ax.set_xlabel("Sequence length $T$")
    ax.set_ylabel("Activation memory (KB, $d_{hid}=256$)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("bptt_compute.svg", bbox_inches="tight")
    plt.close(fig)


# 4. Effect of gradient clipping on a toy task -------------------------------
def draw_clip():
    rng2 = np.random.default_rng(1)
    T = 30
    d_hid = 8
    n_steps = 200
    targets = rng2.normal(size=(T, 1))
    xs = rng2.normal(size=(T, 1))

    def train(clip):
        Wx = rng2.normal(scale=0.5, size=(d_hid, 1))
        Wh = rng2.normal(scale=1.05, size=(d_hid, d_hid))
        Wy = rng2.normal(scale=0.5, size=(1, d_hid))
        b = np.zeros(d_hid)
        losses = []
        for _ in range(n_steps):
            hs = rnn_forward(xs, Wx, Wh, b)
            ys = hs[1:] @ Wy.T
            loss = float(0.5 * np.mean((ys - targets) ** 2))
            losses.append(min(loss, 1e6))
            dys = (ys - targets) / T
            dWx, dWh, dWy, db = rnn_bptt(xs, hs, dys, Wx, Wh, Wy)
            if clip is not None:
                norm = np.sqrt(sum(np.sum(g**2) for g in (dWx, dWh, dWy, db)))
                if norm > clip:
                    s = clip / norm
                    dWx, dWh, dWy, db = dWx * s, dWh * s, dWy * s, db * s
            Wx -= 0.01 * dWx
            Wh -= 0.01 * dWh
            Wy -= 0.01 * dWy
            b -= 0.01 * db
        return losses

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train(None), label="No clipping")
    ax.plot(train(1.0), label="Clip at $\\|g\\|=1$")
    ax.set_xlabel("Optimisation step")
    ax.set_ylabel("Training loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig("clip_effect.svg", bbox_inches="tight")
    plt.close(fig)


# 5. Singular value distribution: random vs orthogonal ----------------------
def draw_orth():
    d = 64
    A = rng.normal(scale=1.0 / np.sqrt(d), size=(d, d))
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    sv_a = np.linalg.svd(A, compute_uv=False)
    sv_q = np.linalg.svd(Q, compute_uv=False)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bins = np.linspace(0.0, max(2.0, sv_a.max() * 1.05), 25)
    ax.hist(sv_a, bins=bins, alpha=0.6, label="Gaussian $1/\\sqrt{d}$ scaling")
    ax.axvline(
        float(sv_q.mean()),
        color="C1",
        lw=2.5,
        label=f"Orthogonal init ($\\sigma_i \\equiv {sv_q.mean():.2f}$)",
    )
    ax.set_xlabel("Singular value of $W_h$")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("orth_init.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    draw_unroll()
    draw_vanish()
    draw_compute()
    draw_clip()
    draw_orth()
    print("ok")
