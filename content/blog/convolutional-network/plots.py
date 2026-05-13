#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Generate plots for the convolutional network blog post."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

rng = np.random.default_rng(0)


def conv2d_forward(x, w, b=None):
    """Valid 2D cross-correlation. x: (H, W, Cin). w: (kH, kW, Cin, Cout). b: (Cout,)."""
    H, W, _ = x.shape
    kH, kW, _, Cout = w.shape
    oH, oW = H - kH + 1, W - kW + 1
    y = np.zeros((oH, oW, Cout))
    for i in range(oH):
        for j in range(oW):
            patch = x[i : i + kH, j : j + kW, :]
            y[i, j, :] = np.tensordot(patch, w, axes=([0, 1, 2], [0, 1, 2]))
    return y if b is None else y + b


def conv2d_backward(x, w, dy):
    """Gradients of valid conv. dy has shape (oH, oW, Cout). Returns dx, dw, db."""
    kH, kW, _, _ = w.shape
    oH, oW, _ = dy.shape
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    for i in range(oH):
        for j in range(oW):
            dx[i : i + kH, j : j + kW, :] += np.tensordot(
                dy[i, j, :], w, axes=([0], [3])
            )
            dw += np.einsum("c,hwd->hwdc", dy[i, j, :], x[i : i + kH, j : j + kW, :])
    db = dy.sum(axis=(0, 1))
    return dx, dw, db


# 1. Conv2D sliding kernel diagram --------------------------------------------
def draw_conv2d():
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    inp = np.array(
        [
            [3, 0, 1, 2, 7],
            [0, 1, 2, 3, 0],
            [1, 0, 1, 2, 1],
            [2, 1, 0, 1, 0],
            [3, 2, 1, 0, 1],
        ],
        dtype=float,
    )
    kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=float)
    out = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            out[i, j] = (inp[i : i + 3, j : j + 3] * kernel).sum()
    titles = ["Input (5x5)", "Kernel (3x3)", "Output (3x3)"]
    for ax, mat, title in zip(axes, [inp, kernel, out], titles):
        ax.imshow(
            mat,
            cmap="Blues" if title.startswith("Input") else "RdBu_r",
            vmin=-3,
            vmax=8,
        )
        for (i, j), v in np.ndenumerate(mat):
            ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=11)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].add_patch(
        Rectangle((-0.5, -0.5), 3, 3, linewidth=2, edgecolor="red", facecolor="none")
    )
    axes[2].add_patch(
        Rectangle((-0.5, -0.5), 1, 1, linewidth=2, edgecolor="red", facecolor="none")
    )
    fig.tight_layout()
    fig.savefig("conv2d.svg", bbox_inches="tight")
    plt.close(fig)


# 2. Receptive field growth ---------------------------------------------------
def draw_receptive_field():
    layers = list(range(1, 9))
    rf_3x3 = [3 + 2 * (depth - 1) for depth in layers]  # k=3, s=1
    rf_3x3_s2 = [3]
    for _ in range(7):
        rf_3x3_s2.append(rf_3x3_s2[-1] + 2 * (2 ** (len(rf_3x3_s2) - 1)))
    rf_5x5 = [5 + 4 * (depth - 1) for depth in layers]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(layers, rf_3x3, "o-", label="3x3, stride 1")
    ax.plot(layers, rf_5x5, "s-", label="5x5, stride 1")
    ax.plot(layers, rf_3x3_s2, "^-", label="3x3, stride 2")
    ax.set_xlabel("Layer depth")
    ax.set_ylabel("Receptive field (pixels)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("receptive_field.svg", bbox_inches="tight")
    plt.close(fig)


# 3. Pooling: max vs average --------------------------------------------------
def draw_pooling():
    inp = np.array(
        [
            [1, 3, 2, 4],
            [5, 6, 1, 2],
            [3, 2, 1, 0],
            [1, 2, 3, 4],
        ],
        dtype=float,
    )
    mp = np.array([[6, 4], [3, 4]], dtype=float)
    ap = np.array([[15 / 4, 9 / 4], [8 / 4, 8 / 4]], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for ax, mat, title in zip(
        axes, [inp, mp, ap], ["Input (4x4)", "Max-pool (2x2)", "Avg-pool (2x2)"]
    ):
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=6)
        for (i, j), v in np.ndenumerate(mat):
            ax.text(
                j,
                i,
                f"{v:.2f}".rstrip("0").rstrip("."),
                ha="center",
                va="center",
                fontsize=11,
            )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig("pooling.svg", bbox_inches="tight")
    plt.close(fig)


# 4. CNN signal norm vs depth (vanishing/exploding) --------------------------
def draw_cnn_vanish():
    H, W, Cin = 16, 16, 8
    Cout = 8
    kH = kW = 3
    fan_in = kH * kW * Cin
    depths = list(range(1, 21))
    schemes = [
        ("Naive sigma=0.1", 0.1, "sigmoid"),
        ("Glorot (1/sqrt(fan_in))", 1.0 / np.sqrt(fan_in), "relu"),
        ("He (sqrt(2/fan_in))", np.sqrt(2.0 / fan_in), "relu"),
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, scale, act in schemes:
        norms = []
        for D in depths:
            x = rng.normal(size=(H, W, Cin))
            for _ in range(D):
                w = rng.normal(scale=scale, size=(kH, kW, Cin, Cout))
                # use stride-1 'same' via padding so spatial size is preserved
                xp = np.pad(x, ((1, 1), (1, 1), (0, 0)))
                y = np.zeros((H, W, Cout))
                for i in range(H):
                    for j in range(W):
                        patch = xp[i : i + kH, j : j + kW, :]
                        y[i, j, :] = np.tensordot(patch, w, axes=([0, 1, 2], [0, 1, 2]))
                if act == "relu":
                    y = np.maximum(0.0, y)
                else:
                    y = 1.0 / (1.0 + np.exp(-y))
                x = y
                Cin = Cout
            norms.append(float(np.sqrt(np.mean(x**2))))
            Cin = 8
        ax.plot(depths, norms, "o-", label=label, markersize=4)
    ax.set_yscale("log")
    ax.set_xlabel("Depth (number of conv layers)")
    ax.set_ylabel("RMS activation")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig("cnn_vanish.svg", bbox_inches="tight")
    plt.close(fig)


# 5. Parameter count: CNN vs MLP at fixed input ------------------------------
def draw_param_count():
    sizes = [16, 32, 64, 128, 224]
    Cin = 3
    hidden = 256
    mlp_params = [(s * s * Cin) * hidden for s in sizes]
    cnn_params = [
        3 * 3 * Cin * 64 + 3 * 3 * 64 * 64 + 3 * 3 * 64 * hidden for _ in sizes
    ]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(sizes, mlp_params, "s-", label="MLP first layer (input -> 256)")
    ax.plot(sizes, cnn_params, "o-", label="3-conv stack (input -> 256ch)")
    ax.set_xlabel("Input side length (pixels)")
    ax.set_ylabel("Parameters")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig("param_count.svg", bbox_inches="tight")
    plt.close(fig)


# 6. Feature maps: edge filters on a toy image -------------------------------
def draw_feature_maps():
    H = W = 24
    img = np.zeros((H, W))
    img[6:18, 6:18] = 1.0
    img[10:14, 10:14] = 0.0
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=float)
    sobel_y = sobel_x.T
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)
    fmaps = []
    for k in [sobel_x, sobel_y, laplacian]:
        out = np.zeros((H - 2, W - 2))
        for i in range(H - 2):
            for j in range(W - 2):
                out[i, j] = (img[i : i + 3, j : j + 3] * k).sum()
        fmaps.append(out)
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Input")
    titles = ["Sobel-x", "Sobel-y", "Laplacian"]
    for ax, fmap, t in zip(axes[1:], fmaps, titles):
        ax.imshow(fmap, cmap="RdBu_r")
        ax.set_title(t)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig("feature_maps.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    draw_conv2d()
    draw_receptive_field()
    draw_pooling()
    draw_cnn_vanish()
    draw_param_count()
    draw_feature_maps()
    print("ok")
