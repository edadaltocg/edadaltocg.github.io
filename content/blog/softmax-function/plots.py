#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Generate plots for the softmax function blog post."""
import numpy as np
import matplotlib.pyplot as plt


def softmax(z, tau=1.0):
    z = z / tau
    z = z - np.max(z, axis=-1, keepdims=True)
    return np.exp(z) / np.sum(np.exp(z), axis=-1, keepdims=True)


logits = np.array([2.0, 1.0, 0.1])
temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]

fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(len(logits))
width = 0.15
for i, tau in enumerate(temperatures):
    ax.bar(x + i * width, softmax(logits, tau), width, label=f"\u03c4 = {tau}")

ax.set_xticks(x + width * 2)
ax.set_xticklabels([f"z{j+1}" for j in range(len(logits))])
ax.set_ylabel("Probability")
ax.set_title("Softmax with temperature scaling")
ax.legend()
fig.tight_layout()
fig.savefig("softmax_temp.svg", bbox_inches="tight")
plt.close(fig)
