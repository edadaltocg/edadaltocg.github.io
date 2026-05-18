+++
title = "The Softmax Function"
date = 2026-04-19
description = "A short note on the softmax function, its properties, and a numerically stable implementation."

[taxonomies]
tags = ["machine-learning", "math"]
categories = ["notes"]

[extra]
math = true
+++

## Definition

The softmax function maps a vector $\mathbf{z} \in \mathbb{R}^K$ to a probability distribution over $K$ classes:

{% math() %}
\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}, \quad i = 1, \dots, K.
{% end %}

Each output satisfies $\sigma(\mathbf{z})\_i \in (0, 1)$ and $\sum\_i \sigma(\mathbf{z})\_i = 1$. The name reads as a _soft_ (smooth, differentiable) approximation to $\operatorname{argmax}$: it returns a probability vector that concentrates on the largest logit instead of picking it outright, so we can use it as the final layer of a classifier and still backpropagate through it.

## Temperature scaling

A temperature parameter $\tau > 0$ controls the sharpness of the distribution:

{% math() %}
\sigma(\mathbf{z}; \tau)_i = \frac{e^{z_i / \tau}}{\sum_{j=1}^{K} e^{z_j / \tau}}.
{% end %}

As $\tau \to 0$, the output approaches a one-hot vector (argmax). As $\tau \to \infty$, the output approaches a uniform distribution.

{% mathblock(kind="note", name="Why temperature?", id="why-temperature") %}
The name is borrowed from statistical mechanics, where the Boltzmann distribution $p\_i \propto e^{-E\_i / kT}$ assigns probabilities to states of energy $E\_i$ at temperature $T$. Cold systems sit in the ground state (one-hot); hot systems explore all states equally (uniform). The softmax with $\tau$ is exactly that distribution with $-E\_i$ playing the role of the logit, which is why we use it whenever we want a knob that interpolates between confident and exploratory predictions, for instance when sampling from a language model. The picture to keep is that low temperature freezes the distribution onto its top option and high temperature melts it into uniform indecision.
{% end %}

## Numerical stability

A naive implementation overflows for large logits. The standard trick is to subtract $\max\_j z\_j$ before exponentiating, which does not change the result:

{% math() %}
\sigma(\mathbf{z})_i = \frac{e^{z_i - \max_j z_j}}{\sum_{j=1}^{K} e^{z_j - \max_j z_j}}.
{% end %}

## Implementation

{{ include_code(path="content/blog/softmax-function/plots.py", syntax="python", start=10, end=18) }}

<figure>
<img src="softmax_temp.svg" alt="Softmax output for different temperature values">
<figcaption>Softmax output for different temperature values</figcaption>
</figure>

## Connection to cross-entropy

The cross-entropy loss for a sample with true class $y$ is:

{% math() %}
\mathcal{L} = -\log \sigma(\mathbf{z})_y = -z_y + \log \sum_{j=1}^{K} e^{z_j},
{% end %}

which is the negative log-likelihood under the softmax model. Its gradient with respect to the logits has a particularly clean form:

{% math() %}
\frac{\partial \mathcal{L}}{\partial z_i} = \sigma(\mathbf{z})\_i - \mathbb{1}[i = y].
{% end %}

{% mathblock(kind="proof", name="(softmax cross-entropy gradient)", id="softmax-grad-proof") %}
Differentiating the two terms of $\mathcal{L} = -z\_y + \log \sum\_j e^{z\_j}$ with respect to $z\_i$: the first contributes $-\partial z\_y / \partial z\_i = -\mathbb{1}[i = y]$, and the second contributes $\partial / \partial z\_i \log \sum\_j e^{z\_j} = e^{z\_i} / \sum\_j e^{z\_j} = \sigma(\mathbf{z})\_i$ by the chain rule. Summing gives the stated identity. $\square$
{% end %}
