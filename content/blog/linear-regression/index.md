+++
title = "Linear Regression"
date = 2026-04-19
description = "A short note on linear regression, its motivation and properties."

[taxonomies]
tags = ["machine-learning", "supervised-learning", "regression"]
categories = ["notes"]

[extra]
math = true
+++

## Univariate Linear Regression

Consider the following problem.
We want to predict a single scalar output $y \in \mathbb{R}$ from a single input $x \in \mathbb{R}$ using a model $f_{\theta}:\mathbb{R} \to \mathbb{R}$ with parameters vector $\boldsymbol{\theta}$.
We assume the relationship between inputs and outputs is non-deterministic, ie, we cannot model it in closed form easily.
So, we will take a data-driven approach.
Our goal is then finding the parameters of the best possible linear model that represents this mapping.

For this task, we have collected a supervised dataset of input/output pairs denoted $\mathcal{D}\=\\{(x\_i, y\_i)\\}\_{i=1}^N$.
We can view the relationship between inputs and outputs through its conditional probability distribution $Pr(y\mid x)$.
We also know that each observed training output $y_i$ should have high probability under its corresponding distribution $Pr(y_i|x_i, \phi)$ where $\phi$ is a set of unknown parameters that fully characterizes the distribution.
Let's rewrite the probability as $Pr(y_i|\phi_i)$, with $\phi_i=f_\theta(x_i)$.
The model can predict the distribution parameters either partially or fully.
Hence, let's pick the model parameters $\theta$ so that they maximise the joint probability of observing _all_ $N$ training outputs given their inputs.
This is the maximum likelihood{% sidenote(id="likelihood") %}The expression $Pr(y \mid x)$ can represent either a probability or a likelihood depending on which argument is considered free: it is a probability when viewed as a function of $y$ with $x$ fixed, and a likelihood when viewed as a function of the model parameters with $y$ fixed.{% end %} principle:

{% equation(id="mle-joint") %}
\hat{\theta}^\* = \argmax\_\theta \Big[ Pr\big(y_1, y_2, \ldots, y_N \mid x_1, x_2, \ldots, x_N, \theta\big)\Big]
{% end %}

You might look into this equation and have no clue how to solve it. And you are right, without further assumptions this joint distribution is intractable. Luckily, we can make one assumption that will simplify our lives a lot.

{% mathblock(kind="assumption", name="i.i.d.", id="iid") %}
The training examples $(x_i, y_i)$ are independently and identically distributed.
{% end %}

Identically distributed means the form of the probability distribution $Pr(y_i \mid x_i, \theta)$ is the same for all samples. In this case, they share the same parametric family and the same parameters $\phi$.
Independence means the joint probability factorises into a product of individual terms:

{% equation(id="factorization") %}
Pr\big(y_1, \ldots, y_N \mid x_1, \ldots, x_N, \theta\big) = \overset{N}{\underset{i=1}{\Pi}} Pr\big(y_i \mid x_i, \theta\big)
{% end %}

Substituting back into our objective, we arrive at

{% equation(id="mle") %}
\hat{\theta}^\* = \argmax*\theta \Bigg[\overset{N}{\underset{i=1}{\Pi}} Pr\big(y_i | f*\theta(x_i)\big)\Bigg]
{% end %}

As you may have noticed, this equation does not look convenient to optimise symbolically or numerically.
Probabilities lie between 0 and 1, and multiplying several of them may vanish very quickly, making it hard to represent with floating-point arithmetic due to its limited precision.
A trick to circumvent this is to apply a monotonic transformation (it preserves the ranking and does not change the solution) that makes the objective more computationally tractable.
One useful transformation is the log transform $\log(z)$.
By applying it to {{ eqref(id="mle") }}, we obtain:{% sidenote(id="log-product") %}We use the property $\log \prod_i a_i = \sum_i \log a_i$, ie, the logarithm of a product equals the sum of the logarithms.{% end %}

{% equation(id="mll") %}
\begin{aligned}
\hat{\theta}^\* &= \argmax*\theta \log \Bigg[\overset{N}{\underset{i=1}{\Pi}} Pr\big(y_i | f*\theta(x*i)\big)\Bigg] \\
&= \argmax*\theta \Bigg[\overset{N}{\underset{i=1}{\sum}} \log Pr\big(y_i | f_\theta(x_i)\big)\Bigg]
\end{aligned}
{% end %}

This is called the maximum log-likelihood criterion. Unlike the product of probabilities, the sum of log-probabilities does not suffer from numerical underflow, since each $\log Pr(\cdot)$ is a manageable negative number and their sum remains well within floating-point precision.

In machine learning, by convention, learning problems are formulated as a minimisation of a loss function. To convert our maximisation into a minimisation, it suffices to multiply the objective by $-1$ and change $\argmax$ to $\argmin$:

{% equation(id="nll") %}
\hat{\theta}^\* = \argmin*\theta \Bigg[-\overset{N}{\underset{i=1}{\sum}} \log Pr\big(y_i | f*\theta(x_i)\big)\Bigg]
{% end %}

This is known as the negative log-likelihood (NLL) loss.
Before we elaborate what loss functions are, let's solve this problem by picking a suitable distribution.
Consider the univariate normal distribution. It has support $y \in \mathbb{R}$ and parameters $\mu \in \mathbb{R}$ and $\sigma^2 \in \mathbb{R}^+$, ie, $\phi = (\mu, \sigma^2)$.
We can rewrite $Pr(y_i \mid x_i, \phi)$ as:

{% equation(id="gaussian-pdf") %}
Pr(y*i \mid x_i, \phi) = \mathcal{N}(y_i \mid f*\theta(x*i), \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - f*\theta(x_i))^2}{2\sigma^2}\right)
{% end %}

In this simplified version, we let the model predict the mean $\mu_i = f_\theta(x_i)$ and treat $\sigma^2$ as a fixed constant. Substituting {{ eqref(id="gaussian-pdf") }} into the NLL objective {{ eqref(id="nll") }}:

{% equation(id="nll-gaussian") %}
\begin{aligned}
\hat{\theta}^\* &= \argmin*\theta \Bigg[-\overset{N}{\underset{i=1}{\sum}} \log \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - f*\theta(x*i))^2}{2\sigma^2}\right)\Bigg] \\
&= \argmin*\theta \Bigg[-\overset{N}{\underset{i=1}{\sum}} \left(-\frac{1}{2}\log(2\pi\sigma^2) - \frac{(y_i - f_\theta(x_i))^2}{2\sigma^2}\right)\Bigg] \\
&= \argmin*\theta \Bigg[\overset{N}{\underset{i=1}{\sum}} \frac{(y_i - f*\theta(x_i))^2}{2\sigma^2} + \frac{N}{2}\log(2\pi\sigma^2)\Bigg]
\end{aligned}
{% end %}

Since $\sigma^2$ is constant with respect to $\theta$, both $\frac{1}{2\sigma^2}$ and $\frac{N}{2}\log(2\pi\sigma^2)$ are constants that do not affect the $\argmin$. We can drop them:

{% equation(id="mse") %}
\hat{\theta}^\* = \argmin*\theta \Bigg[\overset{N}{\underset{i=1}{\sum}} (y_i - f*\theta(x_i))^2\Bigg]
{% end %}

This is the mean squared error (MSE) criterion.{% sidenote(id="mse-note") %}Strictly speaking, the MSE divides by $N$: $\frac{1}{N}\sum_i (y_i - f_\theta(x_i))^2$. Since $N$ is a constant, it does not change the $\argmin$.{% end %}

{% mathblock(kind="definition", name="Loss function", id="loss") %}
A loss function $\ell(y, \hat{y})$ measures the discrepancy between a predicted value $\hat{y}$ and the true value $y$.
{% end %}

A well-behaved loss function should satisfy at least the following properties:

1. **Non-negativity**: $\ell(y, \hat{y}) \geq 0$ for all $y, \hat{y}$.
2. **Zero at perfect prediction**: $\ell(y, \hat{y}) = 0 \iff y = \hat{y}$.
3. **Differentiability** (at least almost everywhere) with respect to $\hat{y}$, so that gradient-based optimisation can be applied.

Convexity in $\hat{y}$ is also desirable since it guarantees a unique global minimum and makes optimisation tractable.
The squared error $\ell(y, \hat{y}) = (y - \hat{y})^2$ satisfies all of these properties.

{% mathblock(kind="proposition", name="Properties of the squared error", id="se-properties") %}
Let $\ell(y, \hat{y}) = (y - \hat{y})^2$. Then $\ell$ is:

1. non-negative,
2. zero if and only if $\hat{y} = y$,
3. infinitely differentiable, and
4. convex in $\hat{y}$.
   {% end %}

{% mathblock(kind="proof", name="", id="se-properties-proof") %}
Let $e = y - \hat{y}$.

**(i)** $\ell = e^2 \geq 0$ since the square of any real number is non-negative.

**(ii)** $\ell = e^2 = 0 \iff e = 0 \iff \hat{y} = y$.

**(iii)** $\ell$ is a polynomial in $\hat{y}$, hence infinitely differentiable. Its first and second derivatives are:
$\frac{\partial \ell}{\partial \hat{y}} = -2(y - \hat{y}), \qquad \frac{\partial^2 \ell}{\partial \hat{y}^2} = 2.$

**(iv)** A twice-differentiable function is convex if and only if its second derivative is non-negative everywhere. Since $\frac{\partial^2 \ell}{\partial \hat{y}^2} = 2 > 0$ for all $\hat{y}$, $\ell$ is (strictly) convex in $\hat{y}$. $\square$
{% end %}

{% mathblock(kind="definition", name="Cost function", id="cost") %}
A cost function $J(\theta)$ is the average loss over the training data: $J(\theta) = \frac{1}{N} \sum_{i=1}^N \ell\big(y_i, f_\theta(x_i)\big)$. The goal of learning is to find the parameters $\theta$ that minimise $J(\theta)$.
{% end %}

We can now dissect {{ eqref(id="mse") }} into these two components. The individual loss is the squared error $\ell(y_i, \hat{y}_i) = (y_i - \hat{y}_i)^2$, and the cost function is its average over the training set:

{% equation(id="mse-cost") %}
\hat{\theta}^\* = \argmin*\theta J(\theta) = \argmin*\theta \frac{1}{N} \overset{N}{\underset{i=1}{\sum}} (y*i - f*\theta(x_i))^2
{% end %}

You may have noticed that our model $f$ no longer predicts $y$ directly, but the mean $\mu$ of the normal distribution over $y$.
At inference time, however, we want a single best prediction given the inputs. This is called a _point estimate_.
A natural choice is the mode of the predicted distribution, i.e., the value of $y$ that maximizes the likelihood:

{% equation(id="point-estimate") %}
\hat{y} = \argmax*y Pr\!\left(y \mid f*{\hat{\theta}^\*}(x), \sigma^2\right)
{% end %}

For the normal distribution, the mode coincides with the mean $\mu$. Therefore $\hat{y} = f_{\hat{\theta}^*}(x)$, and the model's output can be used directly as the point estimate.

{% mathblock(kind="proposition", name="Mode of the normal distribution", id="gaussian-mode") %}
Let $y \sim \mathcal{N}(\mu, \sigma^2)$. Then $\argmax_y Pr(y \mid \mu, \sigma^2) = \mu$.
{% end %}

{% mathblock(kind="proof", name="", id="gaussian-mode-proof") %}
The density of the normal distribution is $Pr(y \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y - \mu)^2}{2\sigma^2}\right)$. Since $\frac{1}{\sqrt{2\pi\sigma^2}}$ is a positive constant and the exponential is a strictly increasing function, maximizing the density is equivalent to maximizing the exponent:

$$\argmax_y Pr(y \mid \mu, \sigma^2) = \argmax_y \left(-\frac{(y - \mu)^2}{2\sigma^2}\right) = \argmin_y (y - \mu)^2$$

The function $(y - \mu)^2$ is a convex quadratic with a unique minimum. Setting its derivative to zero: $\frac{d}{dy}(y - \mu)^2 = 2(y - \mu) = 0$, which gives $y = \mu$. $\square$
{% end %}
