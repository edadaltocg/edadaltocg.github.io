+++
title = "Logistic Regression"
date = 2026-05-13
description = "A short note on logistic regression, its motivation and properties."

[taxonomies]
tags = ["machine-learning", "supervised-learning", "classification"]
categories = ["notes"]

[extra]
math = true
+++

## Univariate Logistic Regression

Consider the following problem.
We want to predict a binary output $y \in \\{0, 1\\}$ from a single input $x \in \mathbb{R}$ using a model $f\_{\theta}:\mathbb{R} \to (0, 1)$ with parameters vector $\boldsymbol{\theta}$.
The output now represents class membership rather than a real-valued quantity, so the model should return the probability that the example belongs to the positive class, $Pr(y=1 \mid x)$.
A bare linear function $\theta\_0 + \theta\_1 x$ ranges over all of $\mathbb{R}$ and is therefore unsuitable on its own. We need to squash it into the unit interval.
The standard choice is the **sigmoid** (logistic) function:

{% equation(id="sigmoid-def") %}
\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \sigma:\mathbb{R} \to (0, 1)
{% end %}

Composing it with a linear score gives the logistic model:

{% equation(id="logistic-model") %}
f_{\theta}(x) = \sigma(\theta_0 + \theta_1 x) = \frac{1}{1 + e^{-(\theta_0 + \theta_1 x)}}
{% end %}

where, as before, $\theta\_0$ is the intercept and $\theta\_1$ is the slope, so $\boldsymbol{\theta} = (\theta\_0, \theta\_1)$.

### Maximum Likelihood Estimation

For this task, we have collected a supervised dataset of input/label pairs denoted $\mathcal{D} = \\{(x\_i, y\_i)\\}\_{i=1}^N$ with $y\_i \in \\{0, 1\\}$.
As in the [regression case](/blog/linear-regression/), we view the relationship between inputs and outputs through its conditional probability distribution $Pr(y \mid x)$, and we seek the parameters $\theta$ that maximise the joint probability of observing all $N$ training labels given their inputs:

{% equation(id="mle-joint") %}
\hat{\theta}^{\ast} = \argmax_{\theta} \Big[ Pr\big(y_1, y_2, \ldots, y_N \mid x_1, x_2, \ldots, x_N, \theta\big)\Big]
{% end %}

Under the same i.i.d. assumption from the regression note, the joint factorises into a product of per-sample probabilities:

{% mathblock(kind="assumption", name="i.i.d.", id="iid") %}
The training examples $(x\_i, y\_i)$ are independently and identically distributed.
{% end %}

{% equation(id="factorization") %}
Pr\big(y_1, \ldots, y_N \mid x_1, \ldots, x_N, \theta\big) = \prod_{i=1}^{N} Pr\big(y_i \mid x_i, \theta\big)
{% end %}

### The Log-Likelihood

Multiplying many probabilities underflows in floating-point arithmetic, so we apply the monotonic log transform and flip the sign to convert the maximisation into a minimisation. The resulting negative log-likelihood (NLL) is:

{% equation(id="nll") %}
\hat{\theta}^{\ast} = \argmin_{\theta} \Bigg[-\sum_{i=1}^{N} \log Pr\big(y_i \mid f_{\theta}(x_i)\big)\Bigg]
{% end %}

### Choosing a Distribution

Because $y$ is binary, the natural choice is the Bernoulli distribution. It has support $y \in \\{0, 1\\}$ and a single parameter $p \in (0, 1)$ giving the probability of the positive class. With the model predicting $p\_i = f\_{\theta}(x\_i)$, we can write:

{% equation(id="bernoulli-pmf") %}
Pr(y_i \mid x_i, \theta) = p_i^{y_i}\,(1 - p_i)^{1 - y_i}
{% end %}

The two cases collapse into a single expression: when $y\_i = 1$ the second factor vanishes and we are left with $p\_i$; when $y\_i = 0$ the first factor vanishes and we are left with $1 - p\_i$. Substituting {{ eqref(id="bernoulli-pmf") }} into the NLL {{ eqref(id="nll") }}:

{% equation(id="nll-bernoulli") %}
\begin{aligned}
\hat{\theta}^{\ast} &= \argmin_{\theta} \Bigg[-\sum_{i=1}^{N} \log\!\Big( p_i^{y_i}\,(1 - p_i)^{1 - y_i} \Big)\Bigg] \\\\
&= \argmin_{\theta} \Bigg[-\sum_{i=1}^{N} \Big( y_i \log p_i + (1 - y_i) \log (1 - p_i) \Big)\Bigg]
\end{aligned}
{% end %}

This is the **binary cross-entropy** (BCE), or log-loss, criterion.

### Loss and Cost Functions

We can dissect the BCE into the same loss/cost components introduced in the [regression note](/blog/linear-regression/). The per-sample loss is:

{% equation(id="bce-loss") %}
\ell(y, \hat{p}) = -\,y \log \hat{p} - (1 - y) \log (1 - \hat{p})
{% end %}

and the cost function is its average over the training set:

{% equation(id="bce-cost") %}
\hat{\theta}^{\ast} = \argmin_{\theta} J(\theta) = \argmin_{\theta} \frac{1}{N} \sum_{i=1}^{N} \ell\big(y_i, f_{\theta}(x_i)\big)
{% end %}

The BCE per-sample loss satisfies the same well-behaved-loss properties as the squared error.

{% mathblock(kind="proposition", name="Properties of the BCE loss", id="bce-properties") %}
Let $\ell(y, \hat{p}) = -y \log \hat{p} - (1 - y) \log (1 - \hat{p})$ for $y \in \\{0, 1\\}$ and $\hat{p} \in (0, 1)$. Then $\ell$ is:

1. non-negative,
2. zero if and only if $\hat{p} = y$,
3. infinitely differentiable on $(0, 1)$, and
4. strictly convex in $\hat{p}$.
   {% end %}

{% mathblock(kind="proof", name="", id="bce-properties-proof") %}
**(i)** Since $\hat{p} \in (0, 1)$, both $\log \hat{p}$ and $\log(1 - \hat{p})$ are non-positive, so the negative sum is non-negative.

**(ii)** With $y = 1$, $\ell = -\log \hat{p} = 0 \iff \hat{p} = 1 = y$. With $y = 0$, $\ell = -\log(1 - \hat{p}) = 0 \iff \hat{p} = 0 = y$.

**(iii)** Both $\log \hat{p}$ and $\log(1 - \hat{p})$ are smooth on $(0, 1)$, and so is their linear combination.

**(iv)** The first and second derivatives are:
$$\frac{\partial \ell}{\partial \hat{p}} = -\frac{y}{\hat{p}} + \frac{1 - y}{1 - \hat{p}}, \qquad \frac{\partial^2 \ell}{\partial \hat{p}^2} = \frac{y}{\hat{p}^2} + \frac{1 - y}{(1 - \hat{p})^2}.$$
For $y \in \\{0, 1\\}$ exactly one of the two terms is non-zero and strictly positive on $(0, 1)$, so the second derivative is strictly positive everywhere. Hence $\ell$ is strictly convex in $\hat{p}$. $\square$
{% end %}

### Properties of the Sigmoid

Two identities will be used throughout the rest of this post. They are worth committing to memory.

{% mathblock(kind="proposition", name="Symmetry of the sigmoid", id="sigmoid-symmetry-prop") %}
For all $z \in \mathbb{R}$, $\sigma(-z) = 1 - \sigma(z)$.
{% end %}

{% mathblock(kind="proof", name="", id="sigmoid-symmetry-proof") %}
Starting from the definition and multiplying numerator and denominator by $e^{-z}$:

$$\sigma(-z) = \frac{1}{1 + e^{z}} = \frac{1}{1 + e^{z}} \cdot \frac{e^{-z}}{e^{-z}} = \frac{e^{-z}}{e^{-z} + 1}.$$

Now write $1 - \sigma(z)$ over the common denominator $1 + e^{-z}$:

$$1 - \sigma(z) = \frac{1 + e^{-z}}{1 + e^{-z}} - \frac{1}{1 + e^{-z}} = \frac{e^{-z}}{1 + e^{-z}}.$$

The two expressions are equal. $\square$
{% end %}

{% mathblock(kind="proposition", name="Derivative of the sigmoid", id="sigmoid-derivative-prop") %}
For all $z \in \mathbb{R}$, $\sigma'(z) = \sigma(z)\,(1 - \sigma(z))$.
{% end %}

{% mathblock(kind="proof", name="", id="sigmoid-derivative-proof") %}
Write $\sigma(z) = (1 + e^{-z})^{-1}$ and apply the chain rule. With $u(z) = 1 + e^{-z}$ and $u'(z) = -e^{-z}$:

$$\sigma'(z) = -u(z)^{-2} \cdot u'(z) = -(1 + e^{-z})^{-2} \cdot (-e^{-z}) = \frac{e^{-z}}{(1 + e^{-z})^2}.$$

Split this into a product of two factors, both of which are sigmoids in disguise:

$$\sigma'(z) = \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}} = \sigma(z) \cdot \big(1 - \sigma(z)\big),$$

where the second factor was identified via {{ eqref(id="sigmoid-symmetry") }} (the proof above showed $1 - \sigma(z) = e^{-z}/(1 + e^{-z})$). $\square$
{% end %}

For reference, the two identities again in display form:

{% equation(id="sigmoid-symmetry") %}
\sigma(-z) = 1 - \sigma(z)
{% end %}

{% equation(id="sigmoid-derivative") %}
\sigma'(z) = \sigma(z)\,\big(1 - \sigma(z)\big)
{% end %}

### Point Estimation

Just as in the regression case, our model no longer predicts $y$ directly: it predicts the parameter $p$ of a distribution over $y$. At inference time we want a single class label, so we again take the **mode** of the predicted distribution as the point estimate.

{% mathblock(kind="proposition", name="Mode of the Bernoulli distribution", id="bernoulli-mode") %}
Let $y \sim \mathrm{Bernoulli}(p)$ with $p \in (0, 1)$. Then $\argmax\_{y \in \\{0, 1\\}} Pr(y \mid p) = \mathbf{1}\\{p \geq 0.5\\}$.
{% end %}

{% mathblock(kind="proof", name="", id="bernoulli-mode-proof") %}
$Pr(y = 1 \mid p) = p$ and $Pr(y = 0 \mid p) = 1 - p$. The argmax picks $y = 1$ exactly when $p \geq 1 - p$, i.e., when $p \geq 0.5$. $\square$
{% end %}

For the logistic model this means thresholding the predicted probability at $0.5$, or equivalently (since $\sigma(z) \geq 0.5 \iff z \geq 0$) thresholding the linear score $\theta\_0 + \theta\_1 x$ at zero. The decision boundary is therefore a hyperplane in input space, exactly as in linear classification.

### No Closed-Form Solution

In the regression case, setting the gradient of the cost to zero yielded a linear system in $\theta$ with a clean closed-form solution. Let us repeat the exercise for BCE and see what changes.

{% mathblock(kind="proposition", name="Per-parameter gradient of the BCE cost", id="bce-grad-prop") %}
Let $J(\theta) = \frac{1}{N} \sum\_i \ell(y\_i, p\_i)$ with $p\_i = \sigma(\theta\_0 + \theta\_1 x\_i)$ and $\ell(y, \hat{p}) = -y \log \hat{p} - (1-y) \log (1 - \hat{p})$. Then for $j \in \\{0, 1\\}$, with the convention $x\_{i,0} \equiv 1$ and $x\_{i,1} = x\_i$,
$$\frac{\partial J}{\partial \theta\_j} = \frac{1}{N} \sum\_{i=1}^N (p\_i - y\_i)\, x\_{i,j}.$$
{% end %}

{% mathblock(kind="proof", name="", id="bce-grad-proof") %}
Apply the chain rule, $\partial \ell\_i / \partial \theta\_j = (\partial \ell\_i / \partial p\_i) \cdot (\partial p\_i / \partial \theta\_j)$, and compute the two factors separately.

**Step 1. Derivative of the loss with respect to $p\_i$.** Differentiating $\ell\_i = -y\_i \log p\_i - (1 - y\_i) \log (1 - p\_i)$ term by term,

$$\frac{\partial \ell\_i}{\partial p\_i} = -\frac{y\_i}{p\_i} + \frac{1 - y\_i}{1 - p\_i}.$$

Put both terms over the common denominator $p\_i (1 - p\_i)$:

$$\frac{\partial \ell\_i}{\partial p\_i} = \frac{-y\_i (1 - p\_i) + (1 - y\_i)\, p\_i}{p\_i (1 - p\_i)} = \frac{-y\_i + y\_i p\_i + p\_i - y\_i p\_i}{p\_i (1 - p\_i)} = \frac{p\_i - y\_i}{p\_i (1 - p\_i)}.$$

**Step 2. Derivative of $p\_i$ with respect to $\theta\_j$.** Let $z\_i = \theta\_0 + \theta\_1 x\_i$ so that $p\_i = \sigma(z\_i)$. By the chain rule and {{ eqref(id="sigmoid-derivative") }},

$$\frac{\partial p\_i}{\partial \theta\_j} = \sigma'(z\_i) \cdot \frac{\partial z\_i}{\partial \theta\_j} = p\_i (1 - p\_i) \cdot x\_{i,j}.$$

**Step 3. Combine.** Multiplying the two factors, the $p\_i (1 - p\_i)$ in the numerator of Step 2 cancels exactly with the same factor in the denominator of Step 1:

$$\frac{\partial \ell\_i}{\partial \theta\_j} = \frac{p\_i - y\_i}{p\_i (1 - p\_i)} \cdot p\_i (1 - p\_i) \cdot x\_{i,j} = (p\_i - y\_i)\, x\_{i,j}.$$

Averaging over the $N$ training samples gives the claim. $\square$
{% end %}

For reference,

{% equation(id="bce-grad") %}
\frac{\partial J}{\partial \theta_j} = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)\, x_{i,j}
{% end %}

This has the same algebraic shape as the gradient of the squared-error cost: a sum of (prediction $-$ target) weighted by the input. The structural difference is that $p\_i = \sigma(\theta\_0 + \theta\_1 x\_i)$ is a non-linear function of $\theta$, so setting $\partial J / \partial \theta\_j = 0$ produces a transcendental system with no closed-form solution. We have to optimise iteratively.

## Multivariate Logistic Regression

We now generalise to a vector input $\mathbf{x} \in \mathbb{R}^D$, while keeping the output $y \in \\{0, 1\\}$ binary. As in the regression case, we absorb the bias by augmenting each input with a leading $1$, so $\tilde{\mathbf{x}} = (1, x\_{1}, \ldots, x\_{D})^{\top} \in \mathbb{R}^{D+1}$ and $\boldsymbol{\theta} = (\theta\_{0}, \theta\_{1}, \ldots, \theta\_{D})^{\top} \in \mathbb{R}^{D+1}$. The model becomes:

{% equation(id="logistic-model-multi") %}
f_{\boldsymbol{\theta}}(\mathbf{x}) = \sigma(\boldsymbol{\theta}^{\top} \tilde{\mathbf{x}}) = \frac{1}{1 + \exp(-\boldsymbol{\theta}^{\top} \tilde{\mathbf{x}})}
{% end %}

### Matrix Form of the Cost Function

Stack the $N$ augmented inputs row-wise into the design matrix $\mathbf{X} \in \mathbb{R}^{N \times (D+1)}$ and the labels into $\mathbf{y} \in \\{0, 1\\}^{N}$, exactly as in the [regression note](/blog/linear-regression/). The vector of predicted probabilities is the elementwise sigmoid of the linear scores:

{% equation(id="probs-vector") %}
\mathbf{p} = \sigma(\mathbf{X}\boldsymbol{\theta}) \in (0, 1)^{N}
{% end %}

Letting $\log$ act elementwise and writing $\mathbf{1}$ for the vector of ones, the BCE cost in matrix form is:

{% equation(id="bce-cost-multi") %}
J(\boldsymbol{\theta}) = -\frac{1}{N} \Big[\, \mathbf{y}^{\top} \log \mathbf{p} + (\mathbf{1} - \mathbf{y})^{\top} \log (\mathbf{1} - \mathbf{p}) \,\Big]
{% end %}

### Gradient and Hessian

The per-parameter gradient {{ eqref(id="bce-grad") }} aggregates into a clean matrix expression. It is the same prediction-error sum, vectorised:

{% equation(id="bce-grad-matrix") %}
\nabla_{\boldsymbol{\theta}} J = \frac{1}{N}\, \mathbf{X}^{\top} (\mathbf{p} - \mathbf{y})
{% end %}

{% mathblock(kind="proposition", name="Hessian of the BCE cost", id="bce-hessian-prop") %}
With $\mathbf{p} = \sigma(\mathbf{X} \boldsymbol{\theta})$ elementwise and $\mathbf{S} = \mathrm{diag}\big(p\_1 (1 - p\_1), \ldots, p\_N (1 - p\_N)\big)$,
$$\mathbf{H} = \nabla\_{\boldsymbol{\theta}}^{2} J = \frac{1}{N}\, \mathbf{X}^{\top} \mathbf{S}\, \mathbf{X}.$$
{% end %}

{% mathblock(kind="proof", name="", id="bce-hessian-proof") %}
Differentiate the gradient {{ eqref(id="bce-grad-matrix") }} once more. The dependence on $\boldsymbol{\theta}$ enters through $\mathbf{p}$ alone, so

$$\nabla\_{\boldsymbol{\theta}}^{2} J = \frac{1}{N}\, \mathbf{X}^{\top}\, \frac{\partial \mathbf{p}}{\partial \boldsymbol{\theta}^{\top}}.$$

The $(i, j)$ entry of the Jacobian $\partial \mathbf{p} / \partial \boldsymbol{\theta}^{\top}$ is $\partial p\_i / \partial \theta\_j$, which by Step 2 of the previous proof equals $p\_i (1 - p\_i)\, x\_{i,j}$. In matrix form, with $\mathbf{S}$ collecting the per-sample weights along the diagonal,

$$\frac{\partial \mathbf{p}}{\partial \boldsymbol{\theta}^{\top}} = \mathbf{S}\, \mathbf{X},$$

since multiplying $\mathbf{X}$ on the left by the diagonal matrix $\mathbf{S}$ scales each row $i$ of $\mathbf{X}$ by $p\_i (1 - p\_i)$. Substituting,

$$\mathbf{H} = \frac{1}{N}\, \mathbf{X}^{\top}\, \mathbf{S}\, \mathbf{X}. \qquad \square$$
{% end %}

The diagonal weight matrix $\mathbf{S}$ encodes the local curvature of the sigmoid: it is largest near $p\_i = 0.5$ (where the model is most uncertain and the score is most sensitive to changes in $\theta$) and shrinks towards zero as $p\_i$ approaches $0$ or $1$.

{% mathblock(kind="proposition", name="Convexity of the BCE cost", id="bce-convex") %}
The cost $J(\boldsymbol{\theta})$ in {{ eqref(id="bce-cost-multi") }} is convex in $\boldsymbol{\theta}$, and strictly convex when $\mathbf{X}$ has full column rank and no $p\_i \in \\{0, 1\\}$.
{% end %}

{% mathblock(kind="proof", name="", id="bce-convex-proof") %}
For any $\mathbf{v} \in \mathbb{R}^{D+1}$,
$$\mathbf{v}^{\top} \mathbf{H}\, \mathbf{v} = \frac{1}{N}\, \mathbf{v}^{\top} \mathbf{X}^{\top} \mathbf{S}\, \mathbf{X} \mathbf{v} = \frac{1}{N}\, \lVert \mathbf{S}^{1/2} \mathbf{X} \mathbf{v} \rVert_{2}^{2} \geq 0,$$
since $\mathbf{S}$ has non-negative diagonal entries and admits a real square root $\mathbf{S}^{1/2}$. Thus $\mathbf{H}$ is positive semidefinite and $J$ is convex. If $\mathbf{X}$ has full column rank and every $p\_i \in (0, 1)$ (equivalently, every diagonal entry of $\mathbf{S}$ is strictly positive), then $\mathbf{S}^{1/2} \mathbf{X} \mathbf{v} = \mathbf{0}$ implies $\mathbf{X} \mathbf{v} = \mathbf{0}$ implies $\mathbf{v} = \mathbf{0}$, the Hessian is positive definite, and $J$ is strictly convex with a unique minimiser. $\square$
{% end %}

### Numerical Solution

Convexity guarantees a unique global minimum (under the conditions above), but unlike the regression case there is no closed-form expression for it. We have to iterate.

#### Gradient Descent

The simplest option is to take steps in the direction of steepest descent:

{% equation(id="gradient-descent") %}
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_{t} - \eta\, \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}_t) = \boldsymbol{\theta}_t - \frac{\eta}{N}\, \mathbf{X}^{\top}(\mathbf{p}_t - \mathbf{y})
{% end %}

with a learning rate $\eta > 0$.

{% mathblock(kind="note", name="Cost per gradient descent step", id="gd-cost") %}
Each iteration takes $O(N D)$ time and $O(N D)$ memory. The time cost comes from two matrix-vector products against the design matrix $\mathbf{X}$. First, the forward pass $\mathbf{p}\_t = \sigma(\mathbf{X} \boldsymbol{\theta}\_t)$ (a matrix-vector product at $O(N D)$ flops, followed by an elementwise sigmoid at $O(N)$). Then the gradient $\mathbf{X}^{\top}(\mathbf{p}\_t - \mathbf{y})$ is another matrix-vector product at $O(N D)$. The parameter update itself is $O(D)$, so the design-matrix passes dominate.

For memory, the design matrix $\mathbf{X}$ at $N \times (D+1)$ entries is the only $O(N D)$ object. Everything else (the vectors $\mathbf{p}\_t$, $\mathbf{y}$, $\boldsymbol{\theta}\_t$, and the gradient) is $O(N) + O(D)$. No second-order information is stored.
{% end %}

Each step is cheap and uses only first-order information. The convergence rate depends on the conditioning of the Hessian, and since $\mathbf{S}$ shrinks the curvature in regions where the model is confident, gradient descent can crawl along nearly-flat directions of the loss surface.

#### Newton-Raphson and IRLS

The natural second-order alternative is the Newton-Raphson update, which uses the Hessian to rescale the step:

{% equation(id="newton-update") %}
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_{t} - \mathbf{H}_{t}^{-1}\, \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}_{t})
{% end %}

Let us derive the IRLS form step by step. Substitute {{ eqref(id="bce-grad-matrix") }} and {{ eqref(id="bce-hessian") }} into {{ eqref(id="newton-update") }} (the $1/N$ factors in the gradient and Hessian cancel against each other when we form $\mathbf{H}^{-1} \nabla J$):

**Step 1.** Plug in:

$$\boldsymbol{\theta}\_{t+1} = \boldsymbol{\theta}\_{t} - (\mathbf{X}^{\top} \mathbf{S}\_{t}\, \mathbf{X})^{-1}\, \mathbf{X}^{\top} (\mathbf{p}\_{t} - \mathbf{y}).$$

**Step 2.** Factor out $(\mathbf{X}^{\top} \mathbf{S}\_{t}\, \mathbf{X})^{-1}$ from both terms. To do so, rewrite $\boldsymbol{\theta}\_{t}$ as $(\mathbf{X}^{\top} \mathbf{S}\_{t}\, \mathbf{X})^{-1} (\mathbf{X}^{\top} \mathbf{S}\_{t}\, \mathbf{X})\, \boldsymbol{\theta}\_{t}$:

$$\boldsymbol{\theta}\_{t+1} = (\mathbf{X}^{\top} \mathbf{S}\_{t}\, \mathbf{X})^{-1}\, \Big[\, \mathbf{X}^{\top} \mathbf{S}\_{t}\, \mathbf{X}\, \boldsymbol{\theta}\_{t} - \mathbf{X}^{\top} (\mathbf{p}\_{t} - \mathbf{y}) \,\Big].$$

**Step 3.** Pull $\mathbf{X}^{\top}$ out as a left factor inside the brackets, and force $\mathbf{S}\_{t}$ next to $(\mathbf{y} - \mathbf{p}\_{t})$ by inserting the identity $\mathbf{S}\_{t} \mathbf{S}\_{t}^{-1}$:

$$\boldsymbol{\theta}\_{t+1} = (\mathbf{X}^{\top} \mathbf{S}\_{t}\, \mathbf{X})^{-1}\, \mathbf{X}^{\top} \mathbf{S}\_{t}\, \Big[\, \mathbf{X}\, \boldsymbol{\theta}\_{t} + \mathbf{S}\_{t}^{-1} (\mathbf{y} - \mathbf{p}\_{t}) \,\Big].$$

This is well-defined whenever no $p\_{i,t} \in \\{0, 1\\}$, since then every diagonal entry of $\mathbf{S}\_{t}$ is strictly positive and $\mathbf{S}\_{t}^{-1}$ exists.

**Step 4.** Define the **working response**

{% equation(id="working-response") %}
\mathbf{z}_{t} = \mathbf{X}\, \boldsymbol{\theta}_{t} + \mathbf{S}_{t}^{-1} (\mathbf{y} - \mathbf{p}_{t})
{% end %}

so the update collapses to:

{% equation(id="irls-update") %}
\boldsymbol{\theta}_{t+1} = (\mathbf{X}^{\top} \mathbf{S}_{t}\, \mathbf{X})^{-1}\, \mathbf{X}^{\top} \mathbf{S}_{t}\, \mathbf{z}_{t}
{% end %}

This is exactly the [normal equations](/blog/linear-regression/) of a **weighted least-squares** problem with design matrix $\mathbf{X}$, target $\mathbf{z}\_t$, and per-sample weights $\mathbf{S}\_t$ (i.e., the closed-form solution to $\min\_{\boldsymbol{\theta}} \lVert \mathbf{S}\_t^{1/2} (\mathbf{z}\_t - \mathbf{X} \boldsymbol{\theta}) \rVert\_2^2$). Each Newton step on the BCE cost is therefore one weighted least-squares solve, with both the weights and the target re-derived from the current $\boldsymbol{\theta}\_t$. This is the **iteratively reweighted least squares** (IRLS) algorithm.

The working response has a nice interpretation. $\mathbf{X} \boldsymbol{\theta}\_t$ is the current vector of linear scores, and $\mathbf{S}\_t^{-1} (\mathbf{y} - \mathbf{p}\_t)$ is a Newton-style correction in score-space. We divide the residual $\mathbf{y} - \mathbf{p}\_t$ by the local sigmoid slope $p\_i (1 - p\_i)$ to convert "off by this much in probability" into "off by this much in score". IRLS then solves a regression problem against this corrected target.

In practice one does not form $(\mathbf{X}^{\top} \mathbf{S}\, \mathbf{X})^{-1}$ explicitly. Each iteration is solved via a [QR factorisation](/blog/linear-regression/) of the rescaled design matrix $\mathbf{S}^{1/2} \mathbf{X}$, exactly as in the regression case (with the same $\kappa(\mathbf{X})^{2}$ argument against the Gram matrix), but applied per Newton step.

#### L-BFGS

IRLS is elegant but expensive.

{% mathblock(kind="note", name="Cost per IRLS step", id="irls-cost") %}
Each iteration takes $O(N D^2 + D^3)$ time and $O(N D + D^2)$ memory. Two operations dominate the time. First, forming the weighted Gram matrix $\mathbf{X}^{\top} \mathbf{S}\_t \mathbf{X}$ requires applying the diagonal $\mathbf{S}\_t$ to $\mathbf{X}$ row-wise at $O(N D)$ flops, then a Gram product of a $(D+1) \times N$ matrix with an $N \times (D+1)$ matrix at $O(N D^2)$. Second, solving the resulting $(D+1) \times (D+1)$ dense linear system via Cholesky or QR costs $O(D^3)$, which becomes the dominant term once $D \gtrsim \sqrt{N}$. Computing $\mathbf{p}\_t$, the residual, and the right-hand side $\mathbf{X}^{\top} \mathbf{S}\_t \mathbf{z}\_t$ adds another $O(N D)$, a lower-order contribution.

Memory splits between the design matrix $\mathbf{X}$ at $O(N D)$ and the dense Gram matrix at $O(D^2)$. The Gram matrix dominates once $D \gtrsim \sqrt{N}$.
{% end %}

For $D$ in the tens of thousands (text classification, genomics, any setting where features outnumber a comfortable Gram matrix), neither time nor memory is acceptable. The $D^3$ flop term and the $D^2$ dense matrix become the bottleneck even for a single iteration. **L-BFGS** (limited-memory Broyden-Fletcher-Goldfarb-Shanno) is the standard alternative and is what production solvers reach for first.{% sidenote(id="lbfgs-default") %}It is the default solver for `LogisticRegression` in scikit-learn and the workhorse behind most generalised linear model packages.{% end %}

L-BFGS belongs to the **quasi-Newton** family. It imitates Newton's method by maintaining an implicit approximation $\mathbf{B}\_t \approx \mathbf{H}\_t^{-1}$ of the inverse Hessian, but constructs it from gradient differences alone (no second derivatives are ever computed). The full BFGS update would store and update a dense $(D+1) \times (D+1)$ matrix, which defeats the purpose. The "limited-memory" variant keeps only the last $m$ pairs of curvature information $\\{(\mathbf{s}\_k, \mathbf{y}\_k)\\}\_{k=t-m}^{t-1}$, where:

{% equation(id="lbfgs-pairs") %}
\mathbf{s}_k = \boldsymbol{\theta}_{k+1} - \boldsymbol{\theta}_k, \qquad \mathbf{y}_k = \nabla J(\boldsymbol{\theta}_{k+1}) - \nabla J(\boldsymbol{\theta}_k)
{% end %}

The product $\mathbf{B}\_t \nabla J(\boldsymbol{\theta}\_t)$ is computed by a two-loop recursion that touches only those $m$ vector pairs and the current gradient, never forming $\mathbf{B}\_t$ explicitly.{% sidenote(id="lbfgs-two-loop") %}The standard reference is Nocedal and Wright, _Numerical Optimization_, Algorithm 7.4. Each iteration costs $O(m D)$ in time and $O(m D)$ in memory; a typical choice is $m \in [5, 20]$.{% end %} The resulting update is:

{% equation(id="lbfgs-update") %}
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_{t} - \alpha_t\, \mathbf{B}_t\, \nabla J(\boldsymbol{\theta}_t)
{% end %}

where the step size $\alpha\_t > 0$ is chosen by a line search satisfying the Wolfe conditions (this is what guarantees that the curvature pairs $(\mathbf{s}\_k, \mathbf{y}\_k)$ keep $\mathbf{B}\_t$ positive definite).

{% mathblock(kind="note", name="Cost per L-BFGS step", id="lbfgs-cost") %}
Each iteration takes $O(N D + m D)$ time and $O(N D + m D)$ memory. The gradient evaluation $\nabla J(\boldsymbol{\theta}\_t)$ is the same $O(N D)$ work as one gradient-descent step: a forward pass $\mathbf{p}\_t = \sigma(\mathbf{X} \boldsymbol{\theta}\_t)$ and a backward matrix-vector product $\mathbf{X}^{\top}(\mathbf{p}\_t - \mathbf{y})$. Applying $\mathbf{B}\_t$ to the gradient via the two-loop recursion touches each of the $m$ stored pairs $(\mathbf{s}\_k, \mathbf{y}\_k)$ twice (once in each loop) with a constant number of length-$D$ inner products and vector updates per touch, costing $O(m D)$ flops.

For memory, the design matrix $\mathbf{X}$ at $O(N D)$ dominates when $N$ is large. Otherwise the $m$ stored vector pairs at $2 m D$ entries set the floor. Crucially, no $D \times D$ matrix is ever materialised.
{% end %}

The trade-off against IRLS is clean. L-BFGS gives up the per-step quadratic convergence rate of Newton's method, but the $D^2$ memory term and the $D^3$ flop term both vanish, replaced by an $m D$ term with $m \approx 10$. This scales to $D$ in the millions where IRLS would be hopeless. For convex problems like BCE it converges superlinearly in practice, and the only inputs it needs from the user are the cost {{ eqref(id="bce-cost-multi") }} and its gradient {{ eqref(id="bce-grad-matrix") }}, both available in closed form.

#### Convergence Guarantees

All three methods admit clean convergence proofs once we establish two structural facts about the BCE cost: it is convex (already proven) and it is $L$-smooth, meaning its gradient is Lipschitz continuous.

{% mathblock(kind="proposition", name="Smoothness of the BCE cost", id="bce-smooth") %}
The BCE cost $J$ in {{ eqref(id="bce-cost-multi") }} is $L$-smooth with $L = \frac{1}{4N}\, \lambda\_{\max}(\mathbf{X}^{\top} \mathbf{X})$, i.e., for all $\boldsymbol{\theta}, \boldsymbol{\theta}' \in \mathbb{R}^{D+1}$,
$$\lVert \nabla J(\boldsymbol{\theta}) - \nabla J(\boldsymbol{\theta}') \rVert\_2 \leq L\, \lVert \boldsymbol{\theta} - \boldsymbol{\theta}' \rVert\_2.$$
{% end %}

{% mathblock(kind="proof", name="", id="bce-smooth-proof") %}
A twice-differentiable convex function is $L$-smooth iff $\nabla^2 J(\boldsymbol{\theta}) \preceq L\, \mathbf{I}$ for all $\boldsymbol{\theta}$, where $\preceq$ is the Loewner order. Pick the smallest such $L$.

By {{ eqref(id="bce-hessian") }}, $\mathbf{H} = \frac{1}{N}\, \mathbf{X}^{\top} \mathbf{S}\, \mathbf{X}$ with $\mathbf{S} = \mathrm{diag}(p\_i (1 - p\_i))$. The function $p \mapsto p(1 - p)$ on $(0, 1)$ attains its maximum $1/4$ at $p = 1/2$, so $\mathbf{S} \preceq \frac{1}{4}\, \mathbf{I}\_N$ uniformly in $\boldsymbol{\theta}$. Hence

$$\mathbf{H} = \frac{1}{N}\, \mathbf{X}^{\top} \mathbf{S}\, \mathbf{X} \preceq \frac{1}{4N}\, \mathbf{X}^{\top} \mathbf{X} \preceq \frac{1}{4N}\, \lambda\_{\max}(\mathbf{X}^{\top} \mathbf{X})\, \mathbf{I}\_{D+1}.$$

The smallest valid $L$ is therefore $L = \lambda\_{\max}(\mathbf{X}^{\top} \mathbf{X}) / (4N)$. $\square$
{% end %}

The cost is **not** strongly convex on separable data. There the MLE escapes to infinity along the separating direction and the cost has no minimiser. With an $\ell\_2$ penalty $\frac{\lambda}{2} \lVert \boldsymbol{\theta} \rVert\_2^2$ added (or with full column rank and bounded data on non-separable problems), the Hessian becomes uniformly positive definite and we additionally get $\mu$-strong convexity with $\mu \geq \lambda$.

{% mathblock(kind="proposition", name="Gradient descent convergence", id="gd-converge") %}
Run {{ eqref(id="gradient-descent") }} from any $\boldsymbol{\theta}\_0$ with constant step size $\eta = 1/L$.

1. (**Convex case.**) If a minimiser $\boldsymbol{\theta}^{\ast}$ exists,
   $$J(\boldsymbol{\theta}\_t) - J(\boldsymbol{\theta}^{\ast}) \leq \frac{L\, \lVert \boldsymbol{\theta}\_0 - \boldsymbol{\theta}^{\ast} \rVert\_2^2}{2\, t}.$$

2. (**Strongly convex case.**) If $J$ is additionally $\mu$-strongly convex (e.g., with an $\ell\_2$ penalty $\lambda > 0$ giving $\mu \geq \lambda$),
   $$\lVert \boldsymbol{\theta}\_t - \boldsymbol{\theta}^{\ast} \rVert\_2^2 \leq \big(1 - \mu/L\big)^t\, \lVert \boldsymbol{\theta}\_0 - \boldsymbol{\theta}^{\ast} \rVert\_2^2.$$
   {% end %}

{% mathblock(kind="proof", name="(sketch)", id="gd-converge-proof") %}
Both rates are textbook results for first-order methods on smooth convex (and strongly-convex) objectives. See Nesterov's _Lectures on Convex Optimization_, Theorems 2.1.14 and 2.1.15. The key ingredients are the **descent lemma**, valid for any $L$-smooth function,

$$J(\boldsymbol{\theta}\_{t+1}) \leq J(\boldsymbol{\theta}\_{t}) + \nabla J(\boldsymbol{\theta}\_t)^{\top} (\boldsymbol{\theta}\_{t+1} - \boldsymbol{\theta}\_t) + \frac{L}{2} \lVert \boldsymbol{\theta}\_{t+1} - \boldsymbol{\theta}\_t \rVert\_2^2,$$

which for the gradient step $\boldsymbol{\theta}\_{t+1} = \boldsymbol{\theta}\_t - \eta\, \nabla J(\boldsymbol{\theta}\_t)$ with $\eta = 1/L$ becomes $J(\boldsymbol{\theta}\_{t+1}) \leq J(\boldsymbol{\theta}\_t) - \frac{1}{2L} \lVert \nabla J(\boldsymbol{\theta}\_t) \rVert\_2^2$ (guaranteeing monotone decrease), together with convexity (resp. strong convexity) to lower-bound the gradient norm in terms of the suboptimality gap. The two inequalities telescope into the rates above. $\square$
{% end %}

So gradient descent converges sublinearly ($O(1/t)$) on the unregularised BCE and linearly on the ridge-regularised version, with rate controlled by the conditioning $L/\mu$.

{% mathblock(kind="proposition", name="Newton-Raphson / IRLS convergence", id="newton-converge") %}
Assume $\mathbf{X}$ has full column rank, the data is non-separable, and $\boldsymbol{\theta}^{\ast}$ is the (unique) minimiser. Let $\boldsymbol{\theta}\_t$ be the iterates of {{ eqref(id="newton-update") }}.

1. (**Local quadratic convergence.**) There exists $\rho > 0$ such that whenever $\lVert \boldsymbol{\theta}\_t - \boldsymbol{\theta}^{\ast} \rVert\_2 \leq \rho$,
   $$\lVert \boldsymbol{\theta}\_{t+1} - \boldsymbol{\theta}^{\ast} \rVert\_2 \leq C\, \lVert \boldsymbol{\theta}\_t - \boldsymbol{\theta}^{\ast} \rVert\_2^2,$$
   for a constant $C$ depending on the local Hessian conditioning.

2. (**Global convergence with damping.**) The damped Newton update $\boldsymbol{\theta}\_{t+1} = \boldsymbol{\theta}\_t - \alpha\_t\, \mathbf{H}\_t^{-1} \nabla J(\boldsymbol{\theta}\_t)$ with $\alpha\_t$ chosen by backtracking line search converges from any starting point, with the rate switching from linear (far from $\boldsymbol{\theta}^{\ast}$) to quadratic (in a neighbourhood of $\boldsymbol{\theta}^{\ast}$).
   {% end %}

{% mathblock(kind="proof", name="(sketch)", id="newton-converge-proof") %}
The local quadratic rate is the classical Newton-Kantorovich theorem: in a neighbourhood where $\mathbf{H}$ is Lipschitz continuous and bounded away from singular, Newton's method achieves $\lVert \boldsymbol{\theta}\_{t+1} - \boldsymbol{\theta}^{\ast} \rVert \leq C \lVert \boldsymbol{\theta}\_t - \boldsymbol{\theta}^{\ast} \rVert^2$. See Boyd and Vandenberghe, _Convex Optimization_, §9.5.

The global statement uses an additional structural fact: the BCE cost is **self-concordant** (after a smooth reparameterisation, or directly for the regularised version), meaning $|\nabla^3 J[\boldsymbol{\theta}](\mathbf{v}, \mathbf{v}, \mathbf{v})| \leq 2\, (\mathbf{v}^{\top} \mathbf{H} \mathbf{v})^{3/2}$. For self-concordant convex functions, damped Newton with backtracking line search achieves global convergence with a quadratic local phase; see _Convex Optimization_, §9.6. $\square$
{% end %}

In practice, IRLS without damping diverges if started far enough from the optimum or on poorly conditioned problems, which is why production solvers always wrap it in a line search.

{% mathblock(kind="proposition", name="L-BFGS convergence", id="lbfgs-converge") %}
Run L-BFGS with a Wolfe line search on a strongly convex, $L$-smooth objective $J$. Then $\boldsymbol{\theta}\_t \to \boldsymbol{\theta}^{\ast}$ globally, and the convergence is **R-superlinear**: $\lVert \boldsymbol{\theta}\_t - \boldsymbol{\theta}^{\ast} \rVert\_2 \leq c\_t\, \lVert \boldsymbol{\theta}\_{t-1} - \boldsymbol{\theta}^{\ast} \rVert\_2$ with $c\_t \to 0$.
{% end %}

{% mathblock(kind="proof", name="(sketch)", id="lbfgs-converge-proof") %}
The Wolfe conditions ensure each curvature pair $(\mathbf{s}\_k, \mathbf{y}\_k)$ satisfies $\mathbf{s}\_k^{\top} \mathbf{y}\_k > 0$, which keeps the implicit inverse-Hessian approximation $\mathbf{B}\_t$ symmetric positive definite. Combined with strong convexity, this yields a uniform bound on the eigenvalues of $\mathbf{B}\_t$, so each step is a descent direction with sufficient length to satisfy the Zoutendijk condition $\sum\_t (\nabla J(\boldsymbol{\theta}\_t)^{\top} \mathbf{B}\_t \nabla J(\boldsymbol{\theta}\_t))^2 / \lVert \mathbf{B}\_t \nabla J(\boldsymbol{\theta}\_t) \rVert\_2^2 < \infty$, forcing $\nabla J(\boldsymbol{\theta}\_t) \to \mathbf{0}$. That is global convergence. The R-superlinear rate is more delicate; the original argument is in Liu and Nocedal (1989), and a textbook treatment is in Nocedal and Wright, _Numerical Optimization_, §7.2. The intuition is that as $\boldsymbol{\theta}\_t \to \boldsymbol{\theta}^{\ast}$, the curvature pairs increasingly capture the true Hessian along the relevant subspace, and $\mathbf{B}\_t$ approximates $\mathbf{H}^{-1}(\boldsymbol{\theta}^{\ast})$ well enough that each step approaches a true Newton step. $\square$
{% end %}

L-BFGS without strong convexity (e.g., unregularised BCE on full-rank, non-separable data) still converges to the optimum, but only the global property holds. The superlinear rate requires strong convexity. This is one of the practical reasons to add even a tiny $\ell\_2$ penalty.

{% mathblock(kind="warning", name="Numerical pitfalls", id="logreg-warning") %}
Two issues bite in practice. **Sigmoid saturation:** when $|z|$ is large, $\sigma(z)$ is numerically indistinguishable from $0$ or $1$ and the naive BCE loss returns $\log 0 = -\infty$. The fix is to combine the sigmoid and the log into a single stable expression. For example, $\log \sigma(z) = -\log(1 + e^{-z})$ computed via `log1p` and a sign-aware branch, or the standard `binary_cross_entropy_with_logits` primitive that takes the linear score directly. **Linearly separable data:** when the two classes can be separated by a hyperplane, the MLE is unbounded. Pushing $\lVert \boldsymbol{\theta} \rVert \to \infty$ along the separating direction drives the loss to zero, the Hessian becomes singular at the limit, and IRLS diverges. The standard remedy is to add an $\ell\_{2}$ penalty $\frac{\lambda}{2} \lVert \boldsymbol{\theta} \rVert\_{2}^{2}$ to the cost (ridge, weight decay), which keeps the Hessian positive definite for any $\lambda > 0$.
{% end %}

## Multinomial Logistic Regression

We now generalise to $K > 2$ classes. The output $y$ is one of $K$ mutually exclusive categories, encoded as a one-hot vector $\mathbf{y} \in \\{0, 1\\}^{K}$ with $\sum\_k y\_k = 1$. We replace the single parameter vector with a parameter matrix $\mathbf{W} \in \mathbb{R}^{(D+1) \times K}$ whose $k$-th column $\mathbf{w}\_k$ scores class $k$, and we replace the sigmoid with the [softmax function](/blog/softmax-function/) to map the $K$ scores into a probability distribution:

{% equation(id="softmax-model") %}
p_k = f_{\mathbf{W}}(\tilde{\mathbf{x}})\*k = \mathrm{softmax}(\mathbf{W}^{\top} \tilde{\mathbf{x}})_k = \frac{\exp(\mathbf{w}_k^{\top} \tilde{\mathbf{x}})}{\sum_{j=1}^{K} \exp(\mathbf{w}_j^{\top} \tilde{\mathbf{x}})}
{% end %}

Each $p\_k \in (0, 1)$ and $\sum\_k p\_k = 1$ by construction.

### The Categorical Cross-Entropy

The Bernoulli is replaced by the categorical distribution, whose probability mass function on a one-hot $\mathbf{y}$ with parameters $\mathbf{p} = (p\_1, \ldots, p\_K)$ is:

{% equation(id="categorical-pmf") %}
Pr(\mathbf{y} \mid \mathbf{x}, \mathbf{W}) = \prod_{k=1}^{K} p_k^{y_k}
{% end %}

Only the term corresponding to the true class survives, since $y\_k = 0$ for all other $k$ and any factor $p\_j^{0} = 1$. Substituting into the NLL {{ eqref(id="nll") }} gives the **categorical cross-entropy** cost:

{% equation(id="cce-cost") %}
J(\mathbf{W}) = -\frac{1}{N} \sum_{i=1}^{N}\, \sum_{k=1}^{K}\, y_{i,k} \log p_{i,k}
{% end %}

### Gradient

We mirror the binary derivation: differentiate one sample's loss with respect to one parameter column $\mathbf{w}\_j$ via the chain rule, then average over samples.

{% mathblock(kind="proposition", name="Softmax Jacobian", id="softmax-jacobian") %}
Let $\mathbf{z} \in \mathbb{R}^K$ and $p\_k = \mathrm{softmax}(\mathbf{z})\_k = e^{z\_k} / \sum\_\ell e^{z\_\ell}$. Then for all $k, j$,
$$\frac{\partial p\_k}{\partial z\_j} = p\_k\,(\delta\_{kj} - p\_j),$$
where $\delta\_{kj}$ is the Kronecker delta.
{% end %}

{% mathblock(kind="proof", name="", id="softmax-jacobian-proof") %}
Write $S = \sum\_\ell e^{z\_\ell}$ so that $p\_k = e^{z\_k} / S$. The denominator $S$ depends on every $z\_j$ through $\partial S / \partial z\_j = e^{z\_j}$. By the quotient rule,

$$\frac{\partial p\_k}{\partial z\_j} = \frac{(\partial e^{z\_k}/\partial z\_j)\, S - e^{z\_k}\, (\partial S / \partial z\_j)}{S^2} = \frac{\delta\_{kj}\, e^{z\_k}\, S - e^{z\_k}\, e^{z\_j}}{S^2}.$$

Split into two fractions and identify each as a softmax probability:

$$\frac{\partial p\_k}{\partial z\_j} = \delta\_{kj}\, \frac{e^{z\_k}}{S} - \frac{e^{z\_k}}{S} \cdot \frac{e^{z\_j}}{S} = \delta\_{kj}\, p\_k - p\_k\, p\_j = p\_k\,(\delta\_{kj} - p\_j). \qquad \square$$
{% end %}

{% mathblock(kind="proposition", name="Gradient of the categorical cross-entropy", id="cce-grad-prop") %}
For a single sample with one-hot label $\mathbf{y}$, scores $\mathbf{z} = \mathbf{W}^{\top} \tilde{\mathbf{x}}$, and probabilities $\mathbf{p} = \mathrm{softmax}(\mathbf{z})$, let $\ell = -\sum\_k y\_k \log p\_k$. Then
$$\frac{\partial \ell}{\partial z\_j} = p\_j - y\_j, \qquad \nabla\_{\mathbf{w}\_j} \ell = (p\_j - y\_j)\, \tilde{\mathbf{x}}.$$
{% end %}

{% mathblock(kind="proof", name="", id="cce-grad-proof") %}
Apply the chain rule:

$$\frac{\partial \ell}{\partial z\_j} = -\sum\_{k=1}^{K} y\_k\, \frac{1}{p\_k}\, \frac{\partial p\_k}{\partial z\_j} = -\sum\_{k=1}^{K} \frac{y\_k}{p\_k}\, p\_k\,(\delta\_{kj} - p\_j) = -\sum\_{k=1}^{K} y\_k\,(\delta\_{kj} - p\_j),$$

using the softmax Jacobian above and cancelling $p\_k$ in numerator and denominator. Distribute the sum:

$$\frac{\partial \ell}{\partial z\_j} = -\sum\_k y\_k \delta\_{kj} + p\_j \sum\_k y\_k = -y\_j + p\_j \cdot 1 = p\_j - y\_j,$$

where the second equality uses $\sum\_k y\_k \delta\_{kj} = y\_j$ (only the $k = j$ term survives) and $\sum\_k y\_k = 1$ (the one-hot sums to one). Then since $z\_j = \mathbf{w}\_j^{\top} \tilde{\mathbf{x}}$, $\partial z\_j / \partial \mathbf{w}\_j = \tilde{\mathbf{x}}$, giving $\nabla\_{\mathbf{w}\_j} \ell = (p\_j - y\_j)\, \tilde{\mathbf{x}}$. $\square$
{% end %}

Stack the one-hot labels into $\mathbf{Y} \in \\{0, 1\\}^{N \times K}$ and the predicted probabilities into $\mathbf{P} \in (0, 1)^{N \times K}$ with $\mathbf{P} = \mathrm{softmax}(\mathbf{X} \mathbf{W})$ row-wise. Averaging the per-sample gradient above over the $N$ samples gives, for each column $\mathbf{w}\_k$,

{% equation(id="cce-grad-col") %}
\nabla_{\mathbf{w}_k} J = \frac{1}{N}\, \mathbf{X}^{\top} (\mathbf{P}_{:,k} - \mathbf{Y}_{:,k})
{% end %}

Stacking column-wise, the full gradient retains the same prediction-error shape as the binary case:

{% equation(id="cce-grad-matrix") %}
\nabla_{\mathbf{W}} J = \frac{1}{N}\, \mathbf{X}^{\top} (\mathbf{P} - \mathbf{Y})
{% end %}

### Identifiability

Adding the same vector $\mathbf{c} \in \mathbb{R}^{D+1}$ to every column of $\mathbf{W}$ shifts every score by the same constant, $\mathbf{w}\_k^{\top} \tilde{\mathbf{x}} + \mathbf{c}^{\top} \tilde{\mathbf{x}}$, which the softmax cancels exactly:

{% equation(id="softmax-shift") %}
\frac{\exp(\mathbf{w}_k^{\top} \tilde{\mathbf{x}} + \mathbf{c}^{\top} \tilde{\mathbf{x}})}{\sum_{j=1}^{K} \exp(\mathbf{w}_j^{\top} \tilde{\mathbf{x}} + \mathbf{c}^{\top} \tilde{\mathbf{x}})} = \frac{\exp(\mathbf{w}_k^{\top} \tilde{\mathbf{x}})}{\sum_{j=1}^{K} \exp(\mathbf{w}_j^{\top} \tilde{\mathbf{x}})}
{% end %}

So $\mathbf{W}$ is only identified up to a global column shift, and the unregularised cost has a $(D+1)$-dimensional flat direction along which the Hessian is singular. Two standard fixes: pin one class as a reference by fixing $\mathbf{w}\_K = \mathbf{0}$ (which recovers the binary model when $K = 2$), or add an $\ell\_2$ penalty $\frac{\lambda}{2} \lVert \mathbf{W} \rVert\_{F}^{2}$, which selects the unique minimum-norm solution and makes the Hessian positive definite.

### Optimisation

The optimisation story carries over essentially unchanged. The cost is convex in $\mathbf{W}$ (modulo the flat direction above), gradient descent works directly with {{ eqref(id="cce-grad-matrix") }}, and Newton-Raphson generalises to a block IRLS scheme on the $K(D+1)$-dimensional parameter vector. The same numerical pitfalls apply, with one addition. Compute the categorical cross-entropy from logits, not from softmax outputs, using the **log-sum-exp** identity $\log p\_k = z\_k - \log \sum\_j \exp z\_j$. This avoids both overflow (subtract $\max\_j z\_j$ first) and the underflow that destroys $\log p\_k$ when $p\_k$ is small. The standard `cross_entropy_with_logits` primitive does exactly this.
