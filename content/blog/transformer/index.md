+++
title = "Transformers"
date = 2026-05-13
description = "A short note on the Transformer architecture: scaled dot-product attention, multi-head attention, positional encodings, encoder and decoder blocks, and the family of attention variants that have grown around them."

[taxonomies]
tags = ["machine-learning", "deep-learning", "attention", "transformers"]
categories = ["notes"]

[extra]
math = true
+++

## From recurrence to attention

The [neural network](/blog/neural-network/) post built models that map a single fixed-size input vector to a single fixed-size output. Sequences (sentences, audio frames, time series) do not fit that mould: their length varies and their structure is sequential. The two pre-Transformer answers to this were recurrent networks (RNNs, LSTMs, GRUs) that consume the sequence one token at a time while carrying a hidden state, and one-dimensional convolutions that slide a fixed-width filter along the sequence. Both have a structural cost. Recurrence forces a strictly serial computation: the state at step $t$ depends on the state at step $t-1$, so an $N$-token sequence requires $N$ sequential steps and the gradient must traverse all of them on the backward pass, which both runs slowly on parallel hardware and decays exponentially through depth (the vanishing-gradient problem from the [neural network](/blog/neural-network/) post, restated in time). Convolutions parallelise across positions but communicate only within the kernel, so two tokens $k$ apart in the sequence can only influence each other after roughly $k / w$ stacked convolutions of width $w$, ie, the path length grows with the distance.

Attention, as introduced by Bahdanau, Cho, and Bengio for neural machine translation{{ reference(key="bahdanau2015attention") }} and refined by Luong, Pham, and Manning{{ reference(key="luong2015effective") }}, sidesteps both costs at once. Every output position is a learned weighted average of every input position, where the weights are computed from the inputs themselves. The path length between any two tokens is exactly one (every output sees every input directly), and the per-layer computation parallelises trivially across the sequence dimension because the weights at one position do not depend on those at any other. The only price is that the per-layer cost is quadratic in the sequence length: with $N$ tokens, computing all pairwise weights costs $O(N^2)$ time and memory. That price is what almost every modern variant of attention is, in one way or another, trying to negotiate down.

The Transformer of Vaswani et al.{{ reference(key="vaswani2017attention") }} took the additional step of removing recurrence and convolution entirely, building the entire sequence model out of attention plus a small position-wise feed-forward network. The resulting architecture is what now powers essentially every modern language model, vision transformer, speech model, and multimodal model. What follows is a dissection of every design decision in that paper, the algebra behind each one, and the family of variants that have grown around it.

## Scaled dot-product attention

The atomic operation of the Transformer takes three sets of vectors as input. A set of $N$ **queries** $\mathbf{Q} \in \mathbb{R}^{N \times d\_k}$, a set of $M$ **keys** $\mathbf{K} \in \mathbb{R}^{M \times d\_k}$, and a set of $M$ **values** $\mathbf{V} \in \mathbb{R}^{M \times d\_v}$. The interpretation is borrowed from information retrieval: each query is matched against every key to produce a similarity score, the scores are turned into a probability distribution over the keys, and the output for that query is the corresponding probability-weighted average of the values. In the symmetric case $N = M$ used in self-attention, the same set of vectors plays all three roles after three different learned projections.

{% mathblock(kind="definition", name="Scaled dot-product attention", id="sdpa") %}
$$\begin{aligned}
\mathrm{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V})
&= \mathrm{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d\_k}}\right) \mathbf{V},
\end{aligned}$$
with the softmax applied row-wise over the $M$ keys. The output has shape $\mathbb{R}^{N \times d\_v}$.
{% end %}

The product $\mathbf{Q}\mathbf{K}^\top \in \mathbb{R}^{N \times M}$ holds the raw similarity scores: entry $(i, j)$ is the inner product $\mathbf{q}\_i^\top \mathbf{k}\_j$ between the $i$-th query and the $j$-th key. Dividing by $\sqrt{d\_k}$ controls the variance of these scores. Applying the row-wise [softmax](/blog/softmax-function/) turns each row into a probability distribution over the $M$ keys, and right-multiplying by $\mathbf{V}$ takes the corresponding weighted average of value vectors.

Three design decisions are baked into this single line, and each one is worth pulling apart.

### Why the dot product

The original Bahdanau attention used an **additive** scoring function $a(\mathbf{q}, \mathbf{k}) = \mathbf{w}^\top \tanh(\mathbf{W}\_q \mathbf{q} + \mathbf{W}\_k \mathbf{k})$, ie, a tiny one-hidden-layer MLP applied to each query-key pair. This is more expressive than a plain inner product but disastrous for parallel hardware, since computing every pair requires an explicit double loop or an expanded $N \times M \times d\_k$ tensor. The dot product collapses the entire pairwise score table into a single matrix multiplication $\mathbf{Q}\mathbf{K}^\top$, which is exactly the operation modern accelerators are best at.{% sidenote(id="sdpa-vs-additive") %}Vaswani et al. report that additive attention slightly outperforms dot-product attention at small $d\_k$, but the gap closes (and reverses) at the scales used in real Transformers, while the throughput advantage of dot-product attention only grows.{% end %} The Transformer accepts the loss in expressivity per attention head and recovers it by stacking many heads and many layers.

### Why divide by √dₖ

This is the "scaled" half of the name and is the most easily missed design choice. The motivation is variance control under the [softmax](/blog/softmax-function/).

{% mathblock(kind="proposition", name="Variance of the unscaled dot product", id="dot-variance") %}
Let $\mathbf{q}, \mathbf{k} \in \mathbb{R}^{d\_k}$ have independent entries with mean zero and variance one. Then
$$\begin{aligned}
\mathbb{E}\!\left[\mathbf{q}^\top \mathbf{k}\right] &= 0, \\\\
\mathrm{Var}\!\left(\mathbf{q}^\top \mathbf{k}\right) &= d\_k.
\end{aligned}$$
{% end %}

{% mathblock(kind="proof", name="", id="dot-variance-proof") %}
Write $\mathbf{q}^\top \mathbf{k} = \sum\_{j=1}^{d\_k} q\_j k\_j$. By independence, $\mathbb{E}[q\_j k\_j] = \mathbb{E}[q\_j] \mathbb{E}[k\_j] = 0$, so the mean is zero. The summands are uncorrelated, so $\mathrm{Var}(\sum\_j q\_j k\_j) = \sum\_j \mathrm{Var}(q\_j k\_j)$. For the per-term variance, $\mathrm{Var}(q\_j k\_j) = \mathbb{E}[q\_j^2 k\_j^2] - \mathbb{E}[q\_j k\_j]^2 = \mathbb{E}[q\_j^2] \mathbb{E}[k\_j^2] - 0 = 1 \cdot 1 = 1$. Summing $d\_k$ such terms gives variance $d\_k$. $\square$
{% end %}

A typical $d\_k$ in a Transformer is $64$, so unscaled scores have standard deviation $8$. Feeding such large logits to the softmax pushes its output close to a one-hot vector, ie, the distribution concentrates on the single largest key. The gradient of the softmax in this regime is nearly zero in every direction except one, so attention freezes early in training and the optimiser has nothing to work with. Dividing by $\sqrt{d\_k}$ rescales the standard deviation back to $1$, keeping the softmax in its dynamic range where every key still contributes a non-negligible probability and the gradients flow to all of them.

### Why the softmax

Softmax is the canonical map from real scores to a probability distribution and was derived in the [softmax](/blog/softmax-function/) post as the unique exponential-family link that makes the score-to-distribution map invertible up to additive shifts. Two of its properties are load-bearing here. First, the output sums to one, so the attention output is a convex combination of value vectors and lives in the same affine span as $\mathbf{V}$. This is what lets the attention operation be interpreted as "look up a value, with soft selection". Second, the softmax is differentiable everywhere with the well-conditioned Jacobian computed in the [softmax](/blog/softmax-function/) post, so backpropagation through attention is numerically stable when combined with the log-sum-exp trick. Hard alternatives (argmax, sparsemax) sacrifice one or both of these.

{% mathblock(kind="note", name="Cost analysis (one attention call)", id="sdpa-cost") %}
Time is $O(N M d\_k + N M d\_v)$, dominated by the two large matrix products: $\mathbf{Q}\mathbf{K}^\top$ at $O(N M d\_k)$ flops and the multiplication of the $N \times M$ attention matrix by $\mathbf{V}$ at $O(N M d\_v)$. The softmax itself is a row-wise $O(N M)$ sweep, absorbed into the lower-order terms. Memory is $O(N M)$ for the attention matrix, which is the term that bites: at $N = M = 16{,}384$ in float32, a single attention matrix already eats $1$ GB of memory, before counting any other tensor. This single number is the entire reason for the existence of FlashAttention, sparse attention, and linear attention discussed later.
{% end %}

## Multi-head attention

A single attention call has a fixed amount of expressivity: it produces one weighted average per query, with weights driven by a single similarity geometry. Different relationships between tokens (syntactic dependencies, semantic similarity, positional proximity) plausibly want to be modelled with different similarity geometries, and forcing all of them through a single softmax averages them away.

The fix is **multi-head attention**: project the inputs into $h$ different subspaces, run an independent scaled dot-product attention in each subspace, and concatenate the results.

{% mathblock(kind="definition", name="Multi-head attention", id="mha") %}
For $h$ heads with per-head dimensions $d\_k$ and $d\_v$, define learned projections $\mathbf{W}\_i^Q \in \mathbb{R}^{d\_{\text{model}} \times d\_k}$, $\mathbf{W}\_i^K \in \mathbb{R}^{d\_{\text{model}} \times d\_k}$, $\mathbf{W}\_i^V \in \mathbb{R}^{d\_{\text{model}} \times d\_v}$ for $i = 1, \ldots, h$, and an output projection $\mathbf{W}^O \in \mathbb{R}^{h d\_v \times d\_{\text{model}}}$. Then
$$\begin{aligned}
\mathrm{head}\_i &= \mathrm{Attention}(\mathbf{X}\mathbf{W}\_i^Q,\, \mathbf{X}\mathbf{W}\_i^K,\, \mathbf{X}\mathbf{W}\_i^V), \\\\
\mathrm{MultiHead}(\mathbf{X}) &= \mathrm{Concat}(\mathrm{head}\_1, \ldots, \mathrm{head}\_h)\, \mathbf{W}^O.
\end{aligned}$$
The standard choice in the original paper is $d\_{\text{model}} = 512$, $h = 8$, $d\_k = d\_v = d\_{\text{model}} / h = 64$.
{% end %}

The deliberate choice $d\_k = d\_{\text{model}} / h$ keeps the total parameter count and total compute identical to a single attention call at full $d\_{\text{model}}$, while allowing the model to attend to information from $h$ different representation subspaces in parallel. Empirically the per-head attention patterns specialise: in a trained language Transformer, some heads track syntactic dependencies, others track coreference, others track positional offsets, others act as "no-op" heads that attend almost uniformly. This division of labour is the entire justification for stacking heads rather than just making a single attention head wider.

The output projection $\mathbf{W}^O$ is what mixes information across heads. Without it, the per-head outputs would be concatenated into disjoint slices of the next layer's input vector, never interacting. With it, each output coordinate is a learned linear combination of every head's contribution.

{% mathblock(kind="note", name="Multi-head as a structured projection", id="mha-as-projection") %}
A useful way to read multi-head attention is as a single attention operation in $\mathbb{R}^{d\_\text{model}}$ whose query-key product matrix is constrained to be block-diagonal across head boundaries, after a learned change of basis $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V$. The block structure is what gives each head its own similarity geometry, and the output projection $\mathbf{W}^O$ undoes the block structure so subsequent layers see a fully mixed representation again. Multi-head is, in this light, neither a sum nor a product of attentions but a structured factorisation of a single large attention.
{% end %}

## Positional encoding

There is one important property of attention that the discussion so far has carefully avoided.

{% mathblock(kind="proposition", name="Permutation equivariance of attention", id="perm-eq") %}
Let $\mathbf{P} \in \\{0, 1\\}^{N \times N}$ be a permutation matrix acting on the rows of the inputs. Then
$$\begin{aligned}
\mathrm{Attention}(\mathbf{P}\mathbf{Q}, \mathbf{P}\mathbf{K}, \mathbf{P}\mathbf{V})
&= \mathbf{P}\, \mathrm{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}).
\end{aligned}$$
{% end %}

{% mathblock(kind="proof", name="", id="perm-eq-proof") %}
The score matrix becomes $(\mathbf{P}\mathbf{Q})(\mathbf{P}\mathbf{K})^\top = \mathbf{P}\mathbf{Q}\mathbf{K}^\top \mathbf{P}^\top$, ie, both rows and columns are permuted. The row-wise softmax is itself row-equivariant and commutes with the row permutation: $\mathrm{softmax}(\mathbf{P}\mathbf{S}\mathbf{P}^\top) = \mathbf{P}\, \mathrm{softmax}(\mathbf{S})\, \mathbf{P}^\top$ when applied row-wise (the row permutation $\mathbf{P}$ reshuffles which row is normalised, and the column permutation $\mathbf{P}^\top$ reshuffles within each row before normalising; both pass through the elementwise exponential). Right-multiplying by $\mathbf{P}\mathbf{V}$ gives $\mathbf{P}\, \mathrm{softmax}(\mathbf{S})\, \mathbf{V}$, ie, the output is permuted in the same way as the inputs. $\square$
{% end %}

This is a problem for sequence modelling. If we shuffle the words in a sentence, attention produces a correspondingly shuffled output, but the underlying meaning of the sentence depends on the word order. To make the model order-aware, we have to inject the position of each token into the input itself.

The Transformer adds a fixed **positional encoding** vector to each token embedding before any attention layer. The chosen function is sinusoidal: for position $\mathrm{pos} \in \\{0, 1, \ldots, N-1\\}$ and dimension $i \in \\{0, 1, \ldots, d\_{\text{model}}-1\\}$,

{% equation(id="sinusoidal-pe") %}
\begin{aligned}
\mathrm{PE}_{\mathrm{pos},\, 2i} &= \sin\!\left(\frac{\mathrm{pos}}{10000^{2i/d_{\text{model}}}}\right), \\
\mathrm{PE}_{\mathrm{pos},\, 2i+1} &= \cos\!\left(\frac{\mathrm{pos}}{10000^{2i/d_{\text{model}}}}\right).
\end{aligned}
{% end %}

Even-indexed dimensions get a sine, odd-indexed dimensions get the matching cosine, and the wavelength varies geometrically from $2\pi$ (the highest-frequency dimension) to $10000 \cdot 2\pi$ (the lowest-frequency dimension). The choice of $10000$ as the base is arbitrary and was selected as a value large enough that the longest expected sequence still occupies less than one period of the lowest-frequency dimension.

The deeper reason for this particular form is that a constant offset in position becomes a linear transformation of the encoding.

{% mathblock(kind="proposition", name="Sinusoidal positions encode shifts linearly", id="pe-shift") %}
For any fixed offset $k \in \mathbb{R}$, there exists a matrix $\mathbf{M}\_k \in \mathbb{R}^{d\_{\text{model}} \times d\_{\text{model}}}$ depending only on $k$ such that
$$\mathrm{PE}\_{\mathrm{pos} + k} = \mathbf{M}\_k\, \mathrm{PE}\_{\mathrm{pos}}.$$
{% end %}

{% mathblock(kind="proof", name="", id="pe-shift-proof") %}
Within the dimension pair $(2i, 2i+1)$ with angular frequency $\omega\_i = 10000^{-2i/d\_{\text{model}}}$, the encoding is the unit vector $(\sin(\omega\_i \mathrm{pos}), \cos(\omega\_i \mathrm{pos}))$. Shifting the position by $k$ rotates this vector by $\omega\_i k$:
$$\begin{pmatrix} \sin(\omega\_i (\mathrm{pos} + k)) \\\\ \cos(\omega\_i (\mathrm{pos} + k)) \end{pmatrix} = \begin{pmatrix} \cos(\omega\_i k) & \sin(\omega\_i k) \\\\ -\sin(\omega\_i k) & \cos(\omega\_i k) \end{pmatrix} \begin{pmatrix} \sin(\omega\_i \mathrm{pos}) \\\\ \cos(\omega\_i \mathrm{pos}) \end{pmatrix},$$
by the angle-addition formulas. Stacking the $d\_{\text{model}} / 2$ rotation blocks into a block-diagonal matrix gives the claimed $\mathbf{M}\_k$. $\square$
{% end %}

A linear self-attention layer can therefore express **relative position** ("attend to whatever is $k$ steps to my left") as a linear function of the absolute encodings. This was the original argument for sinusoidal over learned positional embeddings. In practice both work and modern models almost universally use one of three alternatives discussed in the variants section: learned absolute embeddings (BERT, GPT-2), Rotary Position Embeddings (most modern LLMs), and ALiBi (some long-context models).

{% mathblock(kind="note", name="Why add and not concatenate", id="pe-add") %}
A natural alternative would be to **concatenate** the positional encoding to the token embedding, doubling the input dimension. The Transformer **adds** the two vectors of equal dimension instead. The argument is parameter efficiency: addition keeps $d\_{\text{model}}$ unchanged, costs no extra parameters, and lets the model learn (via the input projection $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V$) which dimensions to use for content and which to reserve for position. The cost is that the embedding layer has to share its dimensions between content and position, but this turns out to be a negligible bottleneck in practice.
{% end %}

## The encoder block

The encoder takes a sequence of token embeddings and outputs a sequence of contextual representations of the same length. It is a stack of $L$ identical blocks, each with the same internal structure. We discuss the structure once.

A single encoder block contains two sublayers: multi-head self-attention and a position-wise feed-forward network. Each sublayer is wrapped in a residual connection followed by layer normalisation, ie, the output of each sublayer is $\mathrm{LayerNorm}(\mathbf{x} + \mathrm{Sublayer}(\mathbf{x}))$.

### Self-attention

In **self-attention**, the queries, keys, and values are all projections of the same input sequence. For an encoder input $\mathbf{X} \in \mathbb{R}^{N \times d\_{\text{model}}}$,

{% equation(id="self-attention") %}
\begin{aligned}
\mathrm{SelfAttn}(\mathbf{X})
&= \mathrm{MultiHead}(\mathbf{X}\mathbf{W}^Q,\, \mathbf{X}\mathbf{W}^K,\, \mathbf{X}\mathbf{W}^V),
\end{aligned}
{% end %}

with the same $\mathbf{X}$ on the left of all three projections. The interpretation is that every token attends to every other token in the same sequence to form its contextual representation, and the model has $h$ different learned views of how to do that mixing.

### Position-wise feed-forward network

After self-attention, every token passes independently through a small two-layer MLP, the same MLP applied at every position.

{% equation(id="ffn") %}
\begin{aligned}
\mathrm{FFN}(\mathbf{x})
&= \mathbf{W}_2\, \phi\!\left(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1\right) + \mathbf{b}_2,
\end{aligned}
{% end %}

with $\mathbf{W}\_1 \in \mathbb{R}^{d\_{\text{ff}} \times d\_{\text{model}}}$, $\mathbf{W}\_2 \in \mathbb{R}^{d\_{\text{model}} \times d\_{\text{ff}}}$, $\phi$ a non-linearity (originally ReLU, GELU in BERT and beyond), and the standard expansion ratio $d\_{\text{ff}} = 4\, d\_{\text{model}}$. The "position-wise" qualifier emphasises that the same MLP is applied independently to each token, so this sublayer adds no cross-token interaction.

The natural question is: why bother with the FFN at all? Self-attention is already a learned mixing of token vectors, and stacking attention layers gives you depth. The answer is that attention is a fundamentally **linear** operation in its values: each output is a convex combination of value vectors, and the only non-linearity is the softmax over scores, which acts on similarities and not on representations. Without the FFN, an arbitrarily deep attention stack collapses (per-head, before mixing) into a single linear map of the values, with weights that depend on the inputs but values that do not. The FFN is what gives every Transformer block its per-position non-linearity, and in modern parameter accounting the FFN holds the majority of the model's parameters: at $d\_{\text{ff}} = 4 d\_{\text{model}}$, the FFN has $8 d\_{\text{model}}^2$ parameters per block versus $4 d\_{\text{model}}^2$ for the four projections of attention.

### Residual connections

Each sublayer is wrapped in a residual connection: $\mathbf{x} \mapsto \mathbf{x} + \mathrm{Sublayer}(\mathbf{x})$. The motivation is identical to that of ResNets in vision. The forward pass establishes a default route (the identity) past every sublayer, so an untrained or noisy sublayer initially contributes nothing and the signal still propagates through the stack. The backward pass benefits in the same way: the gradient flowing backward through a residual sublayer is $\mathbf{1} + \partial\mathrm{Sublayer}/\partial \mathbf{x}$, so the gradient never shrinks below the identity contribution, no matter what the sublayer's Jacobian looks like. This is the single most important architectural reason that 100-layer Transformers are trainable at all.

### Layer normalisation

The original paper places layer normalisation (defined in the [neural network](/blog/neural-network/) post) after the residual addition, ie, $\mathrm{LayerNorm}(\mathbf{x} + \mathrm{Sublayer}(\mathbf{x}))$. This is now called **post-LN** to distinguish it from the **pre-LN** variant $\mathbf{x} + \mathrm{Sublayer}(\mathrm{LayerNorm}(\mathbf{x}))$ used in essentially every modern model. The trade-off is discussed in the variants section; for now, the layer normalisation is what keeps the activations in a stable range as the residual stream grows with depth.

Putting everything together, the full encoder block computes

{% equation(id="encoder-block") %}
\begin{aligned}
\mathbf{x}' &= \mathrm{LayerNorm}\big(\mathbf{x} + \mathrm{SelfAttn}(\mathbf{x})\big), \\
\mathbf{x}'' &= \mathrm{LayerNorm}\big(\mathbf{x}' + \mathrm{FFN}(\mathbf{x}')\big),
\end{aligned}
{% end %}

with the same block stacked $L$ times (the original paper used $L = 6$ for both the encoder and the decoder; modern models go up to $L = 96$ and beyond).

## The decoder block

The decoder mirrors the encoder but adds two complications. It must generate its output sequence one token at a time, conditioned on the encoder's output, which forces two structural changes.

### Masked self-attention

At training time the decoder is fed the entire target sequence in parallel for efficiency, but it must not be allowed to peek at future tokens when predicting the next one. The fix is **causal masking**: in the self-attention of the decoder, every query at position $i$ can only attend to keys at positions $j \leq i$. Operationally, the score matrix gets an additive mask before the softmax,

{% equation(id="causal-mask") %}
\begin{aligned}
\mathrm{MaskedAttn}(\mathbf{Q}, \mathbf{K}, \mathbf{V})
&= \mathrm{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right) \mathbf{V},
\end{aligned}
{% end %}

with $\mathbf{M}\_{i,j} = 0$ if $j \leq i$ and $\mathbf{M}\_{i,j} = -\infty$ otherwise. The infinities push the softmax of forbidden positions to exactly zero, removing them from the weighted average. This single trick is what lets us train a causal Transformer in $O(N^2)$ parallel time instead of $O(N)$ sequential time, while still producing output that is causally consistent with autoregressive decoding at inference.

### Cross-attention

After masked self-attention, the decoder block inserts a third sublayer that performs **cross-attention** to the encoder's output: queries come from the decoder, keys and values come from the encoder.

{% equation(id="cross-attention") %}
\begin{aligned}
\mathrm{CrossAttn}(\mathbf{X}_{\text{dec}}, \mathbf{X}_{\text{enc}})
&= \mathrm{MultiHead}\big(\mathbf{X}_{\text{dec}} \mathbf{W}^Q, \\
&\qquad\quad \mathbf{X}_{\text{enc}} \mathbf{W}^K,\, \mathbf{X}_{\text{enc}} \mathbf{W}^V\big).
\end{aligned}
{% end %}

This is the only point at which the decoder sees the encoder. The interpretation is that each output token, while being generated, looks back at the entire input sequence and pulls relevant information from it. In machine translation, this is where the alignment between source and target words is implicitly learned. In modern decoder-only language models there is no encoder, and cross-attention is dropped entirely; in encoder-decoder models like the original Transformer, T5, and BART, cross-attention is retained and is essential.

The full decoder block stacks three sublayers, each with its own residual and layer norm:

{% equation(id="decoder-block") %}
\begin{aligned}
\mathbf{x}' &= \mathrm{LayerNorm}\big(\mathbf{x} + \mathrm{MaskedAttn}(\mathbf{x})\big), \\
\mathbf{x}'' &= \mathrm{LayerNorm}\big(\mathbf{x}' + \mathrm{CrossAttn}(\mathbf{x}', \mathbf{X}_{\text{enc}})\big), \\
\mathbf{x}''' &= \mathrm{LayerNorm}\big(\mathbf{x}'' + \mathrm{FFN}(\mathbf{x}'')\big).
\end{aligned}
{% end %}

After the final decoder block, a learned linear projection maps the $d\_{\text{model}}$-dimensional vector at each position to the vocabulary size, and a softmax over the result gives the next-token probability distribution. The training loss is the standard categorical cross-entropy from the [logistic regression](/blog/logistic-regression/) post, computed independently at each output position and averaged.

{% mathblock(kind="note", name="Architectural taxonomy", id="arch-taxonomy") %}
The three Transformer flavours correspond to which half of the original architecture you keep. **Encoder-only** models (BERT{{ reference(key="devlin2019bert") }}, RoBERTa, the encoder of T5) drop the decoder and produce contextual embeddings useful for classification, retrieval, and feature extraction. **Decoder-only** models (GPT-2, GPT-3, LLaMA, modern LLMs) drop the encoder and the cross-attention sublayer, yielding a pure causal language model. **Encoder-decoder** models (the original Transformer, T5, BART) keep both halves and use the encoder for the source sequence and the decoder for the target. Vision Transformers{{ reference(key="dosovitskiy2021vit") }} use the encoder-only flavour with image patches replacing word tokens.
{% end %}

## A worked example by hand

To make the moving parts concrete, here is one forward pass through scaled dot-product attention for a single head with three tokens and $d\_k = d\_v = 2$. The input is $\mathbf{X} \in \mathbb{R}^{3 \times 2}$ and we use the identity for all three projection matrices, so $\mathbf{Q} = \mathbf{K} = \mathbf{V} = \mathbf{X}$:

{% equation(id="example-x") %}
\mathbf{X} = \mathbf{Q} = \mathbf{K} = \mathbf{V} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}.
{% end %}

The raw score matrix is

{% equation(id="example-scores") %}
\mathbf{Q}\mathbf{K}^\top = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix} \begin{pmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{pmatrix} = \begin{pmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{pmatrix}.
{% end %}

Scaling by $\sqrt{d\_k} = \sqrt{2}$ gives the matrix $\mathbf{Q}\mathbf{K}^\top / \sqrt{2}$ with entries roughly $0.707$ and $1.414$ in the appropriate places. Applying a row-wise softmax gives the attention weights, ie, every row sums to one:

{% equation(id="example-attn") %}
\mathbf{A} = \mathrm{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{2}}\right) \approx \begin{pmatrix} 0.391 & 0.219 & 0.391 \\ 0.219 & 0.391 & 0.391 \\ 0.249 & 0.249 & 0.502 \end{pmatrix}.
{% end %}

Two sanity checks fall out. The first row, query $(1, 0)$, places equal weight on key 1 (which is identical) and key 3 (which differs only by the $(0, 1)$ component), and a smaller weight on key 2 (which is orthogonal). The third row, query $(1, 1)$, has the largest self-similarity and unsurprisingly places its largest weight on itself. The softmax does not collapse to a one-hot because the scaling has kept the logits in a reasonable range.

The output is

{% equation(id="example-output") %}
\mathbf{A}\, \mathbf{V} \approx \begin{pmatrix} 0.391 & 0.219 & 0.391 \\ 0.219 & 0.391 & 0.391 \\ 0.249 & 0.249 & 0.502 \end{pmatrix} \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix} \approx \begin{pmatrix} 0.781 & 0.609 \\ 0.609 & 0.781 \\ 0.751 & 0.751 \end{pmatrix},
{% end %}

ie, every output token is a soft mix of the three input tokens, with the mix dictated by similarity. Replace the identity projections with learned $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V$, run this in parallel across $h$ heads, mix with $\mathbf{W}^O$, add a residual, layer-normalise, and you have one Transformer sublayer.

## Variants of attention

The original Transformer fixes a single answer to every design question. Each of those answers has been revisited, and a small library of attention variants now coexists in production. They split cleanly into two families: those that change **what** is attended to (sparse, local, or low-rank patterns to escape the $O(N^2)$ wall) and those that change **how** the attention is computed (algorithmic and hardware-aware reformulations of the same mathematics).

### Multi-query and grouped-query attention

The bottleneck during autoregressive generation is not training but **inference**, and specifically the size of the **KV cache**. At decoding step $t$, the model needs the keys and values of all previous $t-1$ tokens to compute attention for the new query. Rather than recomputing them, they are cached. The cache size is $2 \cdot L \cdot N \cdot h \cdot d\_k$ for $L$ layers, $h$ heads, sequence length $N$. For a typical large model this is gigabytes of state per generation, and reading it from memory at every decoding step quickly becomes the bottleneck.

**Multi-Query Attention** (MQA){{ reference(key="shazeer2019fast") }} responds by sharing a single key and value across all heads, while keeping per-head queries:

{% equation(id="mqa") %}
\begin{aligned}
\mathrm{head}_i
&= \mathrm{Attention}(\mathbf{X}\mathbf{W}_i^Q,\, \mathbf{X}\mathbf{W}^K,\, \mathbf{X}\mathbf{W}^V).
\end{aligned}
{% end %}

The KV cache shrinks by a factor of $h$ (so $8\times$ smaller for the standard 8-head model), at the cost of reducing the per-head expressivity of the keys and values. Empirically MQA hurts model quality slightly. **Grouped-Query Attention** (GQA){{ reference(key="ainslie2023gqa") }} interpolates: the $h$ query heads are partitioned into $g$ groups, each of which shares one key/value pair. Setting $g = h$ recovers vanilla multi-head attention, $g = 1$ recovers MQA, and intermediate values (LLaMA-2 70B uses $g = 8$ with $h = 64$) capture most of the inference speedup with negligible quality loss.

### Sliding-window and sparse attention

The full $N \times N$ attention matrix is wasteful when most useful interactions are local. **Sliding-window attention** restricts each query to attend only to the $w$ nearest tokens on either side, ie, $\mathbf{M}\_{i,j} = -\infty$ when $|i - j| > w$. The compute and memory cost drops from $O(N^2)$ to $O(N w)$, and a stack of $L$ layers gives an effective receptive field of $L \cdot w$, ie, distant tokens still influence each other through depth, like in a CNN.

The Longformer{{ reference(key="beltagy2020longformer") }} combines a sliding window with a small number of **global tokens** that attend to everything and that everything attends to (typically the [CLS] token and any task-specific markers), so the model retains a path of length two between any two positions. BigBird{{ reference(key="zaheer2020bigbird") }} adds **random attention** edges to the same recipe, and proves that the resulting sparse pattern is a universal sequence-to-sequence approximator, ie, no expressivity is lost in principle. The Sparse Transformer{{ reference(key="child2019sparse") }} uses fixed strided patterns inspired by 2D convolutions for image-like sequences. All of these are valuable when the natural sequence length is too long for full attention but the dependencies are mostly local.

### Linear attention

A different angle on the $O(N^2)$ problem is to approximate the softmax kernel by a feature map that lets us reorder the computation. Write attention as $\mathrm{softmax}(\mathbf{Q}\mathbf{K}^\top) \mathbf{V}$ and observe that, for any feature map $\phi$ such that $\mathrm{softmax}(\mathbf{q}\_i^\top \mathbf{k}\_j) \approx \phi(\mathbf{q}\_i)^\top \phi(\mathbf{k}\_j) / Z(\mathbf{q}\_i)$, we can rewrite the output as

{% equation(id="linear-attn") %}
\begin{aligned}
\mathbf{o}_i
&= \frac{\sum_{j} \phi(\mathbf{q}_i)^\top \phi(\mathbf{k}_j)\, \mathbf{v}_j}{\sum_{j} \phi(\mathbf{q}_i)^\top \phi(\mathbf{k}_j)} \\
&= \frac{\phi(\mathbf{q}_i)^\top \big(\sum_{j} \phi(\mathbf{k}_j)\, \mathbf{v}_j^\top\big)}{\phi(\mathbf{q}_i)^\top \big(\sum_{j} \phi(\mathbf{k}_j)\big)}.
\end{aligned}
{% end %}

The associativity matters: instead of forming the $N \times N$ matrix $\phi(\mathbf{Q}) \phi(\mathbf{K})^\top$ and then multiplying by $\mathbf{V}$, we first form the $d \times d$ matrix $\sum\_{j} \phi(\mathbf{k}\_j)\, \mathbf{v}\_j^\top$ and then apply it to each query. The cost drops from $O(N^2 d)$ to $O(N d^2)$, ie, **linear in the sequence length**. The trade-off is that we have to live with whatever the feature map $\phi$ approximates. The Performer{{ reference(key="choromanski2021performer") }} uses random Fourier features that give an unbiased estimate of the softmax kernel; Linformer{{ reference(key="wang2020linformer") }} achieves linearity differently, by projecting the keys and values down to a fixed dimension $k \ll N$ before attention; Katharopoulos et al.{{ reference(key="katharopoulos2020transformers") }} use $\phi(\mathbf{x}) = \mathrm{elu}(\mathbf{x}) + 1$ and observe that the resulting causal attention can be implemented as a recurrent state, recovering RNN-like inference cost.

Linear attention has remained niche compared to FlashAttention plus sparse patterns, because the kernel approximation tends to hurt model quality and the constant factors of the $O(N d^2)$ algorithm are unfriendly when $d$ is large.

### FlashAttention

FlashAttention{{ reference(key="dao2022flashattention") }} computes the **exact same attention** as the standard formula but reorganises the computation to minimise reads and writes to high-bandwidth memory (HBM), the slow off-chip memory of a GPU. It is now the default kernel in essentially every Transformer framework and is the single most consequential systems-level change to attention since the original paper. The mathematics of why it works (the online softmax) and the hardware reality of why it matters (the GPU memory hierarchy) deserve their own dedicated treatment, taken up in the [FlashAttention and the GPU](#flashattention-and-the-gpu) section below.

{% mathblock(kind="note", name="Cost analysis (attention variants)", id="attn-cost-table") %}
Standard attention is $O(N^2 d)$ time and $O(N^2 + N d)$ memory. **MQA/GQA** reduces only the inference KV cache, not the asymptotic attention cost; the saving is in memory bandwidth and cache footprint at decode time, $h \times$ smaller for MQA and $h/g \times$ smaller for GQA. **Sliding-window attention** is $O(N w d)$ time and $O(N w + N d)$ memory for window size $w$. **Linear attention** is $O(N d^2)$ time and $O(d^2 + N d)$ memory. **FlashAttention** keeps the $O(N^2 d)$ flop count of standard attention but cuts HBM IO from $O(N^2)$ to $O(N^2 / B)$, giving a 2-4x wall-clock speedup. The right choice depends on the regime: small $N$ with large $d$ favours standard attention plus FlashAttention; very large $N$ with small $d$ and structurally local dependencies favours sliding window or sparse patterns; extremely long $N$ with no locality structure is where linear attention is least bad, despite its quality cost.
{% end %}

## Modern design decisions

The Transformer of 2017 has been quietly upgraded over the years. The differences between the original and a modern decoder-only LLM are not in the high-level structure (still attention plus FFN, residuals, layer norm, plus or minus an encoder) but in a handful of localised choices. Each has a clean motivation.

### Pre-LN vs post-LN

The original paper uses **post-LN**: $\mathrm{LayerNorm}(\mathbf{x} + \mathrm{Sublayer}(\mathbf{x}))$. This places the layer norm **after** the residual addition. In deep stacks (more than a few dozen layers), post-LN is famously hard to train: the gradient through the residual path is rescaled by the layer norm Jacobian at every block, and this rescaling can shrink or amplify gradients in ways that prevent convergence without a carefully tuned learning rate warmup. Xiong et al.{{ reference(key="xiong2020layernorm") }} analysed this and showed that **pre-LN**, which applies the layer norm **inside** the sublayer ($\mathbf{x} + \mathrm{Sublayer}(\mathrm{LayerNorm}(\mathbf{x}))$), preserves the identity path through the residual cleanly and trains stably without warmup at any depth. Essentially every modern Transformer uses pre-LN.

The trade-off is that pre-LN typically reaches a slightly worse final loss than well-tuned post-LN at modest depths, because the un-normalised residual stream grows through the network and the deepest sublayers operate on inputs of large norm. Modern recipes mitigate this with a final layer norm at the end of the stack and with careful initialisation of the residual contributions.

### RMSNorm

**Root Mean Square Norm**{{ reference(key="zhang2019rmsnorm") }} drops the mean-centring step of layer norm and normalises by the root mean square of the activations alone:

{% equation(id="rmsnorm") %}
\begin{aligned}
\mathrm{RMSNorm}(\mathbf{z})
&= \boldsymbol{\gamma} \odot \frac{\mathbf{z}}{\sqrt{\frac{1}{d} \sum_{j=1}^d z_j^2 + \varepsilon}}.
\end{aligned}
{% end %}

The mean-subtraction in standard layer norm is a small but non-trivial fraction of the per-layer compute, and empirically the centring contributes little to training stability once the network has trained for a while. RMSNorm preserves the variance-control benefit of layer norm while saving the cost of a sum and a subtraction, and is the default in LLaMA, PaLM, and most recent open LLMs.

### Rotary position embeddings

Sinusoidal positional encoding adds positional information at the input and trusts the network to preserve it through the stack. **Rotary Position Embeddings** (RoPE){{ reference(key="su2021roformer") }} take a different angle: they encode position as a multiplicative rotation applied to the **queries and keys** at every layer, leaving the values untouched.

For each two-dimensional slice of $\mathbf{q}$ and $\mathbf{k}$ at frequency $\omega\_i = 10000^{-2i/d}$, RoPE rotates the slice by angle $\omega\_i \mathrm{pos}$. Because the dot product of two rotated vectors is

{% equation(id="rope-dot") %}
\begin{aligned}
\big(\mathbf{R}_{\omega \mathrm{pos}_q}\mathbf{q}\big)^\top \big(\mathbf{R}_{\omega \mathrm{pos}_k}\mathbf{k}\big)
&= \mathbf{q}^\top \mathbf{R}_{\omega(\mathrm{pos}_k - \mathrm{pos}_q)} \mathbf{k},
\end{aligned}
{% end %}

the attention score depends only on the **relative** position of the query and key, not on their absolute positions. This is the property the sinusoidal encoding only approximated through a linear map; RoPE makes it exact and structural. The other practical benefit is **length extrapolation**: a model trained at sequence length $N$ can be evaluated at sequence length $2N$ by extending the rotations, with a graceful (though not perfect) degradation in quality. Sinusoidal positional embeddings, learned absolute embeddings, and RoPE all exist in production; RoPE is the most common modern choice.

ALiBi{{ reference(key="press2022alibi") }} pursues the same goal differently: it adds a linear bias proportional to the query-key distance directly to the attention scores, with a per-head slope, and uses no positional embedding at the input. ALiBi gives strong length extrapolation at the cost of a more rigid positional inductive bias.

### GLU variants

The original feed-forward block is a two-layer MLP with a single non-linearity. Shazeer{{ reference(key="shazeer2020glu") }} observed that replacing the first projection with a **gated linear unit** improves quality at constant parameter count.

{% equation(id="glu") %}
\mathrm{GLU}(\mathbf{x}) = (\mathbf{W}_1 \mathbf{x}) \odot \sigma(\mathbf{W}_g \mathbf{x}),
{% end %}

with two parallel input projections, the second wrapped in a sigmoid, multiplied elementwise. **SwiGLU**, the most common variant in modern LLMs, replaces $\sigma$ with the SiLU/Swish activation $\mathrm{SiLU}(z) = z \sigma(z)$:

{% equation(id="swiglu") %}
\mathrm{SwiGLU}(\mathbf{x}) = (\mathbf{W}_1 \mathbf{x}) \odot \mathrm{SiLU}(\mathbf{W}_g \mathbf{x}).
{% end %}

To keep the parameter budget constant against a vanilla FFN of inner dimension $4 d\_{\text{model}}$, the SwiGLU FFN typically uses inner dimension $\frac{8}{3} d\_{\text{model}}$ (because it has three projections instead of two). The gating gives the model a multiplicative interaction between input dimensions inside the FFN that the additive ReLU/GELU cannot express, and the empirical gain is consistent enough that SwiGLU is now the default in LLaMA, PaLM, and most modern decoder-only LLMs.

The KV cache, the inference-time counterpart to all of these training-time choices, gets its own treatment in the [inference optimisation](#inference-optimisation) section below.

{% mathblock(kind="warning", name="What changes from 2017 to today", id="modern-summary") %}
A modern decoder-only LLM differs from the 2017 Transformer in roughly the following way: replace post-LN with **pre-LN**, replace LayerNorm with **RMSNorm**, replace sinusoidal positional encodings with **RoPE** (or ALiBi), replace ReLU FFNs with **SwiGLU**, replace multi-head with **grouped-query attention**, and use **FlashAttention** as the kernel. The high-level architecture is almost untouched: every component still maps to something Vaswani et al. described in 2017. The progress has been, almost entirely, in finding less wasteful versions of the same design.
{% end %}

## FlashAttention and the GPU

Training a Transformer is, in practice, an exercise in pushing as much arithmetic as possible through a piece of silicon that costs tens of dollars per hour to rent. The cost analysis is unforgiving: a single forward + backward pass over a batch of $B$ sequences of length $N$ runs an attention sublayer at $O(B N^2 d)$ flops and, at least in the naive implementation, $O(B N^2)$ memory for the attention matrix. Doubling the context length quadruples both. Once $N$ goes from the original Transformer's $512$ tokens to a modern model's $32{,}768$ or beyond, the attention matrix alone is many gigabytes per layer and dominates not just memory but wall-clock time, since every byte of it has to be written to and read back from off-chip memory at every step.

This is the bottleneck FlashAttention attacks. It is, on paper, a small reorganisation of standard attention: the same arithmetic, performed in a different order, with intermediate tensors kept on-chip instead of spilled to memory. In practice it is a 2-4x training speedup and a memory reduction from $O(N^2)$ to $O(N)$ that has unlocked context lengths nobody seriously trained at before. The gap between the theoretical ease of the change and the practical difficulty of writing the kernel is a clean window into how a modern GPU actually works, and what custom CUDA code has to think about. Before getting to the kernel itself, we need to spend a moment on the hardware it is running on.

### A crash course on the GPU

A modern GPU (treating an NVIDIA H100 as the running example) is best thought of not as a CPU with more cores, but as a stack of memories of wildly different bandwidths and capacities, with a large pool of arithmetic units bolted to the smallest one. The hierarchy, fastest to slowest:

The **registers** sit inside each thread and provide the only memory the arithmetic units can read directly without latency. Each streaming multiprocessor (SM) has 64 KB of register file shared across the threads scheduled on it.

The **shared memory** (also called SRAM or scratchpad), of which the H100 has 228 KB per SM, is on-chip, programmer-controlled, and has bandwidth in the tens of TB/s. It is shared across all threads in a single thread block.

The **L2 cache** is on-chip but shared across all SMs (60 MB on H100), and acts as a transparent cache for HBM accesses.

The **HBM** (high-bandwidth memory) is the off-chip DRAM that holds the model weights, activations, and gradients. The H100 has 80 GB at roughly 3 TB/s, which sounds like a lot but is two orders of magnitude slower than SRAM.

Crucially, the H100's tensor cores produce arithmetic at roughly 1 PFLOPS in float16 and 2 PFLOPS in float8. Dividing by the HBM bandwidth gives the **arithmetic intensity break-even**: any kernel that performs fewer than $\sim 300$ float operations per byte of HBM read is **memory-bound**, ie, it sits idle waiting for memory while the tensor cores do nothing. Standard attention reads the $N \times N$ score matrix from HBM at least twice (once to apply the softmax, once to multiply by $\mathbf{V}$) and writes it twice, making the per-byte arithmetic intensity small and the kernel memory-bound on every modern GPU.

{% mathblock(kind="note", name="Roofline mental model", id="roofline") %}
The roofline model says the achievable throughput of a kernel is $\min(\mathrm{peak~flops}, \mathrm{arithmetic~intensity} \times \mathrm{HBM~bandwidth})$. The "roof" is flat at the peak flops on the right (compute-bound) and slopes up to it on the left (memory-bound), with the slope equal to the HBM bandwidth. Any optimisation that increases arithmetic intensity (more flops per byte) moves the kernel rightward under the roof; any optimisation that decreases HBM traffic does the same. FlashAttention is the canonical case of the second.
{% end %}

A GPU executes code in **kernels**, single procedures launched on a grid of thousands of threads. Threads are grouped into **warps** of 32 that execute the same instruction in lockstep (single-instruction multiple-thread, SIMT), warps are grouped into **thread blocks** that share an SM and can synchronise via barriers, and thread blocks together form the **grid**. The programmer controls the grid and block dimensions, the use of shared memory, and the order in which data is loaded from HBM. Everything else (which warp runs when, which cache line is evicted) is the hardware's job.

### The online softmax

The key algorithmic ingredient that lets attention be tiled is the observation that the softmax can be computed **incrementally** as new chunks of the input arrive. The standard softmax with the log-sum-exp stabilisation from the [softmax](/blog/softmax-function/) post requires two passes over the input row: one to find the maximum, and one to compute the normalised exponentials. The online softmax fuses both into a single streaming pass.

{% mathblock(kind="proposition", name="Online softmax update", id="online-softmax") %}
Suppose we have processed a prefix of scores $s\_1, \ldots, s\_k$ and maintain the running maximum $m\_k = \max\_{j \leq k} s\_j$ and running denominator $\ell\_k = \sum\_{j \leq k} e^{s\_j - m\_k}$. When a new score $s\_{k+1}$ arrives, the updated statistics are
$$\begin{aligned}
m\_{k+1} &= \max(m\_k, s\_{k+1}), \\\\
\ell\_{k+1} &= \ell\_k\, e^{m\_k - m\_{k+1}} + e^{s\_{k+1} - m\_{k+1}}.
\end{aligned}$$
The final softmax of the full row is then $p\_j = e^{s\_j - m\_N} / \ell\_N$.
{% end %}

{% mathblock(kind="proof", name="", id="online-softmax-proof") %}
By the definition of the running denominator after the new sample, $\ell\_{k+1} = \sum\_{j \leq k+1} e^{s\_j - m\_{k+1}}$. Split the sum into the prefix and the new term:
$$\begin{aligned}
\ell\_{k+1} &= \sum\_{j \leq k} e^{s\_j - m\_{k+1}} + e^{s\_{k+1} - m\_{k+1}} \\\\
&= e^{m\_k - m\_{k+1}} \sum\_{j \leq k} e^{s\_j - m\_k} + e^{s\_{k+1} - m\_{k+1}} \\\\
&= e^{m\_k - m\_{k+1}}\, \ell\_k + e^{s\_{k+1} - m\_{k+1}}.
\end{aligned}$$
The maximum case is immediate. $\square$
{% end %}

The same rescaling trick extends to the attention output: if we have an accumulated weighted sum $\mathbf{o}\_k = \sum\_{j \leq k} e^{s\_j - m\_k} \mathbf{v}\_j$, then the update on a new sample is $\mathbf{o}\_{k+1} = e^{m\_k - m\_{k+1}}\, \mathbf{o}\_k + e^{s\_{k+1} - m\_{k+1}}\, \mathbf{v}\_{k+1}$. The unnormalised sum is rescaled by the same factor as the denominator, so the ratio $\mathbf{o}\_N / \ell\_N$ at the end is the exact attention output. We only need to remember three numbers per row at any given time ($m$, $\ell$, and the partial $\mathbf{o}$), which fits in registers.

### The FlashAttention kernel

With the online softmax in hand, FlashAttention is a tiled outer loop over key-value blocks and an inner loop over query blocks. A simplified per-block pseudocode follows.

{% equation(id="flash-pseudo") %}
\begin{aligned}
&\text{for each } \mathbf{K}_j, \mathbf{V}_j \text{ block in HBM:}\\
&\quad \text{load } \mathbf{K}_j, \mathbf{V}_j \text{ into SRAM,}\\
&\quad \text{for each } \mathbf{Q}_i \text{ block in HBM:}\\
&\quad\quad \text{load } \mathbf{Q}_i \text{ and the running } (m_i, \ell_i, \mathbf{O}_i) \text{ into SRAM,}\\
&\quad\quad \mathbf{S}_{ij} = \mathbf{Q}_i \mathbf{K}_j^\top / \sqrt{d_k} \quad \text{(in SRAM)},\\
&\quad\quad \text{update } m_i, \ell_i, \mathbf{O}_i \text{ via the online softmax with } \mathbf{S}_{ij}, \mathbf{V}_j,\\
&\quad\quad \text{write back } (m_i, \ell_i, \mathbf{O}_i) \text{ to HBM,}\\
\end{aligned}
{% end %}

with the final output $\mathbf{O}\_i / \ell\_i$ at the end. The full $N \times N$ score matrix never exists in HBM, only the small $B\_q \times B\_k$ block in SRAM does at any moment. Block sizes are chosen so that $B\_q + B\_k$ rows of $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ plus the score block fit in the 100s of KB of SRAM per SM; typical values are $B\_q = B\_k = 64$ to $128$.

For the **backward pass**, the standard recipe of caching the activation matrix $\mathbf{P} = \mathrm{softmax}(\mathbf{S})$ would defeat the entire memory saving (the cached $\mathbf{P}$ is itself $O(N^2)$). FlashAttention instead **recomputes** $\mathbf{P}$ block by block during the backward pass using the saved $(m, \ell)$ statistics, which fits in $O(N)$ memory and adds only a modest fraction to the backward flop count. This is gradient checkpointing applied at the right granularity, and is what makes attention with $N \in [10^5, 10^6]$ fit on a single GPU at all.

{% mathblock(kind="note", name="Why is the kernel hard to write", id="flash-hardness") %}
The mathematics of the online softmax fits on one slide. The kernel is several thousand lines of CUDA in the reference implementation. The difficulty is hardware-dependent: choosing block sizes that match the SM's shared memory and register budget, scheduling memory loads to overlap with tensor-core arithmetic so the kernel is compute-bound, handling the boundary blocks where $N$ is not a multiple of $B$, supporting the causal mask without wasting work on the masked-out half of the matrix, and (in FlashAttention-2{{ reference(key="dao2023flashattention2") }}) reorganising the parallelism so that the $\mathbf{Q}$ loop is on the outside and parallel across SMs rather than the $\mathbf{K}$ loop, which removes a serial dependency that limited the original. FlashAttention-3{{ reference(key="shah2024flashattention3") }} pushes further by using the H100's asynchronous warp groups and float8 precision to overlap softmax computation with tensor-core matmuls and to reach $\sim 75\%$ of the H100's float8 peak. Each of these is mostly orthogonal to the algorithm and is a pure piece of GPU programming.
{% end %}

### Custom CUDA kernels in practice

Most Transformer code never touches CUDA directly. PyTorch and JAX dispatch to vendor libraries (cuBLAS, cuDNN, CUTLASS) that already have hand-tuned kernels for the common operations, and the `torch.nn.functional.scaled_dot_product_attention` primitive in modern PyTorch dispatches to FlashAttention under the hood when the input shapes and dtypes are supported. The cases where a custom kernel pays off are the ones where the fused operation does not exist in the library. Writing **fused RMSNorm + linear + activation**, **fused attention + bias + dropout**, **fused MoE routing + dispatch**, or **fused KV-cache write + attention** combines several elementwise or memory-bound operations into a single kernel that loads each input from HBM exactly once.

The non-CUDA frontend now in widest use is **Triton**, an LLVM-based language that lets you write block-level kernels in Python-like syntax and have a compiler emit reasonable PTX. The original FlashAttention reference implementation has both a CUDA and a Triton version, and many production kernels (RMSNorm, RoPE, the MoE permute) ship as Triton because the iteration loop is much faster than CUDA. The trade-off is that Triton is slightly less expressive than raw CUDA and lags behind on the very newest hardware features (Hopper's TMA, Blackwell's float4), so the fastest path is usually a Triton prototype that gets ported to CUDA once the algorithm is settled.

The economic argument for spending engineering time on custom kernels is straightforward: at the scale of modern training runs, a 10% kernel speedup translates to single-digit percentage savings in dollars on a multi-million-dollar run. The engineering investment is paid back many times over for any kernel that runs in the inner loop of the model. This is why every major lab maintains a CUDA team and why FlashAttention has earned three full rewrites across three GPU generations.

## Mixture of experts

The position-wise FFN holds the majority of a Transformer's parameters but does not need them on every token. Most tokens are syntactically simple and a small slice of the FFN suffices; a few tokens carry rare semantic content and would benefit from a much wider FFN if it were available. **Mixture of Experts** (MoE) makes this trade-off explicit by replacing the dense FFN with a bank of $E$ smaller FFNs (the experts) plus a small **router** that, for each input token, selects a sparse subset of experts to evaluate. The total parameter count grows by roughly a factor of $E$, while the compute per token grows only by the factor of how many experts are actually used per token (typically $1$ or $2$).

This is the central decoupling that motivates MoE: parameter count and compute, which are coupled in a dense model, become independent design knobs. A 10x larger model that activates only $1/10$ of its parameters per token costs the same to train and run as the original dense model, but has 10x as much memorised knowledge to draw from. The idea predates Transformers (Jacobs and Hinton studied mixtures of local experts in the 1990s) but became practical at scale only with the sparsely-gated MoE layer of Shazeer et al.{{ reference(key="shazeer2017moe") }}, the GShard system that made it scale across thousands of accelerators{{ reference(key="lepikhin2021gshard") }}, and the Switch Transformer that simplified the routing to top-1{{ reference(key="fedus2022switch") }}.

### The MoE layer

Replace the dense FFN of {{ eqref(id="ffn") }} with $E$ parallel FFN experts $\mathrm{FFN}\_1, \ldots, \mathrm{FFN}\_E$ and a router $\mathbf{W}\_r \in \mathbb{R}^{E \times d\_{\text{model}}}$ that produces a logit per expert per token. For each token $\mathbf{x}$:

{% equation(id="moe-router") %}
\mathbf{g}(\mathbf{x}) = \mathrm{softmax}\big(\mathbf{W}_r \mathbf{x}\big) \in \mathbb{R}^{E}.
{% end %}

The full output would be $\sum\_{e=1}^{E} g\_e(\mathbf{x})\, \mathrm{FFN}\_e(\mathbf{x})$, an expensive dense mixture. Sparsity is introduced by restricting the sum to the **top-$k$** experts:

{% equation(id="moe-output") %}
\begin{aligned}
\mathrm{MoE}(\mathbf{x})
&= \sum_{e \in \mathrm{TopK}(\mathbf{g}(\mathbf{x}),\, k)} \tilde{g}_e(\mathbf{x})\, \mathrm{FFN}_e(\mathbf{x}),
\end{aligned}
{% end %}

where $\tilde{g}\_e$ are the gate values renormalised over the selected experts so they sum to one. Switch Transformer uses $k = 1$ for maximum simplicity; Mixtral{{ reference(key="jiang2024mixtral") }} and DeepSeekMoE{{ reference(key="dai2024deepseekmoe") }} use $k = 2$; the original GShard used $k = 2$ as a compromise between expressivity and load-balance variance. The argument for $k > 1$ is that gradients flow through every selected expert and the network can interpolate between experts, while $k = 1$ gives binary routing decisions that are non-differentiable and that GShard handles by passing the gradient straight through the gate.

### Load balancing

The router is free to send all tokens to the same expert and ignore the others, in which case the model has effectively wasted $E - 1$ of its parameters. Worse, an expert that receives no tokens receives no gradient, so the imbalance is self-reinforcing. The standard fix is an **auxiliary load-balancing loss** added to the cross-entropy.

For a batch with $T$ tokens, let $f\_e = \frac{1}{T} \sum\_{t=1}^{T} \mathbb{1}\\{e \in \mathrm{TopK}(\mathbf{g}(\mathbf{x}\_t))\\}$ be the fraction of tokens routed to expert $e$, and $P\_e = \frac{1}{T} \sum\_{t=1}^{T} g\_e(\mathbf{x}\_t)$ be the average gate probability for that expert. The Switch Transformer loss is

{% equation(id="moe-aux-loss") %}
\mathcal{L}_{\text{aux}} = \alpha \cdot E \cdot \sum_{e=1}^{E} f_e\, P_e,
{% end %}

with a small coefficient $\alpha \in [10^{-3}, 10^{-1}]$. Under uniform routing (the desired equilibrium), $f\_e = P\_e = 1/E$ for every $e$, and the loss equals $\alpha$. Any deviation from uniformity correlates the realised load $f\_e$ with the predicted load $P\_e$ and increases the loss; the gradient pushes the router back towards a balanced allocation. The factor $E$ keeps the loss scale-invariant in the number of experts.

A simpler alternative used in some recent models is **expert capacity** with token dropping: each expert is given a fixed budget of how many tokens it can process per batch (typically $1.0$ to $1.25 \cdot T / E$), and tokens routed to a saturated expert are simply dropped from the MoE layer (their residual passes through unchanged). This keeps the per-device computation bounded but introduces information loss that the model has to route around. DeepSeekMoE replaces the auxiliary loss entirely with **expert biasing**: a learnable per-expert bias added to the routing logits and updated to drive the load distribution towards uniformity, which avoids the gradient interaction between the main loss and the load-balancing loss.

### Modern MoE variants

Three design knobs distinguish modern MoE architectures from the original Switch Transformer.

**Fine-grained experts**, introduced by DeepSeekMoE, replace a few wide experts by many narrow ones. Splitting each expert into $m$ pieces with $1/m$ the hidden dimension and increasing $k$ proportionally keeps the per-token compute constant while allowing far more nuanced routing decisions. The combinatorial number of expert subsets grows from $\binom{E}{k}$ to $\binom{m E}{m k}$, exponentially more, and empirically gives better quality at the same compute.

**Shared experts** designate one or two of the experts as **always active**, so every token sees them in addition to the top-$k$ chosen experts. The intuition is that some computation (basic syntactic processing, common patterns) is needed by every token, and forcing it through the gating mechanism wastes routing decisions on a foregone conclusion. DeepSeekMoE and Qwen MoE both use shared experts.

**Expert parallelism** is a parallelism strategy specific to MoE. The $E$ experts are sharded across devices (say, $4$ experts per device for $E = 32$ on $8$ GPUs), and the routing decision determines which device a given token is sent to. The two collective communications involved are **all-to-all** before the experts (every device sends its tokens to the right destinations) and another all-to-all after (the expert outputs are returned to their tokens' home devices). This makes the network bandwidth between devices a hard bottleneck for MoE training and is the reason MoE scaled best on the very-high-bandwidth interconnects (NVLink, NVSwitch, TPU torus) used at large labs.

{% mathblock(kind="note", name="Cost analysis (MoE FFN)", id="moe-cost") %}
For an MoE layer with $E$ experts of inner dimension $d\_{\text{ff}}$ and top-$k$ routing, per-token compute is $O(k \cdot d\_{\text{model}} \cdot d\_{\text{ff}})$ instead of $O(d\_{\text{model}} \cdot d\_{\text{ff}})$ for the dense FFN. For $k = 2$, this is a 2x compute increase per token over a single dense FFN, which is then absorbed by reducing $d\_{\text{ff}}$ to $d\_{\text{ff}} / 2$ per expert to match the dense compute. Parameters grow by a factor of $E$ relative to the matched-compute dense FFN, so an MoE with $E = 8$ experts and $k = 2$ has roughly 4x the parameters at the same compute. Memory is the trade-off: the full set of expert weights must reside in HBM (or be sharded across devices), and the auxiliary loss plus the all-to-all communications add 10-30% wall-clock overhead. Inference KV cache is unchanged (MoE does not touch attention), but expert routing adds latency and the all-to-all becomes a bottleneck unless the experts are co-located.
{% end %}

The headline argument for MoE is the scaling efficiency: at fixed training compute, MoE models match or exceed dense models 2-4x their size in parameters and have lower per-token inference cost than the dense equivalent. The headline disadvantage is that the MoE forward pass is intrinsically harder to deploy: the model is bigger to load, the all-to-all needs a high-bandwidth interconnect, and the routing introduces variance in per-batch latency that does not sit well with strict-latency serving. Mixtral 8x7B (8 experts of 7B parameters with $k = 2$) and DeepSeek-V2/V3 (256 fine-grained experts plus 2 shared, $k = 8$) are the most prominent open MoE models at the time of writing, and both motivate the rest of the MoE-flavoured architectural choices that have emerged.

## Multimodal transformers and vision language models

The Transformer was designed for tokens. Once the input is a sequence of vectors, the model does not care whether they came from text, audio, video, or pixel patches. This is the entire reason multimodal modelling has been so dominated by Transformers: the architecture itself imposes no modality, and the only modality-specific decision is **how to turn the raw input into tokens**.

### Vision transformer

The Vision Transformer (ViT){{ reference(key="dosovitskiy2021vit") }} proves the point most directly. Given an image of resolution $H \times W$ with $C$ channels, partition it into $P \times P$ patches, flatten each patch to a vector of dimension $P^2 C$, and apply a learned linear projection to get a sequence of $N = HW / P^2$ token embeddings of dimension $d\_{\text{model}}$. Add a learnable positional embedding (now indexing 2D spatial positions instead of 1D temporal ones) and a special `[CLS]` token whose final representation is used for classification. Feed everything to a standard Transformer encoder.

The remarkable observation is that nothing else changes. No convolutions, no built-in translation invariance, no spatial inductive bias of any kind: the model has to learn that nearby patches are related, that patches can be translated, that channels carry colour information. With enough data (the original ViT used JFT-300M, $300$ million labelled images), the model learns these from scratch and matches or exceeds CNN accuracy on ImageNet at a fraction of the inference cost. With less data, the lack of inductive bias hurts, and hybrid architectures (the convolutional stem in CoAtNet, the windowed attention of Swin) close the gap by reintroducing some local structure.

A typical ViT uses patch size $P = 16$ on $224 \times 224$ images, giving $N = 196$ tokens, plus the `[CLS]` token. This is short enough that full attention is cheap. For higher-resolution images or video, where $N$ explodes quadratically with resolution, patch size has to grow or some form of sparse/hierarchical attention takes over.

### Contrastive image-text pretraining

CLIP{{ reference(key="radford2021clip") }} was the first model to demonstrate that the multimodal alignment between vision and language could be learned at scale entirely from naturally occurring image-text pairs scraped from the web. The architecture is two separate Transformers: a Vision Transformer that produces an image embedding $\mathbf{u} \in \mathbb{R}^{d}$, and a text Transformer that produces a text embedding $\mathbf{v} \in \mathbb{R}^{d}$. Both are projected to the same dimension. The training objective is a symmetric **contrastive loss**: for a batch of $B$ image-text pairs, the model is asked to identify the correct text for each image (and vice versa) out of the $B$ candidates in the batch.

{% equation(id="clip-loss") %}
\begin{aligned}
\mathcal{L}_{\text{CLIP}}
&= -\frac{1}{2 B} \sum_{i=1}^{B} \log \frac{e^{\tau\, \mathbf{u}_i^\top \mathbf{v}_i}}{\sum_{j=1}^{B} e^{\tau\, \mathbf{u}_i^\top \mathbf{v}_j}} \\
&\quad - \frac{1}{2 B} \sum_{i=1}^{B} \log \frac{e^{\tau\, \mathbf{u}_i^\top \mathbf{v}_i}}{\sum_{j=1}^{B} e^{\tau\, \mathbf{u}_j^\top \mathbf{v}_i}},
\end{aligned}
{% end %}

with a learned temperature $\tau$. This is the categorical cross-entropy from the [logistic regression](/blog/logistic-regression/) post applied symmetrically along both axes of the $B \times B$ similarity matrix. The diagonal entries are the matched pairs, and the loss pushes them up while pushing the off-diagonal entries (the hard negatives) down. With a large enough batch (the original CLIP used $B = 32{,}768$, which required sharding the contrastive loss across devices), every image is forced to be discriminated against tens of thousands of unrelated captions per step, and the resulting embeddings are good enough that **zero-shot classification** works directly: to classify an image into one of $K$ classes, encode each class name as a text prompt, encode the image, and pick the class whose text embedding has the highest cosine similarity with the image embedding.

CLIP itself is not generative. It is the **encoder backbone** that essentially every modern visual language model uses to turn pixels into a representation a text model can ingest.

### Vision language models (VLMs)

A **Vision Language Model** (VLM) takes one or more images plus a text prompt and produces text. The standard recipe assembles three pieces: a frozen vision encoder (typically a CLIP-style ViT), a frozen language model, and a small **connector** that translates the vision encoder's output into something the language model can consume. The differences between VLMs are almost entirely in the connector and in the training data.

**Linear or MLP projection** (LLaVA{{ reference(key="liu2023llava") }} and most modern open VLMs) is the simplest connector: pass the vision encoder's per-patch embeddings through a small MLP that maps from the vision dimension to the language model's input dimension, then prepend the resulting "vision tokens" to the text tokens. The combined sequence is fed to the language model exactly as if it were text. This is $\sim 1$ million parameters of new code for a $\sim 13$ B parameter model and is the cheapest possible thing that could work; empirically it works surprisingly well after instruction-tuning on a few hundred thousand image-text dialogue examples.

**Cross-attention** (Flamingo{{ reference(key="alayrac2022flamingo") }}) leaves the language model's text-token sequence alone but inserts new cross-attention layers between every few transformer blocks of the language model that attend from text tokens to vision tokens. This keeps the original text sequence length unchanged (vision tokens never enter the self-attention) and gives the model fine-grained control over when and how to look at the image. The cost is the additional cross-attention parameters and the added training complexity; the benefit is the ability to handle interleaved image-text inputs and many images per prompt without exploding the sequence length.

**Q-Former** (BLIP-2{{ reference(key="li2023blip2") }}) goes a step further by training a small Transformer that **distils** the variable-length vision encoder output into a fixed number of "query" tokens (typically 32) via cross-attention. The Q-Former is trained in two stages: first a vision-language alignment stage that teaches the queries to summarise the image, then a generation stage that connects the queries to a frozen language model. The argument for this complexity is parameter efficiency: a small Q-Former is much cheaper to train than full LLaVA-style fine-tuning, and the fixed number of query tokens decouples the language model's sequence length from the image resolution. The downside is the two-stage training and the loss of fine spatial detail when 196 patches are compressed to 32 queries.

The deeper trend in 2024 and 2025 is **native multimodal models** (Gemini, GPT-4o, Chameleon) that are trained from scratch on interleaved sequences of text tokens, image tokens (often produced by a discrete tokeniser like VQ-VAE), audio tokens, and so on, with a single decoder-only Transformer. The architecture collapses back to vanilla Transformer plus per-modality input projections and per-modality output heads, and the modality boundaries dissolve at the level of the model's attention.

### Image tokenisation trade-offs

The single design decision that most affects VLM behaviour is how an image becomes tokens, ie, the choice of patch size and the choice of summary mechanism.

**Fine-grained patches** (eg, $14 \times 14$ pixels at $224^2$ resolution, giving $256$ tokens) preserve more detail and let the model do OCR, count small objects, and read fine print. The cost is the linear blow-up of sequence length: a $1024 \times 1024$ image at the same patch size produces $5{,}329$ tokens, eating most of a 32k-token context window before any text is added.

**Tile-based encoding** (used in LLaVA-NeXT and many production VLMs) crops a high-resolution image into smaller tiles, encodes each independently, and concatenates the results. This recovers high-resolution behaviour without changing the patch size of the underlying ViT, at the cost of a sequence length that grows linearly with tile count.

**Perceiver-style summarisation**{{ reference(key="jaegle2021perceiver") }} (adopted in various forms by Q-Former and Flamingo's Perceiver Resampler) replaces the full per-patch sequence by a fixed pool of latent queries that cross-attend to the patches, decoupling sequence length from image resolution. This bounds the sequence-length cost but caps the spatial detail by the number of queries.

The right choice depends on the downstream task. Document understanding and OCR want fine-grained patches; chat about a single natural image is fine with Perceiver-style summaries; a model that needs to handle both well typically uses tile-based encoding plus a moderate patch size and accepts the resulting context-window cost. The trade-off space is the multimodal analogue of the long-context problem in pure language models, and the same toolbox (sliding-window attention, sparse attention, hierarchical pooling) carries over.

{% mathblock(kind="note", name="What to use when (multimodal)", id="vlm-defaults") %}
For zero-shot image classification or image-text retrieval, use a **CLIP-style dual encoder** with contrastive pretraining; the two-tower design keeps inference cheap because image and text are encoded independently. For a chat-style assistant that takes images and produces text, use the **LLaVA recipe**: a frozen CLIP ViT, a frozen LLM, and an MLP connector trained on instruction data; this is the modern default because of its simplicity and parameter efficiency. For applications with many interleaved images per prompt (interleaved dialogue, video understanding), prefer a **Flamingo-style cross-attention** model, which scales better in image count. For document understanding, OCR, or any task that needs spatial detail, accept the higher token cost and use **fine-grained patches with tile-based encoding**. For ambitious models that should handle text, image, audio, and video uniformly, build a **native multimodal decoder** with per-modality input tokenisers and per-modality output heads on a shared Transformer.
{% end %}

## Inference optimisation

The economics of a deployed Transformer are upside-down compared to training. Training is a fixed, one-off capital expense: the team rents a cluster, runs the job for weeks, and ships the final weights once. Inference is a recurring operating expense: every user request pays for a forward pass, and a popular model serves billions of them per day. A 2x speedup on training saves money once; a 2x speedup on inference saves it forever, on every request, for the life of the model. Unsurprisingly, the bulk of the systems work in the post-2022 LLM era has been on inference rather than on training.

Decoder inference also has a structural property that training does not: it is fundamentally **sequential**. To produce token $t+1$, the model must first see token $t$, which it just generated. There is no way to run more than one token of decoding in parallel within a single request, no matter how much hardware is available. The forward pass per token also has a peculiar arithmetic intensity: it loads the entire model's weights from HBM (tens of gigabytes for a modern LLM) to apply them to a single new token's embedding (a few kilobytes), making the per-token work overwhelmingly **memory-bound** in the sense of the [roofline model](#note-roofline). The H100's tensor cores sit at single-digit percent utilisation while the kernel waits for weights to arrive over HBM. This combination, sequential dependency plus memory-boundness, defines what every inference optimisation is trying to fight against.

Two ideas dominate the playbook. The **KV cache** removes redundant computation across decoding steps and is what makes modern decoder inference viable at all. **Speculative decoding** removes the sequential dependency itself, by drafting several tokens ahead in parallel with a smaller model and verifying them with the big one in a single forward pass. The two are orthogonal and routinely stacked.

### The KV cache

Consider what naive decoder inference would have to compute. To generate token $t+1$, the model runs a full forward pass over the prefix of $t$ tokens generated so far. The self-attention sublayer in each of $L$ layers projects the entire prefix into queries, keys, and values, and computes the $t \times t$ attention matrix from scratch. Most of this work is wasted: the keys and values of tokens $1, \ldots, t-1$ are identical to what they were when generating token $t$, since they are determined entirely by the prefix and the (frozen) model weights, not by the new token being added. Recomputing them at every step is the inference equivalent of recomputing the first $t-1$ Fibonacci numbers each time you need the $t$-th one.

The **KV cache** is the obvious memoisation. After computing the keys and values for the prefix once during the **prefill** phase (the initial forward pass over the user's prompt), store them in a buffer keyed by layer, head, and position. At each subsequent **decode** step, the model only computes the query, key, and value for the **single new token**, appends the new key and value to the cache, and runs attention with the new query against the entire cached key/value buffer.

{% mathblock(kind="proposition", name="KV cache complexity", id="kv-cache-cost") %}
Let $L$ be the number of layers, $h$ the number of attention heads, $d\_k$ the per-head key dimension, and $N$ the maximum generation length. With the KV cache, generating $N$ tokens autoregressively after the prefill costs $O(N^2 d)$ attention flops (matching a single training pass), $O(N \cdot P)$ flops for everything else (where $P$ is the parameter count), and stores $2 L h d\_k N$ floats per request in HBM. Without the cache, generation would cost $O(N^3 d)$ attention flops, ie, an extra factor of $N$.
{% end %}

{% mathblock(kind="proof", name="(sketch)", id="kv-cache-cost-proof") %}
With the cache, decode step $t$ runs each layer's attention as one query against $t$ keys, costing $O(t d)$ per layer and $O(L t d)$ per step. Summing over $t = 1$ to $N$ gives $\sum\_{t=1}^{N} L t d = O(L N^2 d)$ attention flops, which absorbed into $d$ is $O(N^2 d)$. The non-attention cost of every step is one forward pass over $P$ parameters (a constant per step regardless of position), totalling $O(N P)$. The cache itself stores, per layer and head, one $d\_k$-dimensional key and one $d\_k$-dimensional value per cached token, ie, $2 L h d\_k N$ floats in total. Without the cache, decode step $t$ recomputes attention from scratch on the full $t$-token prefix at $O(L t^2 d)$ per step, summing to $O(L N^3 d)$. $\square$
{% end %}

The cache is what made GPT-style inference economically viable. The size of the cache is also what now defines the operational profile of a serving stack. For a 70B-parameter model with $L = 80$, $h = 64$, $d\_k = 128$, and a context of $N = 32{,}768$ in float16, the per-request cache is

{% equation(id="kv-cache-size") %}
2 \cdot 80 \cdot 64 \cdot 128 \cdot 32{,}768 \cdot 2~\text{bytes} \approx 86~\text{GB},
{% end %}

ie, more than a single H100's 80 GB of HBM, before counting the model weights themselves. Two structural responses follow. First, **GQA and MQA** (covered earlier in the variants section) shrink the $h$ factor by sharing keys and values across query heads, with $g = 8$ shrinking the cache by 8x at near-zero quality cost. Second, **paged attention** (vLLM and friends) borrows the virtual-memory abstraction from operating systems: the KV cache is stored in fixed-size pages of (say) 16 tokens, and a per-request page table maps logical token positions to physical pages. This eliminates internal fragmentation (one request never has to reserve $N$-token-sized contiguous memory just because it might generate that long), permits **prefix sharing** between requests that begin with the same system prompt, and is the foundation of every modern high-throughput inference server. Quantising the cache itself to 8-bit or 4-bit precision is the third lever, halving or quartering the footprint at a small quality cost.

{% mathblock(kind="note", name="Prefill vs decode", id="prefill-decode") %}
The two phases of inference look like completely different workloads to the hardware. **Prefill** processes the entire user prompt in one forward pass: the attention sublayer runs at the same arithmetic intensity as training (matmuls of shape $N \times d$, compute-bound, tensor cores saturated), and a long prompt finishes in milliseconds even on a single GPU. **Decode** processes one token at a time: each forward pass is an $O(P)$ memory-bound traversal of the model weights to advance a $1 \times d$ vector, and most of the GPU sits idle. The standard production trick is **continuous batching**: many concurrent requests are batched together at the decode step so that the model weights, which had to be loaded from HBM anyway, are amortised over many sequences instead of one. This raises the per-request latency slightly but multiplies throughput, and is the second-biggest lever after the KV cache itself.
{% end %}

### Speculative decoding

The KV cache makes each decode step as cheap as it can possibly be, but the $O(N)$ sequential dependency between steps remains: to produce $N$ tokens we must run $N$ sequential forward passes of the big model, and the wall-clock time scales linearly with the number of tokens generated. **Speculative decoding** attacks the sequential dependency directly. The trick is that decode is memory-bound: a single forward pass that processes one token and a forward pass that processes (say) five tokens take essentially the same wall-clock time, because both are bottlenecked on loading the model weights from HBM, not on the $5\times$ more arithmetic the second pass does. If we could somehow propose five candidate tokens ahead and verify all of them in one forward pass of the big model, we would generate up to five tokens in the time of one.

The catch is that we don't know what tokens to verify, since the autoregressive distribution depends on the actual generated prefix. Speculative decoding solves this by introducing a small **draft model** (typically a much smaller LLM from the same family, or a few extra heads on the big model itself) that proposes a candidate continuation cheaply, and using the big **target model** to verify the proposal in parallel. The crucial property of the verification scheme is that the resulting samples are **distributed exactly the same** as samples drawn from the target model alone: speculative decoding is an exact algorithm, not an approximation, with no quality loss whatsoever.

{% mathblock(kind="proposition", name="Speculative decoding sampler", id="spec-dec-sampler") %}
Let $p$ be the target model's next-token distribution and $q$ the draft model's. Suppose the draft proposes a token $\tilde{x} \sim q(\cdot)$. Accept $\tilde{x}$ with probability $\min(1,\, p(\tilde{x}) / q(\tilde{x}))$. If rejected, sample a replacement $x$ from the **residual distribution**
$$\begin{aligned}
p\_{\text{res}}(x)
&= \frac{\max\!\big(0,\, p(x) - q(x)\big)}{\sum\_{x'} \max\!\big(0,\, p(x') - q(x')\big)}.
\end{aligned}$$
The accepted (or replacement) token is distributed exactly according to $p$.
{% end %}

{% mathblock(kind="proof", name="(sketch)", id="spec-dec-proof") %}
For any target token $x$, the probability that the algorithm outputs $x$ is the sum of two disjoint events: the draft proposed $x$ and was accepted, or the draft was rejected (whatever it proposed) and the residual sampler returned $x$. The first event has probability $q(x) \cdot \min(1, p(x)/q(x)) = \min(p(x), q(x))$. The total rejection probability is $1 - \sum\_{x'} \min(p(x'), q(x')) = \sum\_{x'} \max(0, p(x') - q(x'))$, and conditional on rejection the sampler returns $x$ with probability $\max(0, p(x) - q(x)) / \sum\_{x'} \max(0, p(x') - q(x'))$, contributing $\max(0, p(x) - q(x))$ to the marginal. Adding the two: $\min(p(x), q(x)) + \max(0, p(x) - q(x)) = p(x)$, since the two cases (whether $p(x) \leq q(x)$ or not) cover all values of $x$. The output is exactly $p$-distributed. $\square$
{% end %}

The full algorithm extends this one-token sampler to $K$ tokens at a time. The draft model autoregressively proposes a sequence of $K$ candidate tokens $\tilde{x}\_1, \ldots, \tilde{x}\_K$, each cheap because the draft is small. The target model then runs **a single forward pass** over the entire candidate sequence (using the standard parallel-attention trick that made training a causal model in $O(N^2)$ instead of $O(N)$ possible) and produces target distributions $p\_1, \ldots, p\_K$ at every position. The candidates are accepted left to right via the rejection rule above, stopping at the first rejection. If all $K$ are accepted, a free $(K+1)$-th token is sampled from $p\_{K+1}$, since it is computed as a side effect of the same forward pass. If the $j$-th candidate is rejected, tokens $\tilde{x}\_1, \ldots, \tilde{x}\_{j-1}$ are kept and a replacement $x\_j$ is sampled from the residual distribution.

The expected speedup depends entirely on the **acceptance rate**, ie, how well the draft model approximates the target. Empirically, when the draft is a 1-2B model from the same family as a 70B target, $60$-$80\%$ of draft tokens are accepted, giving a $2$-$3\times$ end-to-end wall-clock speedup with no quality loss. The draft can also be the target itself with a few extra prediction heads (Medusa) or a small set of pre-computed n-gram completions of the prefix (prompt lookup decoding) for the prompts that are highly templated or repetitive.

{% mathblock(kind="note", name="Cost analysis (speculative decoding)", id="spec-dec-cost") %}
Per round, the draft model runs $K$ sequential forward passes at cost $K \cdot c\_q$ where $c\_q$ is the draft's per-step cost. The target model runs **one** forward pass over the $K$-token candidate sequence at cost $c\_p$ (the same $c\_p$ as a single decode step, because the bottleneck is loading the target's weights from HBM, not the $K \times$ more arithmetic on activations). If the average number of accepted tokens per round is $\alpha \cdot K$ for some $\alpha \in (0, 1]$, the per-token wall-clock cost is $(K c\_q + c\_p) / (\alpha K + 1)$, compared to $c\_p$ for vanilla decoding. The speedup ratio is $c\_p / [(K c\_q + c\_p) / (\alpha K + 1)] = (\alpha K + 1) / (1 + K c\_q / c\_p)$. With $c\_q / c\_p \approx 0.05$ (a 1B draft for a 20B target), $K = 5$, and $\alpha = 0.7$, the ratio is roughly $(3.5 + 1) / (1 + 0.25) = 3.6\times$. The two failure modes are a draft that is too slow ($c\_q$ too close to $c\_p$, eats the savings) or a draft that is too inaccurate ($\alpha$ too low, most candidates get rejected and the draft work is wasted).
{% end %}

Speculative decoding is now the default in every major open-source serving stack (vLLM, TensorRT-LLM, SGLang). The combination of KV cache, paged attention, GQA, FlashAttention, continuous batching, and speculative decoding is what turns a research codebase that does $5$ tokens/second per GPU into a production system that does $500$, on the same hardware and with bit-identical outputs.

## Training considerations

A few details of how Transformers are trained are unobvious enough that they are part of the architecture in spirit if not in code.

### Learning-rate warmup

The original paper used the schedule

{% equation(id="warmup-schedule") %}
\eta_t = d_{\text{model}}^{-0.5}\, \min\!\left(t^{-0.5},\, t \cdot t_{\text{warmup}}^{-1.5}\right),
{% end %}

which linearly increases the learning rate for the first $t\_{\text{warmup}}$ steps (4000 in the original) and then decays as $1/\sqrt{t}$. The motivation is that the residual stream of an untrained Transformer is dominated by the (small) random initialisation of the sublayers, so the early gradient signal is noisy and large steps destabilise training. The warmup gives the layer norm statistics and the attention patterns time to stabilise before the optimiser starts moving aggressively. Pre-LN architectures tolerate skipping the warmup, but most modern recipes still use a short warmup followed by cosine decay (the same schedule discussed in the [neural network](/blog/neural-network/) post).

### Label smoothing

Cross-entropy training with one-hot targets pushes the model to assign probability 1 to the true class and 0 to all others, which encourages large pre-softmax logits and overconfident predictions. **Label smoothing** with strength $\varepsilon \in [0, 0.1]$ replaces the one-hot target by a mixture: $1 - \varepsilon$ on the true class and $\varepsilon / (K - 1)$ on each of the other $K - 1$ classes. The model is no longer rewarded for arbitrarily large logits and the resulting calibration is better. The trade-off is a slight increase in training perplexity (since the loss is no longer minimised by perfect prediction) but a typically improved test performance and downstream metric. The Vaswani paper used $\varepsilon = 0.1$; modern LLM training varies.

### Adam, AdamW, and their friends

The optimiser of choice for Transformers is **AdamW** (defined in the [neural network](/blog/neural-network/) post) with $\beta\_1 = 0.9$, $\beta\_2 = 0.95$ to $0.98$ depending on the model, and a small weight decay $\lambda \in [0.01, 0.1]$ applied to all parameters except biases and layer norm scales. The lower $\beta\_2$ relative to the standard Adam default of $0.999$ is a quirk of large-scale training: the second-moment estimate becomes biased when the gradient norm changes quickly through training, and a slightly faster decay tracks the change better.

## Practical pitfalls

Three issues are characteristic of training Transformers and worth flagging up front.

The first is **attention collapse**, where many heads converge to attending uniformly or to a single position (often the first token) and stop contributing useful signal. This is more of a symptom than a cause and usually points to insufficient model capacity, a learning rate that decayed too aggressively, or a regularisation strength that is too high. Removing the collapsed heads at inference time often costs essentially nothing in quality, which is the basis for **head pruning**.

The second is **NaN losses from the softmax**. With float16 training, the maximum logit before the softmax can overflow to $+\infty$ and the entire row becomes NaN after the normalisation. The standard defence is to upcast the attention scores to float32 before the softmax and downcast the output, which costs a small amount of memory but is essentially mandatory in mixed-precision Transformer training. FlashAttention handles this internally with the online softmax, which is naturally numerically stable.

The third is the **KV cache running out of memory at inference**. A 70B-parameter model with a 32k context and float16 KV cache requires tens of GB of HBM per request, and a single GPU typically holds a handful of concurrent requests. Long-context inference budgets are dominated by this number, and reducing it (via GQA, KV cache quantisation, or paged attention) is currently the most active subfield of inference engineering.

{% mathblock(kind="note", name="What to use when", id="attn-defaults") %}
For a fresh Transformer from scratch, use **pre-LN with RMSNorm**, **RoPE** for positions, **SwiGLU** in the FFN, **grouped-query attention** with $g \approx h / 8$, and **FlashAttention** as the kernel; train with **AdamW** at $\beta\_2 = 0.95$, a short linear warmup followed by cosine decay, weight decay $\approx 0.05$ on non-norm parameters, and **label smoothing** $\varepsilon = 0.1$ for translation or $\varepsilon = 0$ for language modelling. Use the **encoder-decoder** flavour for tasks with a clear input-output sequence pair (translation, summarisation), the **encoder-only** flavour for representation learning (classification, retrieval), and the **decoder-only** flavour for autoregressive generation (language modelling, instruction following). Reach for **sparse or sliding-window attention** only when the natural sequence length exceeds what FlashAttention can handle on the available hardware and the dependencies are demonstrably local; reach for **linear attention** only when even sparse patterns are infeasible and a mild quality regression is acceptable.
{% end %}
