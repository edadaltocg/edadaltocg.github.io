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

## From Recurrence to Attention

The [neural network](/blog/neural-network/) post built models that map a single fixed-size input vector to a single fixed-size output. Sequences (sentences, audio frames, time series) do not fit that mould: their length varies and their structure is sequential. The two pre-Transformer answers to this were recurrent networks (RNNs, LSTMs, GRUs) that consume the sequence one token at a time while carrying a hidden state, and one-dimensional convolutions that slide a fixed-width filter along the sequence. Both have a structural cost. Recurrence forces a strictly serial computation: the state at step $t$ depends on the state at step $t-1$, so an $N$-token sequence requires $N$ sequential steps and the gradient must traverse all of them on the backward pass, which both runs slowly on parallel hardware and decays exponentially through depth (the vanishing-gradient problem from the [neural network](/blog/neural-network/) post, restated in time). Convolutions parallelise across positions but communicate only within the kernel, so two tokens $k$ apart in the sequence can only influence each other after roughly $k / w$ stacked convolutions of width $w$, ie, the path length grows with the distance.

Attention, as introduced by Bahdanau, Cho, and Bengio for neural machine translation{{ reference(key="bahdanau2015attention") }} and refined by Luong, Pham, and Manning{{ reference(key="luong2015effective") }}, sidesteps both costs at once. Every output position is a learned weighted average of every input position, where the weights are computed from the inputs themselves. The path length between any two tokens is exactly one (every output sees every input directly), and the per-layer computation parallelises trivially across the sequence dimension because the weights at one position do not depend on those at any other. The only price is that the per-layer cost is quadratic in the sequence length: with $N$ tokens, computing all pairwise weights costs $O(N^2)$ time and memory. That price is what almost every modern variant of attention is, in one way or another, trying to negotiate down.

The Transformer of Vaswani et al.{{ reference(key="vaswani2017attention") }} took the additional step of removing recurrence and convolution entirely, building the entire sequence model out of attention plus a small position-wise feed-forward network. The resulting architecture is what now powers essentially every modern language model, vision transformer, speech model, and multimodal model. What follows is a dissection of every design decision in that paper, the algebra behind each one, and the family of variants that have grown around it.

## Scaled Dot-Product Attention

The atomic operation of the Transformer takes three sets of vectors as input. A set of $N$ **queries** $\mathbf{Q} \in \mathbb{R}^{N \times d\_k}$, a set of $M$ **keys** $\mathbf{K} \in \mathbb{R}^{M \times d\_k}$, and a set of $M$ **values** $\mathbf{V} \in \mathbb{R}^{M \times d\_v}$. The interpretation is borrowed from information retrieval: each query is matched against every key to produce a similarity score, the scores are turned into a probability distribution over the keys, and the output for that query is the corresponding probability-weighted average of the values. In the symmetric case $N = M$ used in self-attention, the same set of vectors plays all three roles after three different learned projections.

{% mathblock(kind="definition", name="Scaled dot-product attention", id="sdpa") %}
$$\mathrm{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \mathrm{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d\_k}}\right) \mathbf{V},$$
with the softmax applied row-wise over the $M$ keys. The output has shape $\mathbb{R}^{N \times d\_v}$.
{% end %}

The product $\mathbf{Q}\mathbf{K}^\top \in \mathbb{R}^{N \times M}$ holds the raw similarity scores: entry $(i, j)$ is the inner product $\mathbf{q}\_i^\top \mathbf{k}\_j$ between the $i$-th query and the $j$-th key. Dividing by $\sqrt{d\_k}$ controls the variance of these scores. Applying the row-wise [softmax](/blog/softmax-function/) turns each row into a probability distribution over the $M$ keys, and right-multiplying by $\mathbf{V}$ takes the corresponding weighted average of value vectors.

Three design decisions are baked into this single line, and each one is worth pulling apart.

### Why the Dot Product

The original Bahdanau attention used an **additive** scoring function $a(\mathbf{q}, \mathbf{k}) = \mathbf{w}^\top \tanh(\mathbf{W}\_q \mathbf{q} + \mathbf{W}\_k \mathbf{k})$, ie, a tiny one-hidden-layer MLP applied to each query-key pair. This is more expressive than a plain inner product but disastrous for parallel hardware, since computing every pair requires an explicit double loop or an expanded $N \times M \times d\_k$ tensor. The dot product collapses the entire pairwise score table into a single matrix multiplication $\mathbf{Q}\mathbf{K}^\top$, which is exactly the operation modern accelerators are best at.{% sidenote(id="sdpa-vs-additive") %}Vaswani et al. report that additive attention slightly outperforms dot-product attention at small $d\_k$, but the gap closes (and reverses) at the scales used in real Transformers, while the throughput advantage of dot-product attention only grows.{% end %} The Transformer accepts the loss in expressivity per attention head and recovers it by stacking many heads and many layers.

### Why Divide by √dₖ

This is the "scaled" half of the name and is the most easily missed design choice. The motivation is variance control under the [softmax](/blog/softmax-function/).

{% mathblock(kind="proposition", name="Variance of the unscaled dot product", id="dot-variance") %}
Let $\mathbf{q}, \mathbf{k} \in \mathbb{R}^{d\_k}$ have independent entries with mean zero and variance one. Then
$$\mathbb{E}\!\left[\mathbf{q}^\top \mathbf{k}\right] = 0, \qquad \mathrm{Var}\!\left(\mathbf{q}^\top \mathbf{k}\right) = d\_k.$$
{% end %}

{% mathblock(kind="proof", name="", id="dot-variance-proof") %}
Write $\mathbf{q}^\top \mathbf{k} = \sum\_{j=1}^{d\_k} q\_j k\_j$. By independence, $\mathbb{E}[q\_j k\_j] = \mathbb{E}[q\_j] \mathbb{E}[k\_j] = 0$, so the mean is zero. The summands are uncorrelated, so $\mathrm{Var}(\sum\_j q\_j k\_j) = \sum\_j \mathrm{Var}(q\_j k\_j)$. For the per-term variance, $\mathrm{Var}(q\_j k\_j) = \mathbb{E}[q\_j^2 k\_j^2] - \mathbb{E}[q\_j k\_j]^2 = \mathbb{E}[q\_j^2] \mathbb{E}[k\_j^2] - 0 = 1 \cdot 1 = 1$. Summing $d\_k$ such terms gives variance $d\_k$. $\square$
{% end %}

A typical $d\_k$ in a Transformer is $64$, so unscaled scores have standard deviation $8$. Feeding such large logits to the softmax pushes its output close to a one-hot vector, ie, the distribution concentrates on the single largest key. The gradient of the softmax in this regime is nearly zero in every direction except one, so attention freezes early in training and the optimiser has nothing to work with. Dividing by $\sqrt{d\_k}$ rescales the standard deviation back to $1$, keeping the softmax in its dynamic range where every key still contributes a non-negligible probability and the gradients flow to all of them.

### Why the Softmax

Softmax is the canonical map from real scores to a probability distribution and was derived in the [softmax](/blog/softmax-function/) post as the unique exponential-family link that makes the score-to-distribution map invertible up to additive shifts. Two of its properties are load-bearing here. First, the output sums to one, so the attention output is a convex combination of value vectors and lives in the same affine span as $\mathbf{V}$. This is what lets the attention operation be interpreted as "look up a value, with soft selection". Second, the softmax is differentiable everywhere with the well-conditioned Jacobian computed in the [softmax](/blog/softmax-function/) post, so backpropagation through attention is numerically stable when combined with the log-sum-exp trick. Hard alternatives (argmax, sparsemax) sacrifice one or both of these.

{% mathblock(kind="note", name="Cost analysis (one attention call)", id="sdpa-cost") %}
Time is $O(N M d\_k + N M d\_v)$, dominated by the two large matrix products: $\mathbf{Q}\mathbf{K}^\top$ at $O(N M d\_k)$ flops and the multiplication of the $N \times M$ attention matrix by $\mathbf{V}$ at $O(N M d\_v)$. The softmax itself is a row-wise $O(N M)$ sweep, absorbed into the lower-order terms. Memory is $O(N M)$ for the attention matrix, which is the term that bites: at $N = M = 16{,}384$ in float32, a single attention matrix already eats $1$ GB of memory, before counting any other tensor. This single number is the entire reason for the existence of FlashAttention, sparse attention, and linear attention discussed later.
{% end %}

## Multi-Head Attention

A single attention call has a fixed amount of expressivity: it produces one weighted average per query, with weights driven by a single similarity geometry. Different relationships between tokens (syntactic dependencies, semantic similarity, positional proximity) plausibly want to be modelled with different similarity geometries, and forcing all of them through a single softmax averages them away.

The fix is **multi-head attention**: project the inputs into $h$ different subspaces, run an independent scaled dot-product attention in each subspace, and concatenate the results.

{% mathblock(kind="definition", name="Multi-head attention", id="mha") %}
For $h$ heads with per-head dimensions $d\_k$ and $d\_v$, define learned projections $\mathbf{W}\_i^Q \in \mathbb{R}^{d\_{\text{model}} \times d\_k}$, $\mathbf{W}\_i^K \in \mathbb{R}^{d\_{\text{model}} \times d\_k}$, $\mathbf{W}\_i^V \in \mathbb{R}^{d\_{\text{model}} \times d\_v}$ for $i = 1, \ldots, h$, and an output projection $\mathbf{W}^O \in \mathbb{R}^{h d\_v \times d\_{\text{model}}}$. Then
$$\mathrm{head}\_i = \mathrm{Attention}(\mathbf{X}\mathbf{W}\_i^Q,\, \mathbf{X}\mathbf{W}\_i^K,\, \mathbf{X}\mathbf{W}\_i^V),$$
$$\mathrm{MultiHead}(\mathbf{X}) = \mathrm{Concat}(\mathrm{head}\_1, \ldots, \mathrm{head}\_h)\, \mathbf{W}^O.$$
The standard choice in the original paper is $d\_{\text{model}} = 512$, $h = 8$, $d\_k = d\_v = d\_{\text{model}} / h = 64$.
{% end %}

The deliberate choice $d\_k = d\_{\text{model}} / h$ keeps the total parameter count and total compute identical to a single attention call at full $d\_{\text{model}}$, while allowing the model to attend to information from $h$ different representation subspaces in parallel. Empirically the per-head attention patterns specialise: in a trained language Transformer, some heads track syntactic dependencies, others track coreference, others track positional offsets, others act as "no-op" heads that attend almost uniformly. This division of labour is the entire justification for stacking heads rather than just making a single attention head wider.

The output projection $\mathbf{W}^O$ is what mixes information across heads. Without it, the per-head outputs would be concatenated into disjoint slices of the next layer's input vector, never interacting. With it, each output coordinate is a learned linear combination of every head's contribution.

{% mathblock(kind="note", name="Multi-head as a structured projection", id="mha-as-projection") %}
A useful way to read multi-head attention is as a single attention operation in $\mathbb{R}^{d\_\text{model}}$ whose query-key product matrix is constrained to be block-diagonal across head boundaries, after a learned change of basis $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V$. The block structure is what gives each head its own similarity geometry, and the output projection $\mathbf{W}^O$ undoes the block structure so subsequent layers see a fully mixed representation again. Multi-head is, in this light, neither a sum nor a product of attentions but a structured factorisation of a single large attention.
{% end %}

## Positional Encoding

There is one important property of attention that the discussion so far has carefully avoided.

{% mathblock(kind="proposition", name="Permutation equivariance of attention", id="perm-eq") %}
Let $\mathbf{P} \in \\{0, 1\\}^{N \times N}$ be a permutation matrix acting on the rows of the inputs. Then
$$\mathrm{Attention}(\mathbf{P}\mathbf{Q}, \mathbf{P}\mathbf{K}, \mathbf{P}\mathbf{V}) = \mathbf{P}\, \mathrm{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}).$$
{% end %}

{% mathblock(kind="proof", name="", id="perm-eq-proof") %}
The score matrix becomes $(\mathbf{P}\mathbf{Q})(\mathbf{P}\mathbf{K})^\top = \mathbf{P}\mathbf{Q}\mathbf{K}^\top \mathbf{P}^\top$, ie, both rows and columns are permuted. The row-wise softmax is itself row-equivariant and commutes with the row permutation: $\mathrm{softmax}(\mathbf{P}\mathbf{S}\mathbf{P}^\top) = \mathbf{P}\, \mathrm{softmax}(\mathbf{S})\, \mathbf{P}^\top$ when applied row-wise (the row permutation $\mathbf{P}$ reshuffles which row is normalised, and the column permutation $\mathbf{P}^\top$ reshuffles within each row before normalising; both pass through the elementwise exponential). Right-multiplying by $\mathbf{P}\mathbf{V}$ gives $\mathbf{P}\, \mathrm{softmax}(\mathbf{S})\, \mathbf{V}$, ie, the output is permuted in the same way as the inputs. $\square$
{% end %}

This is a problem for sequence modelling. If we shuffle the words in a sentence, attention produces a correspondingly shuffled output, but the underlying meaning of the sentence depends on the word order. To make the model order-aware, we have to inject the position of each token into the input itself.

The Transformer adds a fixed **positional encoding** vector to each token embedding before any attention layer. The chosen function is sinusoidal: for position $\mathrm{pos} \in \\{0, 1, \ldots, N-1\\}$ and dimension $i \in \\{0, 1, \ldots, d\_{\text{model}}-1\\}$,

{% equation(id="sinusoidal-pe") %}
\mathrm{PE}_{\mathrm{pos},\, 2i} = \sin\!\left(\frac{\mathrm{pos}}{10000^{2i/d_{\text{model}}}}\right), \qquad \mathrm{PE}_{\mathrm{pos},\, 2i+1} = \cos\!\left(\frac{\mathrm{pos}}{10000^{2i/d_{\text{model}}}}\right).
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

## The Encoder Block

The encoder takes a sequence of token embeddings and outputs a sequence of contextual representations of the same length. It is a stack of $L$ identical blocks, each with the same internal structure. We discuss the structure once.

A single encoder block contains two sublayers: multi-head self-attention and a position-wise feed-forward network. Each sublayer is wrapped in a residual connection followed by layer normalisation, ie, the output of each sublayer is $\mathrm{LayerNorm}(\mathbf{x} + \mathrm{Sublayer}(\mathbf{x}))$.

### Self-Attention

In **self-attention**, the queries, keys, and values are all projections of the same input sequence. For an encoder input $\mathbf{X} \in \mathbb{R}^{N \times d\_{\text{model}}}$,

{% equation(id="self-attention") %}
\mathrm{SelfAttn}(\mathbf{X}) = \mathrm{MultiHead}(\mathbf{X}\mathbf{W}^Q,\, \mathbf{X}\mathbf{W}^K,\, \mathbf{X}\mathbf{W}^V),
{% end %}

with the same $\mathbf{X}$ on the left of all three projections. The interpretation is that every token attends to every other token in the same sequence to form its contextual representation, and the model has $h$ different learned views of how to do that mixing.

### Position-wise Feed-Forward Network

After self-attention, every token passes independently through a small two-layer MLP, the same MLP applied at every position.

{% equation(id="ffn") %}
\mathrm{FFN}(\mathbf{x}) = \mathbf{W}_2\, \phi\!\left(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1\right) + \mathbf{b}_2,
{% end %}

with $\mathbf{W}\_1 \in \mathbb{R}^{d\_{\text{ff}} \times d\_{\text{model}}}$, $\mathbf{W}\_2 \in \mathbb{R}^{d\_{\text{model}} \times d\_{\text{ff}}}$, $\phi$ a non-linearity (originally ReLU, GELU in BERT and beyond), and the standard expansion ratio $d\_{\text{ff}} = 4\, d\_{\text{model}}$. The "position-wise" qualifier emphasises that the same MLP is applied independently to each token, so this sublayer adds no cross-token interaction.

The natural question is: why bother with the FFN at all? Self-attention is already a learned mixing of token vectors, and stacking attention layers gives you depth. The answer is that attention is a fundamentally **linear** operation in its values: each output is a convex combination of value vectors, and the only non-linearity is the softmax over scores, which acts on similarities and not on representations. Without the FFN, an arbitrarily deep attention stack collapses (per-head, before mixing) into a single linear map of the values, with weights that depend on the inputs but values that do not. The FFN is what gives every Transformer block its per-position non-linearity, and in modern parameter accounting the FFN holds the majority of the model's parameters: at $d\_{\text{ff}} = 4 d\_{\text{model}}$, the FFN has $8 d\_{\text{model}}^2$ parameters per block versus $4 d\_{\text{model}}^2$ for the four projections of attention.

### Residual Connections

Each sublayer is wrapped in a residual connection: $\mathbf{x} \mapsto \mathbf{x} + \mathrm{Sublayer}(\mathbf{x})$. The motivation is identical to that of ResNets in vision. The forward pass establishes a default route (the identity) past every sublayer, so an untrained or noisy sublayer initially contributes nothing and the signal still propagates through the stack. The backward pass benefits in the same way: the gradient flowing backward through a residual sublayer is $\mathbf{1} + \partial\mathrm{Sublayer}/\partial \mathbf{x}$, so the gradient never shrinks below the identity contribution, no matter what the sublayer's Jacobian looks like. This is the single most important architectural reason that 100-layer Transformers are trainable at all.

### Layer Normalisation

The original paper places layer normalisation (defined in the [neural network](/blog/neural-network/) post) after the residual addition, ie, $\mathrm{LayerNorm}(\mathbf{x} + \mathrm{Sublayer}(\mathbf{x}))$. This is now called **post-LN** to distinguish it from the **pre-LN** variant $\mathbf{x} + \mathrm{Sublayer}(\mathrm{LayerNorm}(\mathbf{x}))$ used in essentially every modern model. The trade-off is discussed in the variants section; for now, the layer normalisation is what keeps the activations in a stable range as the residual stream grows with depth.

Putting everything together, the full encoder block computes

{% equation(id="encoder-block") %}
\begin{aligned}
\mathbf{x}' &= \mathrm{LayerNorm}\big(\mathbf{x} + \mathrm{SelfAttn}(\mathbf{x})\big), \\
\mathbf{x}'' &= \mathrm{LayerNorm}\big(\mathbf{x}' + \mathrm{FFN}(\mathbf{x}')\big),
\end{aligned}
{% end %}

with the same block stacked $L$ times (the original paper used $L = 6$ for both the encoder and the decoder; modern models go up to $L = 96$ and beyond).

## The Decoder Block

The decoder mirrors the encoder but adds two complications. It must generate its output sequence one token at a time, conditioned on the encoder's output, which forces two structural changes.

### Masked Self-Attention

At training time the decoder is fed the entire target sequence in parallel for efficiency, but it must not be allowed to peek at future tokens when predicting the next one. The fix is **causal masking**: in the self-attention of the decoder, every query at position $i$ can only attend to keys at positions $j \leq i$. Operationally, the score matrix gets an additive mask before the softmax,

{% equation(id="causal-mask") %}
\mathrm{MaskedAttn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \mathrm{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right) \mathbf{V},
{% end %}

with $\mathbf{M}\_{i,j} = 0$ if $j \leq i$ and $\mathbf{M}\_{i,j} = -\infty$ otherwise. The infinities push the softmax of forbidden positions to exactly zero, removing them from the weighted average. This single trick is what lets us train a causal Transformer in $O(N^2)$ parallel time instead of $O(N)$ sequential time, while still producing output that is causally consistent with autoregressive decoding at inference.

### Cross-Attention

After masked self-attention, the decoder block inserts a third sublayer that performs **cross-attention** to the encoder's output: queries come from the decoder, keys and values come from the encoder.

{% equation(id="cross-attention") %}
\mathrm{CrossAttn}(\mathbf{X}_{\text{dec}}, \mathbf{X}_{\text{enc}}) = \mathrm{MultiHead}\big(\mathbf{X}_{\text{dec}} \mathbf{W}^Q,\, \mathbf{X}_{\text{enc}} \mathbf{W}^K,\, \mathbf{X}_{\text{enc}} \mathbf{W}^V\big).
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

## A Worked Example by Hand

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

## Variants of Attention

The original Transformer fixes a single answer to every design question. Each of those answers has been revisited, and a small library of attention variants now coexists in production. They split cleanly into two families: those that change **what** is attended to (sparse, local, or low-rank patterns to escape the $O(N^2)$ wall) and those that change **how** the attention is computed (algorithmic and hardware-aware reformulations of the same mathematics).

### Multi-Query and Grouped-Query Attention

The bottleneck during autoregressive generation is not training but **inference**, and specifically the size of the **KV cache**. At decoding step $t$, the model needs the keys and values of all previous $t-1$ tokens to compute attention for the new query. Rather than recomputing them, they are cached. The cache size is $2 \cdot L \cdot N \cdot h \cdot d\_k$ for $L$ layers, $h$ heads, sequence length $N$. For a typical large model this is gigabytes of state per generation, and reading it from memory at every decoding step quickly becomes the bottleneck.

**Multi-Query Attention** (MQA){{ reference(key="shazeer2019fast") }} responds by sharing a single key and value across all heads, while keeping per-head queries:

{% equation(id="mqa") %}
\mathrm{head}_i = \mathrm{Attention}(\mathbf{X}\mathbf{W}_i^Q,\, \mathbf{X}\mathbf{W}^K,\, \mathbf{X}\mathbf{W}^V).
{% end %}

The KV cache shrinks by a factor of $h$ (so $8\times$ smaller for the standard 8-head model), at the cost of reducing the per-head expressivity of the keys and values. Empirically MQA hurts model quality slightly. **Grouped-Query Attention** (GQA){{ reference(key="ainslie2023gqa") }} interpolates: the $h$ query heads are partitioned into $g$ groups, each of which shares one key/value pair. Setting $g = h$ recovers vanilla multi-head attention, $g = 1$ recovers MQA, and intermediate values (LLaMA-2 70B uses $g = 8$ with $h = 64$) capture most of the inference speedup with negligible quality loss.

### Sliding-Window and Sparse Attention

The full $N \times N$ attention matrix is wasteful when most useful interactions are local. **Sliding-window attention** restricts each query to attend only to the $w$ nearest tokens on either side, ie, $\mathbf{M}\_{i,j} = -\infty$ when $|i - j| > w$. The compute and memory cost drops from $O(N^2)$ to $O(N w)$, and a stack of $L$ layers gives an effective receptive field of $L \cdot w$, ie, distant tokens still influence each other through depth, like in a CNN.

The Longformer{{ reference(key="beltagy2020longformer") }} combines a sliding window with a small number of **global tokens** that attend to everything and that everything attends to (typically the [CLS] token and any task-specific markers), so the model retains a path of length two between any two positions. BigBird{{ reference(key="zaheer2020bigbird") }} adds **random attention** edges to the same recipe, and proves that the resulting sparse pattern is a universal sequence-to-sequence approximator, ie, no expressivity is lost in principle. The Sparse Transformer{{ reference(key="child2019sparse") }} uses fixed strided patterns inspired by 2D convolutions for image-like sequences. All of these are valuable when the natural sequence length is too long for full attention but the dependencies are mostly local.

### Linear Attention

A different angle on the $O(N^2)$ problem is to approximate the softmax kernel by a feature map that lets us reorder the computation. Write attention as $\mathrm{softmax}(\mathbf{Q}\mathbf{K}^\top) \mathbf{V}$ and observe that, for any feature map $\phi$ such that $\mathrm{softmax}(\mathbf{q}\_i^\top \mathbf{k}\_j) \approx \phi(\mathbf{q}\_i)^\top \phi(\mathbf{k}\_j) / Z(\mathbf{q}\_i)$, we can rewrite the output as

{% equation(id="linear-attn") %}
\mathbf{o}_i = \frac{\sum_{j} \phi(\mathbf{q}_i)^\top \phi(\mathbf{k}_j)\, \mathbf{v}_j}{\sum_{j} \phi(\mathbf{q}_i)^\top \phi(\mathbf{k}_j)} = \frac{\phi(\mathbf{q}_i)^\top \big(\sum_{j} \phi(\mathbf{k}_j)\, \mathbf{v}_j^\top\big)}{\phi(\mathbf{q}_i)^\top \big(\sum_{j} \phi(\mathbf{k}_j)\big)}.
{% end %}

The associativity matters: instead of forming the $N \times N$ matrix $\phi(\mathbf{Q}) \phi(\mathbf{K})^\top$ and then multiplying by $\mathbf{V}$, we first form the $d \times d$ matrix $\sum\_{j} \phi(\mathbf{k}\_j)\, \mathbf{v}\_j^\top$ and then apply it to each query. The cost drops from $O(N^2 d)$ to $O(N d^2)$, ie, **linear in the sequence length**. The trade-off is that we have to live with whatever the feature map $\phi$ approximates. The Performer{{ reference(key="choromanski2021performer") }} uses random Fourier features that give an unbiased estimate of the softmax kernel; Linformer{{ reference(key="wang2020linformer") }} achieves linearity differently, by projecting the keys and values down to a fixed dimension $k \ll N$ before attention; Katharopoulos et al.{{ reference(key="katharopoulos2020transformers") }} use $\phi(\mathbf{x}) = \mathrm{elu}(\mathbf{x}) + 1$ and observe that the resulting causal attention can be implemented as a recurrent state, recovering RNN-like inference cost.

Linear attention has remained niche compared to FlashAttention plus sparse patterns, because the kernel approximation tends to hurt model quality and the constant factors of the $O(N d^2)$ algorithm are unfriendly when $d$ is large.

### FlashAttention

FlashAttention{{ reference(key="dao2022flashattention") }} computes the **exact same attention** as the standard formula but reorganises the computation to minimise reads and writes to high-bandwidth memory (HBM), the slow off-chip memory of a GPU.

The standard attention kernel materialises the full $N \times N$ score matrix $\mathbf{S}$ in HBM, then materialises the attention probabilities $\mathbf{P} = \mathrm{softmax}(\mathbf{S})$ in HBM, then computes $\mathbf{P} \mathbf{V}$. This is three HBM round trips for a tensor that does not fit in the small but fast on-chip SRAM. FlashAttention uses **tiling**: it loads small blocks of $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$ into SRAM, computes the corresponding block of $\mathbf{S}$ on-chip, applies an **online softmax** (keeping running maximum and running sum statistics so the softmax can be updated incrementally as new blocks arrive), multiplies by the local block of $\mathbf{V}$, and accumulates into the output. The full $\mathbf{S}$ and $\mathbf{P}$ matrices are never written to HBM. The result is the bit-exact attention output (no approximation), with HBM traffic reduced from $O(N^2)$ to $O(N^2 / B)$ for block size $B$.

The downstream effect is a 2-4x training speedup and a similar reduction in attention memory, and FlashAttention is now the default kernel in essentially every Transformer training and inference framework. The lesson it embodies is that the bottleneck for many transformer operations is not flops but memory bandwidth, and that the right unit of optimisation is the IO complexity, not the arithmetic complexity.

{% mathblock(kind="note", name="Cost analysis (attention variants)", id="attn-cost-table") %}
Standard attention is $O(N^2 d)$ time and $O(N^2 + N d)$ memory. **MQA/GQA** reduces only the inference KV cache, not the asymptotic attention cost; the saving is in memory bandwidth and cache footprint at decode time, $h \times$ smaller for MQA and $h/g \times$ smaller for GQA. **Sliding-window attention** is $O(N w d)$ time and $O(N w + N d)$ memory for window size $w$. **Linear attention** is $O(N d^2)$ time and $O(d^2 + N d)$ memory. **FlashAttention** keeps the $O(N^2 d)$ flop count of standard attention but cuts HBM IO from $O(N^2)$ to $O(N^2 / B)$, giving a 2-4x wall-clock speedup. The right choice depends on the regime: small $N$ with large $d$ favours standard attention plus FlashAttention; very large $N$ with small $d$ and structurally local dependencies favours sliding window or sparse patterns; extremely long $N$ with no locality structure is where linear attention is least bad, despite its quality cost.
{% end %}

## Modern Design Decisions

The Transformer of 2017 has been quietly upgraded over the years. The differences between the original and a modern decoder-only LLM are not in the high-level structure (still attention plus FFN, residuals, layer norm, plus or minus an encoder) but in a handful of localised choices. Each has a clean motivation.

### Pre-LN vs Post-LN

The original paper uses **post-LN**: $\mathrm{LayerNorm}(\mathbf{x} + \mathrm{Sublayer}(\mathbf{x}))$. This places the layer norm **after** the residual addition. In deep stacks (more than a few dozen layers), post-LN is famously hard to train: the gradient through the residual path is rescaled by the layer norm Jacobian at every block, and this rescaling can shrink or amplify gradients in ways that prevent convergence without a carefully tuned learning rate warmup. Xiong et al.{{ reference(key="xiong2020layernorm") }} analysed this and showed that **pre-LN**, which applies the layer norm **inside** the sublayer ($\mathbf{x} + \mathrm{Sublayer}(\mathrm{LayerNorm}(\mathbf{x}))$), preserves the identity path through the residual cleanly and trains stably without warmup at any depth. Essentially every modern Transformer uses pre-LN.

The trade-off is that pre-LN typically reaches a slightly worse final loss than well-tuned post-LN at modest depths, because the un-normalised residual stream grows through the network and the deepest sublayers operate on inputs of large norm. Modern recipes mitigate this with a final layer norm at the end of the stack and with careful initialisation of the residual contributions.

### RMSNorm

**Root Mean Square Norm**{{ reference(key="zhang2019rmsnorm") }} drops the mean-centring step of layer norm and normalises by the root mean square of the activations alone:

{% equation(id="rmsnorm") %}
\mathrm{RMSNorm}(\mathbf{z}) = \boldsymbol{\gamma} \odot \frac{\mathbf{z}}{\sqrt{\frac{1}{d} \sum_{j=1}^d z_j^2 + \varepsilon}}.
{% end %}

The mean-subtraction in standard layer norm is a small but non-trivial fraction of the per-layer compute, and empirically the centring contributes little to training stability once the network has trained for a while. RMSNorm preserves the variance-control benefit of layer norm while saving the cost of a sum and a subtraction, and is the default in LLaMA, PaLM, and most recent open LLMs.

### Rotary Position Embeddings

Sinusoidal positional encoding adds positional information at the input and trusts the network to preserve it through the stack. **Rotary Position Embeddings** (RoPE){{ reference(key="su2021roformer") }} take a different angle: they encode position as a multiplicative rotation applied to the **queries and keys** at every layer, leaving the values untouched.

For each two-dimensional slice of $\mathbf{q}$ and $\mathbf{k}$ at frequency $\omega\_i = 10000^{-2i/d}$, RoPE rotates the slice by angle $\omega\_i \mathrm{pos}$. Because the dot product of two rotated vectors is

{% equation(id="rope-dot") %}
\big(\mathbf{R}_{\omega \mathrm{pos}_q}\mathbf{q}\big)^\top \big(\mathbf{R}_{\omega \mathrm{pos}_k}\mathbf{k}\big) = \mathbf{q}^\top \mathbf{R}_{\omega(\mathrm{pos}_k - \mathrm{pos}_q)} \mathbf{k},
{% end %}

the attention score depends only on the **relative** position of the query and key, not on their absolute positions. This is the property the sinusoidal encoding only approximated through a linear map; RoPE makes it exact and structural. The other practical benefit is **length extrapolation**: a model trained at sequence length $N$ can be evaluated at sequence length $2N$ by extending the rotations, with a graceful (though not perfect) degradation in quality. Sinusoidal positional embeddings, learned absolute embeddings, and RoPE all exist in production; RoPE is the most common modern choice.

ALiBi{{ reference(key="press2022alibi") }} pursues the same goal differently: it adds a linear bias proportional to the query-key distance directly to the attention scores, with a per-head slope, and uses no positional embedding at the input. ALiBi gives strong length extrapolation at the cost of a more rigid positional inductive bias.

### GLU Variants

The original feed-forward block is a two-layer MLP with a single non-linearity. Shazeer{{ reference(key="shazeer2020glu") }} observed that replacing the first projection with a **gated linear unit** improves quality at constant parameter count.

{% equation(id="glu") %}
\mathrm{GLU}(\mathbf{x}) = (\mathbf{W}_1 \mathbf{x}) \odot \sigma(\mathbf{W}_g \mathbf{x}),
{% end %}

with two parallel input projections, the second wrapped in a sigmoid, multiplied elementwise. **SwiGLU**, the most common variant in modern LLMs, replaces $\sigma$ with the SiLU/Swish activation $\mathrm{SiLU}(z) = z \sigma(z)$:

{% equation(id="swiglu") %}
\mathrm{SwiGLU}(\mathbf{x}) = (\mathbf{W}_1 \mathbf{x}) \odot \mathrm{SiLU}(\mathbf{W}_g \mathbf{x}).
{% end %}

To keep the parameter budget constant against a vanilla FFN of inner dimension $4 d\_{\text{model}}$, the SwiGLU FFN typically uses inner dimension $\frac{8}{3} d\_{\text{model}}$ (because it has three projections instead of two). The gating gives the model a multiplicative interaction between input dimensions inside the FFN that the additive ReLU/GELU cannot express, and the empirical gain is consistent enough that SwiGLU is now the default in LLaMA, PaLM, and most modern decoder-only LLMs.

### KV Cache

At inference, the decoder generates one token at a time. Naively, computing self-attention at step $t$ would require running the full $t \times t$ attention from scratch, an $O(t^2)$ cost per step and $O(N^3)$ for an $N$-token generation. The **KV cache** observes that the keys and values of the first $t-1$ tokens do not change when we add token $t$: they are functions of the input only, not of the new token. Storing them in a buffer that grows by one row per step reduces per-step attention to $O(t d)$ and full generation to $O(N^2 d)$, the same as a single training pass.

The cache is what made the original LLM-style inference economically viable, and its size (linear in $N$, in $L$, and in $h$) is the reason MQA and GQA are so widely adopted. Modern inference servers spend most of their effort on KV cache management: paging it across requests, sharing prefix caches between requests, and quantising it to 8-bit or 4-bit precision to fit more tokens per GB of HBM.

{% mathblock(kind="warning", name="What changes from 2017 to today", id="modern-summary") %}
A modern decoder-only LLM differs from the 2017 Transformer in roughly the following way: replace post-LN with **pre-LN**, replace LayerNorm with **RMSNorm**, replace sinusoidal positional encodings with **RoPE** (or ALiBi), replace ReLU FFNs with **SwiGLU**, replace multi-head with **grouped-query attention**, and use **FlashAttention** as the kernel. The high-level architecture is almost untouched: every component still maps to something Vaswani et al. described in 2017. The progress has been, almost entirely, in finding less wasteful versions of the same design.
{% end %}

## Training Considerations

A few details of how Transformers are trained are unobvious enough that they are part of the architecture in spirit if not in code.

### Learning-Rate Warmup

The original paper used the schedule

{% equation(id="warmup-schedule") %}
\eta_t = d_{\text{model}}^{-0.5}\, \min\!\left(t^{-0.5},\, t \cdot t_{\text{warmup}}^{-1.5}\right),
{% end %}

which linearly increases the learning rate for the first $t\_{\text{warmup}}$ steps (4000 in the original) and then decays as $1/\sqrt{t}$. The motivation is that the residual stream of an untrained Transformer is dominated by the (small) random initialisation of the sublayers, so the early gradient signal is noisy and large steps destabilise training. The warmup gives the layer norm statistics and the attention patterns time to stabilise before the optimiser starts moving aggressively. Pre-LN architectures tolerate skipping the warmup, but most modern recipes still use a short warmup followed by cosine decay (the same schedule discussed in the [neural network](/blog/neural-network/) post).

### Label Smoothing

Cross-entropy training with one-hot targets pushes the model to assign probability 1 to the true class and 0 to all others, which encourages large pre-softmax logits and overconfident predictions. **Label smoothing** with strength $\varepsilon \in [0, 0.1]$ replaces the one-hot target by a mixture: $1 - \varepsilon$ on the true class and $\varepsilon / (K - 1)$ on each of the other $K - 1$ classes. The model is no longer rewarded for arbitrarily large logits and the resulting calibration is better. The trade-off is a slight increase in training perplexity (since the loss is no longer minimised by perfect prediction) but a typically improved test performance and downstream metric. The Vaswani paper used $\varepsilon = 0.1$; modern LLM training varies.

### Adam, AdamW, and their Friends

The optimiser of choice for Transformers is **AdamW** (defined in the [neural network](/blog/neural-network/) post) with $\beta\_1 = 0.9$, $\beta\_2 = 0.95$ to $0.98$ depending on the model, and a small weight decay $\lambda \in [0.01, 0.1]$ applied to all parameters except biases and layer norm scales. The lower $\beta\_2$ relative to the standard Adam default of $0.999$ is a quirk of large-scale training: the second-moment estimate becomes biased when the gradient norm changes quickly through training, and a slightly faster decay tracks the change better.

## Practical Pitfalls

Three issues are characteristic of training Transformers and worth flagging up front.

The first is **attention collapse**, where many heads converge to attending uniformly or to a single position (often the first token) and stop contributing useful signal. This is more of a symptom than a cause and usually points to insufficient model capacity, a learning rate that decayed too aggressively, or a regularisation strength that is too high. Removing the collapsed heads at inference time often costs essentially nothing in quality, which is the basis for **head pruning**.

The second is **NaN losses from the softmax**. With float16 training, the maximum logit before the softmax can overflow to $+\infty$ and the entire row becomes NaN after the normalisation. The standard defence is to upcast the attention scores to float32 before the softmax and downcast the output, which costs a small amount of memory but is essentially mandatory in mixed-precision Transformer training. FlashAttention handles this internally with the online softmax, which is naturally numerically stable.

The third is the **KV cache running out of memory at inference**. A 70B-parameter model with a 32k context and float16 KV cache requires tens of GB of HBM per request, and a single GPU typically holds a handful of concurrent requests. Long-context inference budgets are dominated by this number, and reducing it (via GQA, KV cache quantisation, or paged attention) is currently the most active subfield of inference engineering.

{% mathblock(kind="note", name="What to use when", id="attn-defaults") %}
For a fresh Transformer from scratch, use **pre-LN with RMSNorm**, **RoPE** for positions, **SwiGLU** in the FFN, **grouped-query attention** with $g \approx h / 8$, and **FlashAttention** as the kernel; train with **AdamW** at $\beta\_2 = 0.95$, a short linear warmup followed by cosine decay, weight decay $\approx 0.05$ on non-norm parameters, and **label smoothing** $\varepsilon = 0.1$ for translation or $\varepsilon = 0$ for language modelling. Use the **encoder-decoder** flavour for tasks with a clear input-output sequence pair (translation, summarisation), the **encoder-only** flavour for representation learning (classification, retrieval), and the **decoder-only** flavour for autoregressive generation (language modelling, instruction following). Reach for **sparse or sliding-window attention** only when the natural sequence length exceeds what FlashAttention can handle on the available hardware and the dependencies are demonstrably local; reach for **linear attention** only when even sparse patterns are infeasible and a mild quality regression is acceptable.
{% end %}
