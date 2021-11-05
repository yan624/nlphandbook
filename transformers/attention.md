## Sparse Attention

## Linearized Attention
旨在优化 self-attention 的时间复杂度 $O(L^2)$。简单来说，如果将 $(QK^T)V$ 的计算顺序转变为 $Q(KV^T)$，时间复杂度就可以从 $O(L^2)$ 降为 $O(D^2)$，其中 $L \gg D$，一般 $D=64$。不过，$QK^T$ 的结果还需要经过 softmax 函数，这使得很难改变计算顺序。为此研究人员首先拆解了 softmax，然后设计了众多核函数以近似 softmax。

### Efficient Attention: Attention with Linear Complexities
比 Linear Transformer 更早提出，但是好像到 2021 年才中会议，简单来说就是定义 $\phi(x) = softmax(x)$。不过他们称，严格来说 $\text{softmax}(q) \cdot \text{softmax}(k^T) \neq \text{softmax}(qk^T)$。但是可以近似原始的 softmax attention。可以看看作者的[讲解视频](https://www.bilibili.com/video/BV1Gt4y1Y7E3)。

### Linear Transformer
我们将式(1)这种原始的 Transformer 称为 softmax attention，其中 $Q,K \in \mathbb{R}^{L \times D_k}, V \in \mathbb{R}^{L \times D_v}$，$L$ 是序列长度，$D_k, D_v$ 是特征维度。

$$Attn(Q, K, V) = A = softmax(\frac{QK^T}{\sqrt{D_k}}) V \tag{1}
$$

**softmax attention** 的时间复杂度是 $O(L^2 max(D_k, D_v)) = O(L^2 D_k) + O(L^2 D_v)$。一般来说 $L \gg D_k, D_v$，因此其时间复杂度为 $O(L^2)$。*注：参与注意力计算的向量维度一般是 64 维，而序列长度可以是几百。*

可以发现 softmax attention 的时间复杂度是序列的平方，其主要来自计算公式 $QK^T$（即 $O(L^2 D_k)$）。如果按照以下顺序计算 $(Q(K^TV))$，那么时间复杂度就会缩减到 $O(L D_k D_v) = O(L D_k D_v) + O(D_k L D_v)$。一般来说，$D_k, D_v$ 的维度是相同的，因此其时间复杂度可以被视为 $O(D^2)$。将这种计算方式称为 **linearized attention**。

综上所述，使用 linearized attention 可以将 self-attention 的时间复杂度从 $O(L^2)$ 降为 $O(D^2)$，其中 $L \gg D$。

现在的问题是，$QK^T$ 被 softmax 函数包裹着，$K^T V$是不可能被优先计算的。为了解决这一问题，我们首先将 softmax attention 拆解。对于矩阵的第 $i$ 行，我们将公式 (1) 归纳为：

$$A_i = \frac{\sum^N_{j=1} \text{sim}(Q_i, K_j) V_j}{\sum^N_{j=1} \text{sim}(Q_i, K_j)} \tag{2}
$$

对于 softmax attention 而言，$\text{sim}(q, k) = \frac{qk^T}{\sqrt{D}}$。

式(2)是通用公式，可以被进一步使用核函数 $\phi(x)$ 改写为：

$$A_i = \frac{\sum^N_{j=1} \phi(Q_i) \phi(K_j)^T V_j}{\sum^N_{j=1} \phi(Q_i) \phi(K_j)^T} \tag{3}
$$

简化为：

$$A_i = \frac{\phi(Q_i) \sum^N_{j=1} \phi(K_j) {V_j}^T}{\phi(Q_i) \sum^N_{j=1} \phi(K_j)^T} \tag{4}
$$

注意：$\phi(\cdot)$ 函数中的值都是向量。因此对于 softmax attention 而言，$\phi(\cdot) = exp(\cdot)$。

还有很多种[常见的核函数](https://www.baidu.com/s?wd=常见核函数)。[@pmlr-v119-katharopoulos20a] 认为指数核函数和多项式核函数的时间复杂度还是很高，他们使用 $\phi(x) = elu(x) + 1$。虽然它没有旨在近似点积注意力，但是实验证明其与标准的 transformer 同样出色。*注：softmax attention 保证 $\phi(\cdot)$ 的输出总是整数，他们在设计核函数时也尽可能地满足了这一约束。*

对 decoder 的 causal masking，[@pmlr-v119-katharopoulos20a] 也提供了对应的计算公式。他们认为带有 causal masking 的 transformer 实际上就是 RNN。

### Performer

### RFA
RANDOM FEATURE ATTENTION

## Query Prototyping and Memory Compression

### Linformer
@Wang2020 利用一个线性映射将键值 $K, V$ 从序列长度 $L$ 映射到一个更小的维度 $L_k$。这从原来的复杂度 $O(L^2)$ 降为 $O(LL_k)$。如果 $L_k$ 足够小，那么复杂度就降到了 $O(L)$。

*这看起来有点奇怪，别人降维是从特征维度出发，Linformer 却从序列长度维度出发。*

这篇文章给出了一大堆证明，但是主要就是提出了这么一个映射函数，并且只出现在第 4 章，几句话就讲完了。这可以看作是 CNN 中的池化{[1](https://zhuanlan.zhihu.com/p/149890569)}{[2](https://zhuanlan.zhihu.com/p/147225773)}，经过一个函数就从 12x12 降到了 9x9。CNN 减少的是像素点，Linformer 降的是序列长度。咱也不知道为什么，咱也不敢问。

最后，在使用时，让 $L_k = O(\frac{D}{\epsilon^2})$ 可以做到使用线性 attention 近似原始的 attention，只附带 $\epsilon$ 的误差（error）。另外，还有一些额外的技巧，例如共享映射函数的权重等等。

### Poolingformer

### Luna
Luna: Linear Unified Nested Attention

## Low-rank Self-Attention
一些实验[@]和理论[@Wang2020]上的报告已经证明 self-attention 矩阵 $A \in \mathbb{R}^{L \times L}$ 通常是低秩的。这意味着：1）可以使用参数明确地建模低秩属性；2）self-attention 矩阵可以被低秩近似。

### Nyströmformer

## Attention with Prior

## Improved Multi-Head Mechanism

## 参考文献
<textarea id="bibtex_input" style="display:none;">
@InProceedings{pmlr-v119-katharopoulos20a,
  author    = {Katharopoulos, Angelos and Vyas, Apoorv and Pappas, Nikolaos and Fleuret, Fran{\c{c}}ois},
  booktitle = {Proceedings of the 37th International Conference on Machine Learning},
  title     = {Transformers are {RNN}s: Fast Autoregressive Transformers with Linear Attention},
  year      = {2020},
  editor    = {III, Hal Daumé and Singh, Aarti},
  month     = {13--18 Jul},
  pages     = {5156--5165},
  publisher = {PMLR},
  series    = {Proceedings of Machine Learning Research},
  volume    = {119},
  abstract  = {Transformers achieve remarkable performance in several tasks but due to their quadratic complexity, with respect to the input’s length, they are prohibitively slow for very long sequences. To address this limitation, we express the self-attention as a linear dot-product of kernel feature maps and make use of the associativity property of matrix products to reduce the complexity from $\bigO{N^2}$ to $\bigO{N}$, where $N$ is the sequence length. We show that this formulation permits an iterative implementation that dramatically accelerates autoregressive transformers and reveals their relationship to recurrent neural networks. Our \emph{Linear Transformers} achieve similar performance to vanilla Transformers and they are up to 4000x faster on autoregressive prediction of very long sequences.},
  file      = {:pdf/model arch/20_Transformers are RNNs-Fast Autoregressive Transformers with Linear Attention.pdf:PDF},
  groups    = {Transformer},
  pdf       = {http://proceedings.mlr.press/v119/katharopoulos20a/katharopoulos20a.pdf},
  url       = {https://proceedings.mlr.press/v119/katharopoulos20a.html},
}
@Article{Wang2020,
  author        = {Sinong Wang and Belinda Z. Li and Madian Khabsa and Han Fang and Hao Ma},
  title         = {Linformer: Self-Attention with Linear Complexity},
  year          = {2020},
  month         = jun,
  abstract      = {Large transformer models have shown extraordinary success in achieving state-of-the-art results in many natural language processing applications. However, training and deploying these models can be prohibitively costly for long sequences, as the standard self-attention mechanism of the Transformer uses $O(n^2)$ time and space with respect to sequence length. In this paper, we demonstrate that the self-attention mechanism can be approximated by a low-rank matrix. We further exploit this finding to propose a new self-attention mechanism, which reduces the overall self-attention complexity from $O(n^2)$ to $O(n)$ in both time and space. The resulting linear transformer, the \textit{Linformer}, performs on par with standard Transformer models, while being much more memory- and time-efficient.},
  archiveprefix = {arXiv},
  eprint        = {2006.04768},
  groups        = {Transformer},
  keywords      = {cs.LG, stat.ML},
  primaryclass  = {cs.LG},
  url           = {http://arxiv.org/pdf/2006.04768},
}

</textarea>

