位置编码大致分为绝对位置编码和相对位置编码。


## 绝对位置编码
绝对位置编码直接将位置信息融入进输入中。原始 Transformer 采用的绝对位置编码，对于位置 $t$，由一个向量 $p_t = PE(t)$ 表示：

$$
PE(t)_i = 
\begin{cases}
	sin(\omega_i t) & \text{if i is even}, \\
	cos(\omega_i t) & \text{if i is odd} \\
\end{cases}
$$

其中 $\omega_i = \frac{1}{10000^{2i/d_{model}}}$，$d_{model}$ 是模型的隐藏状态维度。

另一种方法是训练一组位置编码，每个位置都有一个唯一的表征。但是这种方法可表示的最大长度在训练之前就已经决定好了，后期无法调整。



后来，有些工作[@]发现将位置编码加在每个 Transformer 层的输入中是有益的。


## 相对位置编码
相对位置编码指的是使用一组超参数表示一对输入元素之间的关系（方向和距离），并将其融入 self-attention 机制，称为 Relation-aware Self-Attention [@shaw-etal-2018-self]。不同注意力机制头的超参数是共享的。

输入元素 $x_i$ 和 $x_j$ 之间的一条边由两个向量 $a^V_{ij}, a^K_{ij}$ 表示。那么将 self-attention 机制改写为：

$$z_i = \sum^n_j=1 \alpha_{ij} (V_j + a^V_{ij})
$$

其中 $\alpha_{ij}$ 代表注意力矩阵中的元素，$V_j = x_j W^V$，$W^V$ 是权重矩阵。该机制也可以类似地与 $K$ 结合。

$$e_{ij} = \frac{Q_i(K_j + a^K_{ij})^T}{\sqrt{D}}
$$

其中 $e_{ij}$ 是注意力分数，用于计算注意力矩阵，即使用 softmax。

对于线性序列，边可以捕获不同输入元素之间的相对位置关系。[@shaw-etal-2018-self] 将最大相对位置裁剪为 $k$ 的绝对值。他们假设精确的绝对位置在一定位置以外是没用的。因此，边的超参数可以表示为：

$$
\begin{aligned}
a^K_{ij} & = w^K_{\text{clip}(j-i, k)} \\
a^V_{ij} & = w^V_{\text{clip}(j-i, k)} \\
\text{clip}(x, k) & = max(-k, min(k, x))
\end{aligned}
$$

通过以上公式就可以训练相对位置表征 $w^K = (w^K_{-k}, \dots, w^K_k), w^V = (w^V_{-k}, \dots, w^V_k)$，其维度与 $KV$ 相同。经原文实验分析，当 $k \ge 2$ 时 BLEU 已无明显变化。

@2020t5 (T5) 简化了上述做法，每个位置对应一个标量而非向量，将这些标量加在 attention softmax 之前的 logits 上，每个 head 有自己的表征，但是在层间共享。

## 8


## 参考文献
<textarea id="bibtex_input" style="display:none;">
@InProceedings{shaw-etal-2018-self,
  author    = {Shaw, Peter and Uszkoreit, Jakob and Vaswani, Ashish},
  booktitle = {Proceedings of the 2018 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers)},
  title     = {Self-Attention with Relative Position Representations},
  year      = {2018},
  address   = {New Orleans, Louisiana},
  month     = jun,
  pages     = {464--468},
  publisher = {Association for Computational Linguistics},
  abstract  = {Relying entirely on an attention mechanism, the Transformer introduced by Vaswani et al. (2017) achieves state-of-the-art results for machine translation. In contrast to recurrent and convolutional neural networks, it does not explicitly model relative or absolute position information in its structure. Instead, it requires adding representations of absolute positions to its inputs. In this work we present an alternative approach, extending the self-attention mechanism to efficiently consider representations of the relative positions, or distances between sequence elements. On the WMT 2014 English-to-German and English-to-French translation tasks, this approach yields improvements of 1.3 BLEU and 0.3 BLEU over absolute position representations, respectively. Notably, we observe that combining relative and absolute position representations yields no further improvement in translation quality. We describe an efficient implementation of our method and cast it as an instance of relation-aware self-attention mechanisms that can generalize to arbitrary graph-labeled inputs.},
  doi       = {10.18653/v1/N18-2074},
  groups    = {Transformer},
  url       = {https://aclanthology.org/N18-2074},
}
@Article{2020t5,
  author  = {Colin Raffel and Noam Shazeer and Adam Roberts and Katherine Lee and Sharan Narang and Michael Matena and Yanqi Zhou and Wei Li and Peter J. Liu},
  journal = {Journal of Machine Learning Research},
  title   = {Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer},
  year    = {2020},
  number  = {140},
  pages   = {1-67},
  volume  = {21},
  groups  = {Transformer},
  url     = {http://jmlr.org/papers/v21/20-074.html},
}
</textarea>