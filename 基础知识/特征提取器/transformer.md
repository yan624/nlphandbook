写下此文章时，距 Transformer 发表已过去 3 年，网络上早已经充斥着各种讲解，所以本文不打算重复这些工作。对 Transformer 结构的讲解可参考[《可视化理解Transformer结构》](https://zhuanlan.zhihu.com/p/59629215)，或者英文版 [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)。以下只简单地介绍 Transformer 各项结构的计算公式。

将多头注意力机制（Multi-Head Attention）记作 MHA，将前馈神经网络（Forward-Feed Network）记作 FFN。

## Transformer通用结构

将 Transformer 模型定义为函数 $y = F(x)$，其中 $y$ 是 Transformer 顶层的输出，$x$ 是底层的输入。$F(x)$ 将重复调用一个模块（称为 Transformer layer） N 次，该模块又由 B 个（2或3个）Transformer Block 组成。Transformer Block 可以表示为：

$$
a^l_i = f(x) = \text{LayerNorm}^l_i(a^{l-1}_i + \text{sublayer}^l_i(a^{l-1}_i))
$$

其中 $l$ 代表层数，$i$ 代表第 $l$ 层的第 $i$ 个子层。

无论是 Transformer encoder 还是 Transformer decoder，其计算流程为：

```
for l in range(N):
	for i in range(B):
		f(x)
```

从 Transformer 结构来说，$f(x)$ 的计算流程是：输入序列 $x$ 先进入子层（MHA 或 FFN），然后将输出与 $x$ 相加，即残差网络，最后将结果输入层标准化（layer normalization）中。以此类推完成一层的计算。

有以下几点需要说明：1）输入词向量；2）MHA 和 FNN 的计算公式分别是什么。

**输入词向量**是

$$X = E_{token} + E_{pos} + E_{seg}
$$

?> 由于 Transformer 不包含循环和卷积，因此为了使模型利用序列顺序，必须注入符号相对或绝对的位置信息。为此，encoder 和 decoder 底层的输入嵌入被加上了“位置编码”。有许多学好的或者固定的位置编码可供选择[[@gehring2017convolutional]](#gehring2017convolutional)。论文对比两种类别的位置编码之后，发现它们的结果非常接近。选择固定编码是因为它可以推理序列长度，以此表示比训练集中已知最长序列更长的序列。在 [get_timing_signal_1d()](https://github.com/tensorflow/tensor2tensor/blob/23bd23b9830059fbc349381b70d9429b5c40a139/tensor2tensor/layers/common_attention.py#L387) 函数中可以看到用于生成位置编码的代码。

**MHA** 的计算公式如下所示，其中 $[Q, K, V] = \text{Linear}(X) \in \mathbb{R}^{<d(X)}$。
$$
\begin{aligned}
	\text{MHA}(X) & = \text{Linear}([Attn_1; \cdots; Attn_8]) \in \mathbb{R}^{d(X)} \\
	\text{Attn}(Q, K, V) & = \text{softmax}(\frac{Q K^T}{\sqrt{d_k}}) V \in \mathbb{R}^{<d(X)} \\
\end{aligned}
$$

**FFN** 的计算公式如下所示，其中激活函数不一定非要是 $ReLU(x) = max(0, x)$，有些论文还会用 $\text{GELU}$。

$$\text{FNN}(x) = max(0, x \cdot W_1 + b_1) \cdot W_2 + b_2
$$

值得注意的是，multi-head attention 看英文名感觉“高大上”。其实很简单，就是将 self-attention 执行多次。Transformer 执行了 8 次，因此产生 8 个向量。然而，在训练时只需要一个向量。为此，Transformer 拼接 8 个向量，再乘上一个权重矩阵，使其维度还原。

## Encoder-Decoder Attention
对于 Transformer decoder 来说，由于它需要与编码器交互，因此它还多了一个 Encoder-Decoder Attention 子层，具体在 MHA 和 FFN 之间。

具体来说，Encoder Decoder Attention 的 QKV 获取方式与 self-attention 略有不同。Q 来自输入序列的线性变换，但是 KV 来自 encoder 顶层输出的线性变换。这有助于解码器将注意力集中在输入序列的合适位置。**博主注**：以上 QKV 的用法来自对 pytorch 官方源码的分析。

## encoder和decoder中self-attention的区别
在 NLP 中，输入到模型的**一批**序列大概率不等长。在训练时需要 padding mask，让模型不要提取填充单词的信息。不过，Transformer decoder 为了不让模型在解码时关注“未来”的单词，还需要 sequence mask。因此 encoder 和 decoder 的 self-attention 的区别是：decoder 的 self-attention 是 **masked multi-head attention**，而 encoder 的仅仅是 **multi-head attention**。详见下面 **mask** 一节。

## 🤿mask
Transformer decoder 的 mask 指的是 **sequence/padding mask**，encoder 的 mask 指的是 **padding mask**。可参考文献 3、4。

**padding mask** 是为了防止填充值 PAD 具有意义。在反向传播时，如果不做 padding mask，框架会对 PAD 求导。但是通常 PAD 被认为是一个无意义的填充符，所以最好不要计算其梯度。**对于 Transformer，无论 encoder 还是 decoder 都要做 mask**。**sequence mask** 是为了遮住来自未来的信息，使得 decoder 在解码时，不会依赖未来的信息。

### padding mask
在 encoder 中,每次执行 scaled dot-product 后都要做一次 padding mask。由于我们要让序列的长度相等以便做向量化操作，所以必不可少地需要对输入序列进行**截断**或**补零**（即填充 PAD，PAD 不一定非要是 0）操作。所以 padding mask 的**主要目的**是使得 self-attention 不要关注向量中的 PAD 符号，使神经网络忽略 PAD 的信息。

在做 attention 之前，先把 PAD 所在位置的值置为一个极小值，甚至是无穷小（由于 attention 需要经过 softmax，softmax 需要求 e 的次方，要想 e 的某次方为 0，只能使得值为无穷小才可以，这是一个数学问题）。

mask 的**具体操作**是：在执行 scaled dot-product **后**，将序列中补零位置所对应的隐藏状态置为 -INF，使得序列经过 softmax 层时，**该对应位置所计算出的概率为 0**。（*mask 操作在 Transformer 中貌似是可选的。*）

?> 除了 Transformer，其他算法可能也需要 padding mask，下面列举 Embedding、LSTM 和 loss 计算三种情况。

1. **对于 Embedding 层**来说，符号 PAD 的词向量是无意义的，可以使用上述类似的方法，乘一个 mask。不过这样略微麻烦，所幸的是 Pytorch 提供了简单的实现，只需要填入 `padding_idx` 即可实现上述功能。但是这只是让 PAD 这个词向量无意义并且不计算它的梯度而已。在经过复杂的计算之后 PAD 所在位置的值依旧会变为非 0。对于嵌入层之后的隐藏层，pytorch 没有提供这么简单的方式，需要乘上一个 mask 矩阵。
```
nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
```
2. **对于 LSTM 等的时序特征提取层**可以使用 `torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True)` 函数将 embedding 打包。然后将这个打包后的 embedding 输入 LSTM，之后使用 `torch.nn.utils.rnn.pad_packed_sequence(sequence, batch_first=False, padding_value=0.0, total_length=None)` 函数，将 LSTM 的输出还原。
以以下代码为例，其中第一行代码的 `lengths` 必须是 pytorch 的 tensor，它代表着你输入序列的长度，比如你输入 [['你', '好', '。'], ['我', '是', '鱼', '。']]，那么 `lengths` 就是 `torch.Tensor([3, 4])`。`batch_first` 默认是 False，这是因为 LSTM 的输入要求是 (S, N, *)，其中 S 是序列长度，N 是批次大小。`batch_first` 顾名思义就是输入的序列的第一个维度是否为批次大小，显然为了使得 pytorch 内部兼容，`batch_first` 默认为 `False` 是最好的。一般我都懒得对序列进行长度上的排序，所以将 `enforce_sorted` 设置为 False。
```
# lstm padding mask
packed_feature = nn.utils.rnn.pack_padded_sequence(embed, lengths, enforce_sorted=False)
packed_bi_feature, hx = self.lstm(packed_feature)
bi_feature, lengths_unpacked = nn.utils.rnn.pad_packed_sequence(packed_bi_feature)
```
3. 最后**在计算 loss 时**也需要 padding mask，这可以使用 `torch.masked_select(input, mask, out=None)` 进行计算。

### sequence mask
通常，decoder 被禁止看见未来的信息。使用 sequence mask 可以使其只关注当前时间步之前的单词，而不使用后面未解码出单词的信息。

sequence mask 的示例如下所示。其实 sequence mask 就是一个 $L \times L$ 维的矩阵，L 代表一条语句的长度。其内容是，下三角以及对角线全部为 1，代表需要这部分信息。上三角全部为 0，代表这部分信息被掩盖。

```
我**
我是*
我是鱼
```

## 📚参考文献
1. [放弃幻想，全面拥抱Transformer：自然语言处理三大特征抽取器（CNN/RNN/TF）比较](https://zhuanlan.zhihu.com/p/54743941)
2. [《BERT大火却不懂Transformer？读这一篇就够了》](https://zhuanlan.zhihu.com/p/54356280)
1. [Transformer模型的PyTorch实现](https://luozhouyang.github.io/transformer/)
2. [The Transformer](https://www.jianshu.com/p/405bc8d041e0)
3. [深度学习中的 mask 到底是什么意思？](https://www.zhihu.com/question/320615749/answer/1080485410)
4. [变长序列怎么使用 mini-batch SGD 训练？](https://www.zhihu.com/question/264501322/answer/433784349)
5. [transformer 在解码的时候，用的 k 和 v 的向量来自于编码器的输出还是来自于之前解码输出的值呢？](https://www.zhihu.com/question/347366108/answer/832932755)
6. [『计算机视觉』各种Normalization层辨析](https://www.cnblogs.com/hellcat/p/9735041.html#_label3_0) 
7. [layer normalization 简单总结](https://www.jianshu.com/p/c357c5717a60)

<textarea id="bibtex_input" style="display:none;">
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  pages={5998--6008},
  year={2017}
}
@article{gehring2017convolutional,
  title={Convolutional sequence to sequence learning},
  author={Gehring, Jonas and Auli, Michael and Grangier, David and Yarats, Denis and Dauphin, Yann N},
  journal={arXiv preprint arXiv:1705.03122},
  year={2017}
}
</textarea>
