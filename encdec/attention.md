本文首先介绍最初的 attention 机制[@bahdanau2014neural; @luong2015effective]，然后简要地介绍一些其他衍生的 attention机制。attention 的更详细内容见《[attentions](/)》。

!> 补充：attention 机制有什么作用？

attention 可以从源端的大量信息中筛选出**少量**重要的信息并聚焦到这些信息上[2]，聚焦体现在各个信息的权重上。权重越大说明该信息越重要。*需实验验证“少量重要信息”是否正确，例如是否大部分信息的权重都接近于 0。*

以下以公式解释。

假设有源句 $X=\{x_1, x_2, x_3\}$，目标句为 $Y=\{y_1, y_2, y_3, y_4\}$。如果现在生成 $y_3$，就可以使用概率公式表示为 $p(y_3 | X y_1 y_2)$。

假设 $y_3$ 与 $x_1$ 无关，那么它们可以看作是独立的，也就是 $p(y_3 | X y_1 y_2) = p(y_3 |x_2 x_3 y_1 y_2)$。普通的 seq2seq 只做到了等式的左边，还需要额外学习 $y_3$ 和 $x_1$ 之间的关系，这是无意义的，而 attention 做到了等式的右边。

## 引言
[@bahdanau2014neural] 在机器翻译领域首次提出 attention 机制。RNN 领域的第一篇文章为《Recurrent Models of Visual Attention》，*事实上 attention 机制早就有人提出了*。

[@xu2015show]提出 **soft/hard attention** 的概念。

[@luong2015effective] 对 attention 进行了深度的探索，告诉人们它还有很多玩法，提出了 **global/local attention** 的概念。其中 global attention 与 soft attention 类似。但是你也可以将 soft attention 视为一个类型，包括 global/local attention。此外，[@luong2015effective] 还提出了 input-fedding approach。

我们所熟知或者刚接触到的 attention 基本上都是 global attention，它主要的做法是对 encoder 每个时间步上的输出都去加权和。而 local attention 就是只取局部的加权和。hard attention 其实就是只关注一个地方。最后[@luong2015effective]指出对比 global/local attention，更常用的还是 global attention，即我们熟知的那个。

为了区分两种 Attention 机制，后来将[@bahdanau2014neural]提出的方法和这篇论文方法分别称为 **Bahdanau Attention**（亦称 additive attention）和 **Luong Attention**（亦称 multiplicative attention）。

接下来将主要介绍：**1）**attention 的基础知识；**2）**在机器翻译领域最早提出的 attention 机制——Bahdanau Attention 以及 Luong 提出的几种 attention 变体；**3）**hard attention。

## 基础知识
本章开头已经介绍了很多关于 attention 的知识以及历史，但是有一点没讲，即以什么方式区分那么多 attention？其实 attention model 的形式多种多样，不过大都是由两个函数进行控制的：**1）**alignment function；**2）**score function。**alignment function 控制 decoder 应该关注 encoder 中的哪些部分；score function 控制应该以怎么样的方式进行关注。**因此 alignment function 区分了 **global/local/soft/hard attention**。[@luong2015effective]提出四种不同的 score function（最后一种可能不常见）：

$$
score(h_t, \bar{h}_s) = 
\begin{cases}
h^T_t \bar{h}_s & \text{dot} \\ 
h^T_t W_a \bar{h}_s & \text{general} \\ 
v^T_a tanh(W_a [h_t;\bar{h}_s]) & \text{concat} \\
W_a h_t & \text{location} \\
\end{cases}
$$

## global attention
下面几张图片分别是 Bahdanau attention、Luong attention（global attention + input-feeding） 以及 local attention 的执行步骤。（注：下面两幅图可能画有点问题，2020.08.24 留）

![bahdanau attention](https://blog-content-1256924128.cos.ap-shanghai.myqcloud.com/zcy/深度学习算法（三）：RNN%20各种机制/bahdanau-attention.gif 'bahdanau attention :size=49%')
![global attention](https://blog-content-1256924128.cos.ap-shanghai.myqcloud.com/zcy/深度学习算法（三）：RNN%20各种机制/global-attention.gif 'global attention :size=50%')

global attention 相较于 Bahdanau attention，虽然在灵感上与其相似，但是细微的不同之处还是反映了其是如何从原模型简化和泛化的。

1. global attention 简单地使用了最上一层 LSTM 的隐藏状态。而 Bahdanau attention 使用的是 bi-encoder 正反向隐藏状态和 uni-decoder 隐藏状态的拼接版；
2. global attention 的计算路线是 $h_t \to a_t \to c_t \to \tilde{h}_t$，即在时间步 $t$，先由 decoder 生成隐藏状态 $h_t$，然后使用它计算 attention $c_t$，最后再使用 $c_t$ 与 $h_t$ 拼接获得用于生成单词的 $\tilde{h}_t$。
	- 而 Bahdanau attention 的计算路线是 $h_{t-1} \to a_t \to c_t \to h_t$，它使用的是上一个时间步 $t-1$ 的隐藏状态 $h_{t-1}$，然后计算 attention $c_t$，最后将 $c_t$ 输入进 decoder 获得当前时间步 $t$ 的隐藏状态 $h_t$。

上述 Bahdanau attention 的计算路线取自[@luong2015effective]的论文。但是我认为这样的曲线无法直观的体现出 attention 的计算路线，所以我重画了它：

$$
\begin{cases}
[y_{t-1}; \tilde{h}_{t-1}], h_{t-1} \to h_t \to a_t \to c_t \to tanh([c_t; h_t]) \to \tilde{h}_t \to predict & \text{(global attention)} \\
h_{t-1} \to a_t \to c_t \to [y_{t-1}; c_t], h_{t-1} \to h_t \to predict & \text{(bahdanau attention)}\\
\end{cases}
$$

总的来说，global attention 于 $t$ 时间步，RNN **执行完毕之后**，计算注意力向量 $c_t$，并用其与隐藏状态 $h_t$ 合并得到 $\tilde{h}_t$ 从而预测结果；而 Bahdanau attention 则是在 $t$ 时间步，RNN **执行之前**，计算注意力向量 $c_t$，并将其**输入进 RNN**，得到隐藏状态 $h_t$，最后**直接**预测结果。可以通过观察上面两组公式的对比，或者上文的描述得知，**Luong attention 在计算得到当前的隐藏状态之后又做了一系列的操作才去执行预测，而 Bahdanau attention 则是在获得隐藏状态之后直接执行预测**。

此外，无论是 Global 还是下节介绍 local 方法，注意决策都是独立执行的，这是一个次优解。鉴于在标准的机器翻译中，在翻译阶段通常需要维护一个 coverage 集，以跟踪哪个源词已被翻译。类似地，在基于注意的 NMT 中，对齐决策应该结合过去的对齐信息。所以该论文提出了 input-feeding 方法，指的是将注意力向量 $\tilde{h}_t$ 与下一个时间步的输入做拼接。这种连接方式有两方面作用：**1）**使模型充分了解到以前的对齐选择；**2）**创建了一个水平和垂直跨度非常深的网络。

## local attention
文中指出，local attention 与机器视觉领域[@gregor2015draw]等提出的 selective attention mechanism 类似。

global attention 的一个缺陷是，它必须关注源句中的所有单词，这种方式计算代价昂贵并且对于解码较长序列比较不切实际，例如按段落或者文档翻译。为了解决这种低效问题，[@luong2015effective]提出了 local attention，它仅仅选择源句中的一个子集。

相比于不可微的 hard attention，local attention 是可微的[@luong2015effective]。在具体细节上，在时间步 $t$，模型首先生成一个对齐位置 $p_t$，那么上下文向量 $c_t$ 就是源句窗口 $[p_t - D, p_t + D]$ 内的隐藏状态集合的加权平均值，$D$ **根据经验选择**。与 global attention 不同，local attention 的对齐向量 $a_t$ 现在是一个固定的维度，即 $\in \mathbb{R}^{2D + 1}$。接下来将介绍两类变种。

Monotonic alignment(**local-m**)：无变化对齐。我们简单地设置 $p_t = t$，假定了源句和目标句的单词是一一对应的。那么此时的 attention 计算方式其实与 global attention 一致。这种假设在真实环境中是不切实际的，但是可能有部分特殊的任务可以使用，例如序列标注。

Predictive alignment(**local-p**)：模型预测对齐位置为 $p_t = S \cdot \text{sigmoid}(v^T_p tanh(W_p h_t))$

## hard attention

## 参考文献
1. [浅谈Attention注意力机制及其实现](https://zhuanlan.zhihu.com/p/67909876)
2. [深度学习中的注意力模型 （2017 版）](https://zhuanlan.zhihu.com/p/37601161)
2. [Attention历史](https://www.cnblogs.com/robert-dlut/p/5952032.html)。实际上九几年的时候在CV领域已经有这概念了；
3. [真正的完全图解 Seq2Seq Attention 模型](https://mp.weixin.qq.com/s/0k71fKKv2SRLv9M6BjDo4w)；
30. *[吴恩达李宏毅综合学习笔记：RNN 入门#attention](https://yan624.github.io/posts/5e27260b.html#Attention)；*
31. *[CS224n学习笔记#attention](https://yan624.github.io/posts/d9a134a.html#attention)*

<textarea id="bibtex_input" style="display:none;">
@article{bahdanau2014neural,
  title={Neural machine translation by jointly learning to align and translate},
  author={Bahdanau, Dzmitry and Cho, Kyunghyun and Bengio, Yoshua},
  journal={arXiv preprint arXiv:1409.0473},
  year={2014}
}
@article{luong2015effective,
  title={Effective approaches to attention-based neural machine translation},
  author={Luong, Minh-Thang and Pham, Hieu and Manning, Christopher D},
  journal={arXiv preprint arXiv:1508.04025},
  year={2015}
}
@inproceedings{xu2015show,
  title={Show, attend and tell: Neural image caption generation with visual attention},
  author={Xu, Kelvin and Ba, Jimmy and Kiros, Ryan and Cho, Kyunghyun and Courville, Aaron and Salakhudinov, Ruslan and Zemel, Rich and Bengio, Yoshua},
  booktitle={International conference on machine learning},
  pages={2048--2057},
  year={2015}
}
@article{gregor2015draw,
  title={Draw: A recurrent neural network for image generation},
  author={Gregor, Karol and Danihelka, Ivo and Graves, Alex and Rezende, Danilo Jimenez and Wierstra, Daan},
  journal={arXiv preprint arXiv:1502.04623},
  year={2015}
}
</textarea>