## 引言
最近在 TRADE[@wu-etal-2019-transferable] 中看到了一些关于 Pointer Networks（PtrNets）的分类，正好实验里需要用到它，并且它在 NLP 中也是一项常用的技术，因此特地花了点时间研究一下。虽然不知道这种分类方法是不是业界通用的方法，但是本文还是按照该分类法描述各种 PtrNet。接下来本文将 [@vinyals2015pointer] 提出的 Pointer Networks 概念称为 PtrNets，将论文提出的模型称为 PtrNet。

1. index-based copy
2. hard-gated copy
3. soft-gated copy

## Index-based
sequence-to-sequence 算法无法处理词表会随着输入序列的改变而改变的情况，无法预测表外词的情况就是其中之一。[@vinyals2015pointer] 举了一个求解凸包的例子。简单来说，给定一堆点，要求找出其中几个点并连接成圈，使得所有其余点均在该圈中。我们将所有给定的点填入词表并分配 id，那么模型就要使用 sequence-to-sequence 算法输出那几个可以连成一圈的点。显然，如果固定词表的大小，假设为 10，那么算法就无法处理 11 个点的情况。因为第 11 个点不在词表中，是一个表外词，sequence-to-sequence 自然没办法输出。

[@vinyals2015pointer] 提出使用 Pointer Networks 解决以上问题，其原理与 attention 机制非常类似。给定一个训练对 $(\mathcal{P}, \mathcal{C}^{\mathcal{P}})$，则 PtrNets 的数学公式如下所示，其中 $v$，$W_1$ 和 $W_2$ 是可训练权重，$e_j$ 是编码器中第 $j$ 个时间步的隐藏状态，$d_i$ 是解码器中第 $i$ 个时间步的隐藏状态。如果模型不是 seq2seq 结构，那么 $d_i$ 也有可能是其他任意来源的隐藏状态。观察该式，与 attention 机制对比，可以发现其只是少了最后一步上下文向量缩放并相加的步骤。它的评分函数是 [@luong2015effective] 中提出的 `concat` 方法，实际上评分函数可以自行选择。
$$
\begin{aligned}
    u^i_j & = v^T tanh(W_1 e_j + W_2 d_i) \quad j \in (1, \cdots, n) \\
    p(\mathcal{C}_i | \mathcal{C}_1, \cdots, \mathcal{C}_{i-1}, \mathcal{P}) & = softmax(u^i)
\end{aligned}
$$

该论文的主要贡献是提出了 Pointer Networks 的概念，论文本身并未对 NLP 领域进行相关的实验，例如文本摘要任务或者命名体识别任务。

## Hard-gated Copy
### PS
[@Gulcehre2016] 进一步完善 PtrNet，提出了 Pointer Softmax (PS)，并在文本摘要任务和翻译任务上进行了实践。PS 能够做到 (i) 预测在每个时间步是否需要用 PtrNet；(ii) 指出上下文序列中的任意位置。其中 PtrNet[@vinyals2015pointer] 无法做到 (i)，[@luong2015effective] 无法做到 (ii)。

简单来说，PS 通过一个 sigmoid 函数，从输入序列或者词表中选择性地生成单词，该函数的输出被人工地设置为 0 或 1。这意味着序列的位置分布和词表的概率分布独立且不会相互影响。因此被称为 **hard-gated** copy。

具体来说，PS 的目标是给定上下文序列 $x = (x_1, x_2, \cdots, x_{T_x})$，最大化可观测目标单词序列 $y = (y_1, y_2, \cdots, y_{T_y})$ 和单词生成源 $z = (z_1, z_2, \cdots, z_{T_y})$ 的概率，其中 $z \in \{0, 1\}$ 应该指的是选择哪种算法生成单词。公式为：

$$p_{\theta}(y,z | x) = \prod^{T_y}_{t=1} p_{\theta}(y_t, z_t | y_{<t}, z_{<t}, x)
$$

注意 $y_t$ 既可以是来自 shortlist 的单词 $w_t$，也可以是来自 location softmax 的位置 $l_t$，取决于开关变量（switching variable）$z_t$。那么上式就可以被拆解为：

$$
\begin{aligned}
	p(y,z | x) & = \prod_{t \in \mathcal{T}_w} p(w_t, z_t | (y, z)_{<t}, x) \times \prod_{t' \in \mathcal{T}_l} p(l_{t'}, z_{t'} | (y, z)_{<t'}, x) \\
	p(w_t, z_t | (y, z)_{<t}) & = p(w_t | z_t = 1, (y, z)_{<t}) \times p(z_t = 1 | (y, z)_{<t}) \\
	p(l_t, z_t | (y, z)_{<t}) & = p(l_t | z_t = 0, (y, z)_{<t}) \times p(z_t = 0 | (y, z)_{<t}) \\
	p(z_t = 1 | (y, z)_{<t}) & = \sigma(f(x, h_{t-1}; \theta)) \\
	p(z_t = 0 | (y, z)_{<t}) & = 1 - \sigma(f(x, h_{t-1}; \theta))
\end{aligned}
$$

**上述公式看起来吓人，其实是纸老虎！**我们一点一点分析。

首先，第 1 个公式为什么可以这么拆？答：易得 $p(y,z|x) = p(y_t, z_t, (y,z)_{<t-1}|x)$，联合概率中字母顺序无所谓。已知 $y_t \in \{w_t, l_t\}$，进一步等于 

$$p_{t \in \mathcal{T}_w}(w_t, z_t, (y,z)_{<t-1}|x) \times p_{t' \in \mathcal{T}_l}(l_{t'}, z_{t'}, (y,z)_{<t'-1}|x)
$$ 

这样拆是用到了定理：当 A 和 B **独立**时，$p(AB) = p(A) \times p(B)$。而 $w_t$ 和 $l_t$ 应该是独立的。再用定理：$p(AB|C) = p(A|BC) \times p(B|C)$，那么上式最终就变成第 1 个式子了。

证明：

$$
\begin{aligned}
	\because \quad & p(AB|C) = \frac{p(ABC)}{p(C)} \\
	         \quad & p(ABC) = p(A|BC) p(BC) \\
	\therefore \quad & p(AB|C) = \frac{p(A|BC) p(BC)}{p(C)} = p(A|BC) \frac{p(BC)}{p(C)} = p(A|BC) p(B|C)
\end{aligned}
$$

其次，第 2、3 个公式是怎么改写的？写不动了……应该用的传统概率。

需要注意的是， $z_t \in [0, 1]$ 是 sigmoid 函数的输出，无法保证结果正好是 0 或 1。解决办法估计是以 0.5 为阈值，基于规则手动调整。这篇论文的思路感觉像**门控机制**（gated mechanism），只不过是硬性门。下面介绍的 Pointer-Generator 虽然也用的是门控机制，但是多了一点花样。它将两个概率分布合并了，而不是两个中挑一个。

### Mem2Seq
Mem2seq: Effectively incorporating knowledge bases into end-to-end task-oriented dialog systems

### GLMP
Global-to-local memory pointer networks for task-oriented dialogue

## Soft-gated Copy

### CopyNet
几乎与 PS 同时，[@Gu2016] 提出了 **CopyNet**。这篇论文没看过，不作评价。

### Pointer-Gnerator
[@See2017] 在文本摘要领域提出了 ***Pointer-Generator*** + Coverage 技术，以此缓解无法生成表外词以及避免重复某些语句。Pointer-Generator 的计算公式为：

$$p_{final} = p_{gen} \times p_{vocab} + (1 - p_{gen}) \times p_{history}
$$

其中 $p_{gen} = \sigma(W^T_{h^*} h^*_t + W^T_s s_t + W^T_x x_t + b_{ptr})$，分别由上下文向量 $h^*_t$，解码器状态 $s_t$ 和解码器输入 $x_t$ 计算得到。那么 $p_{gen}$ 就可以被视为一个软开关（soft switch），要么从词表中生成单词，要么从输入序列中拷贝单词。

值得注意的是，上式是目前比较常见的写法，但是原论文中的写法并不是如此。计算公式为：

$$P(w) = p_{gen} P_{vocab}(w) + (1- p_{gen}) \sum_{i: w_i = w} a^t_i 
$$

其中，$a^t_i$ 表示在解码端第 $t$ 个时间步上，编码端第 $i$ 个单词的概率。如果 $w$ 是表外词，那么 $P_{vocab} = 0$；与之类似，如果 $w$ 没有出现在编码端的语句中，那么 $\sum_{i: w_i = w} a^t_i = 0$。

个人认为还是原论文写得合理些，网络上常见的写法总感觉很容易让人误解。不过，原论文的公式可能更难理解一些。总而言之，实际上上述公式的思想是将两个概率分布**合并到一起**，更形象的说法是将两个概率分布摞成一摞。具体来说，对于表内词而言，它们被摞在其对应单词的词表分布上。对于表外词而言，则直接与词表分布拼接。

### d
The natural language decathlon: Multitask learning as question answering

## 参考文献
1. [Pointer Networks简介及其应用](https://zhuanlan.zhihu.com/p/48959800)
2. [NLP 硬核入门 - PointerNet 和 CopyNet](https://zhuanlan.zhihu.com/p/73590690)
3. [论文笔记](https://yan624.github.io/posts/d7a5fd2b.html)
2. 《[李宏毅深度学习学习笔记](https://yan624.github.io/posts/5e27260b.html#Pointer-Network)》记录了一个例子。

<textarea id="bibtex_input" style="display:none;">
@inproceedings{vinyals2015pointer,
  author = {Vinyals, Oriol and Fortunato, Meire and Jaitly, Navdeep},
  booktitle = {Advances in Neural Information Processing Systems},
  editor = {C. Cortes and N. Lawrence and D. Lee and M. Sugiyama and R. Garnett},
  pages = {},
  publisher = {Curran Associates, Inc.},
  title = {Pointer Networks},
  url = {https://proceedings.neurips.cc/paper/2015/file/29921001f2f04bd3baee84a12e98098f-Paper.pdf},
  volume = {28},
  year = {2015}
}
@InProceedings{Gulcehre2016,
  author    = {Caglar Gulcehre and Sungjin Ahn and Ramesh Nallapati and Bowen Zhou and Yoshua Bengio},
  booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  title     = {Pointing the Unknown Words},
  year      = {2016},
  publisher = {Association for Computational Linguistics},
  doi       = {10.18653/v1/p16-1014},
  groups    = {enc2dec},
}
@InProceedings{Gu2016,
  author    = {Jiatao Gu and Zhengdong Lu and Hang Li and Victor O.K. Li},
  booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  title     = {Incorporating Copying Mechanism in Sequence-to-Sequence Learning},
  year      = {2016},
  publisher = {Association for Computational Linguistics},
  doi       = {10.18653/v1/p16-1154},
  groups    = {enc2dec},
}
@InProceedings{See2017,
  author    = {Abigail See and Peter J. Liu and Christopher D. Manning},
  booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  title     = {Get To The Point: Summarization with Pointer-Generator Networks},
  year      = {2017},
  publisher = {Association for Computational Linguistics},
  doi       = {10.18653/v1/p17-1099},
  groups    = {enc2dec},
}

@inproceedings{wu-etal-2019-transferable,
    title = "Transferable Multi-Domain State Generator for Task-Oriented Dialogue Systems",
    author = "Wu, Chien-Sheng  and
      Madotto, Andrea  and
      Hosseini-Asl, Ehsan  and
      Xiong, Caiming  and
      Socher, Richard  and
      Fung, Pascale",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    doi = "10.18653/v1/P19-1078",
    pages = "808--819",
    abstract = "Over-dependence on domain ontology and lack of sharing knowledge across domains are two practical and yet less studied problems of dialogue state tracking. Existing approaches generally fall short when tracking unknown slot values during inference and often have difficulties in adapting to new domains. In this paper, we propose a Transferable Dialogue State Generator (TRADE) that generates dialogue states from utterances using copy mechanism, facilitating transfer when predicting (domain, slot, value) triplets not encountered during training. Our model is composed of an utterance encoder, a slot gate, and a state generator, which are shared across domains. Empirical results demonstrate that TRADE achieves state-of-the-art 48.62{\%} joint goal accuracy for the five domains of MultiWOZ, a human-human dialogue dataset. In addition, we show the transferring ability by simulating zero-shot and few-shot dialogue state tracking for unseen domains. TRADE achieves 60.58{\%} joint goal accuracy in one of the zero-shot domains, and is able to adapt to few-shot cases without forgetting already trained domains.",
}
@article{luong2015effective,
  title={Effective approaches to attention-based neural machine translation},
  author={Luong, Minh-Thang and Pham, Hieu and Manning, Christopher D},
  journal={arXiv preprint arXiv:1508.04025},
  year={2015}
}
</textarea>


