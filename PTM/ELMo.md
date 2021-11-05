与以往的预训练词向量（word2vec、GloVe、fasttext）不同，**ELMo[[@peters2018deep]](#peters2018deep) 的词向量表征是一个基于整条输入语句的函数**。这个函数就是 biLM。下面将分别介绍（1）什么是 biLM，（2）ELMo 词向量是如何产生的，（3）如何在下游任务使用，（4）biLM 的具体结构。最后总结 ELMo 的缺陷。

## biLM
biLM 实际上就是一个双向的语言模型，使用 LSTM 构建。给定一个包含 N 个符号的序列 $(t_1, t_2, \cdots, t_N)$，**前向语言模型**计算给定 $(t_1, \cdots, t_{k-1})$ 时 $t_k$ 的条件概率：

$$p(t_1, t_2, \cdots, t_N) = \prod^N_{k=1} p(t_k | t_1, t_2, \cdots, t_{k-1})
$$

在深度学习领域中，计算该概率的普遍做法是将**上下文**<font color='red'>**无关**</font>的词向量 $x^{LM}_k$（可以是任意种类的词向量）输入 $L$ 层的前向 LSTM。那么在位置 $k$，**每层** LSTM 都会输出一个**上下文**<font color='red'>**相关**</font>的表征 $\overrightarrow{h}^{LM}_{k, j}$，其中 $j = 1, \cdots, L$。最后，$\overrightarrow{h}^{LM}_{k, j}$ 通过一个 softmax layer 去预测下一个符号 $t_{k+1}$。

之前说了 ELMo 使用的是一个双向语言模型，所以自然还有一个**后向语言模型**。它与前向语言模型类似，基于下文预测当前位置的单词：

$$p(t_1, t_2, \cdots, t_N ) = \prod^N_{k=1} (t_k | t_{k+1}, t_{k+2}, \cdots, t_N)
$$

可以发现这个双向语言模型其实类似于 word2vec 的 CBOW 算法。**这个双向语言模型的优化目标就是二者的对数似然损失之和**：

$$\sum^N_{k=1}(\log p(t_k | t_1, t_2, \cdots, t_{k-1}; \theta_x, \overrightarrow{\theta}_{LSTM}, \overrightarrow{\theta_s}) + \log p(t_k | t_{k+1}, t_{k+2}, \cdots, t_N; \theta_x, \overleftarrow{\theta}_{LSTM}, \overleftarrow{\theta_s}))
$$

其中 $\theta_x$ 是词向量，$\theta_{LSTM}$ 是 LSTM 的参数，$\theta_s$ 是 softmax layer 的参数。**注意 ELMo 中，正反向的参数是独立的。**下一节将介绍一个与先前工作不同的新方法：**将 biLM 学到的表征线性组合起来。**

!> 需要解释两个问题：1）ELMo 为什么要使用两个单向 LSTM 实现双向语言模型，而不直接用 BiLSTM？2）ELMo 为什么要实现双向语言模型？  
先回答第 2 个问题：因为 ELMo 为了训练词向量，需要比语言模型看到的更多，它不光要知道上文，还需要下文。   
再回答第 1 个问题：因为 BiLSTM 根本实现不了双向语言模型。  
“BiLM”和“使用 BiLSTM 实现的 LM”不同是因为：使用反向 LSTM 实现的“这个东西”根本不是语言模型，它只是把正向语言模型反向输入到了 LSTM 中。反向语言模型指的是，给定序列 $w_1, \dots, w_n$，有条件概率 $p(w_i|w_{i+1}, \dots, w_n)$。正向语言模型很简单，就是 $p(w_i | w_1, \dots, w_{i-1})$。。  
最后综合两个问题，再问：既然需要上下文并且前人已经研发出 BiLSTM，那不要实现双向语言模型，直接用 BiLSTM 不就好了？  
虽然 BiLSTM 也可以看到上下文，因此足以训练词向量（函数）且也不需要使用双向语言模型，但是 BiLSTM 会泄露未来的信息。泄露信息很好理解，可以参考[文章](https://zhuanlan.zhihu.com/p/72839501)。  

## ELMo词向量
对于每一个符号 $t_k$，$L$ 层的 biLM 可以计算得到 $2L + 1$ 个表征（ELMo 只用了两层，所以 $2L + 1 = 5$），$1$ 代表词向量层。当正反向表征合并之后可以得到三个表征，即 $x^{LM}, h^{LM}_{k, 1}, h^{LM}_{k, 2}$，以上描述可由如下公式表示，其中 $h^{LM}_0$ 代表词嵌入 $x^{LM}$，$h^{LM}_{k, j} = [\overrightarrow{h}^{LM}_{k, j}; \overleftarrow{h}^{LM}_{k, j}]$：
$$
\begin{aligned}
R_k = & \{x^{LM}_k, \overrightarrow{h}^{LM}_{k, j}, \overleftarrow{h}^{LM}_{k, j} | j = 1, \cdots, L\} \\
= & \{h^{LM}_{k, j} | j = 0, \cdots, L\}
\end{aligned}
$$
在下游任务，ELMo 将多层表征融入一个向量中，$ELMo_k = E(R_k; \theta_e)$。最简单的做法是只取最上层的表征，即 $ELMo_k = h^{LM}_{k, j}$，这也是比较普遍的做法。更通用的做法是对所有层计算一个特定于任务的加权和：

$$ELMo^{task} = \gamma^{task} \sum^L_{j = 0} s^{task}_j h^{LM}_{k, j}
$$

其中 $\gamma^{task}$ 是一个特定于任务的缩放因子，$s^{task}_j$ softmax 标准化权重（注意，这是一个可学习的参数）。其中 $\gamma$ 对优化过程起到非常重要的作用，详见论文中的附加材料。

### 如何在下游任务
一般来说，对于监督 NLP 模型的通用做法是：将上下文无关的词向量（可以是预训练的、字符级的、来自 CNN 的等等类型）输入进一个模型得到一个上下文敏感的表征，通常是 bi-RNN，CNN或者是线性层。

关于下游任务具体使用哪种模型还是有很多选择的，论文中提到可以使用 RNN、CNN或者线性层。但是由于预训练模型本身就比较大，一些内存小点的服务器可能无法再支撑后面接一个 RNN 之类的大型神经网络了。所以实战的时候可以自行调节。

那么如何将 ELMo 加入进这个监督模型呢？**首先**冻结 biLM 的权重，**然后**拼接词向量 $x_k$ 和 ELMo 的产生的 $ELMo^{task}_k$，**最后**将 ELMO 增强的表征 $[x_k; ELMo^{task}_k]$ 输入进任务相关的 RNN。对于一些任务，不同的组合可能还有提升。

**简单来说，就是将 ELMo 的权重冻结，然后使用 ELMo 产生的表征去替换任务中原先使用的词向量（例如 ELMo 替换 Glove）。**

## biLM的具体结构
biLM 的结构与 A、B（由于一个人的名字不是英文，无法复制，我就简称 A、B 了，具体是谁可以去原文看） 的工作一样（大致应该是引入了 char-level 的信息），但是改成了支持双向的联合训练，以及在 LSTM 各层中加入残差连接。

biLM 使用的是 CNN-BIG-LSTM，考虑到模型的复杂度对下游任务的影响，作者对原模型的参数减了半。最后模型使用 2 层双向的 LSTM（4096 个隐藏单元，512 维）以及一个在一二层之间的残差连接层。输入是 2048 个的 n-gram 级卷积 filter，512 维。

## 改进以及缺陷
改进：  
1. 利用 biLSTM 获得了基于上下文的词向量，缓解了一词多义的问题。

缺陷：  
1. 使用的是自回归语言模型（Autoregressive LM），缺点是只能看上文或者下文。尽管 ELMo 使用了两个正反向的自回归来近似捕获上下文，但是这终究只是近似。
2. ELMo 官方推荐把它当做冻结的词向量用，那么就无法微调了，这样可能使得词向量无法适应自己的任务。（后续听说不微调确实要好，可以参考[论文](https://arxiv.org/pdf/1903.05987 'To Tune or Not to Tune? Adapting Pretrained Representations to Diverse Tasks')，在知乎上看到这篇论文）这种冻结词向量而不去微调的思想，最直观的体现就是在 $\gamma^{task}$ 这个参数上。。。要自己根据任务去调，那还不如直接微调。那么 ELMo 为什么不能微调呢？
3. 无法更改词表的内容
4. 有人认为还有个缺点是用了 LSTM，没用 Transformer。。。

## 📚参考文献
1. [关于 ELMo 的若干问题整理记录](https://zhuanlan.zhihu.com/p/82602015)

<textarea id="bibtex_input" style="display:none;">
@article{peters2018deep,
  title={Deep contextualized word representations},
  author={Peters, Matthew E and Neumann, Mark and Iyyer, Mohit and Gardner, Matt and Clark, Christopher and Lee, Kenton and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:1802.05365},
  year={2018}
}
</textarea>



