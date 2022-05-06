## 特征缩放 <!-- {docsify-ignore-all} -->
问：讲讲常见的特征缩放方法  
答：最值缩放、均值标准化、均值方差标准化

问：为什么要缩放数值型特征？  
答：1）主要是为了消除不同数据之间量纲的影响，使它们位于同一数量级。2）*可以加速模型收敛*。

问：为什么特征缩放可以加速收敛？  
答：从二维角度考虑，如果不标准化，在梯度下降时，每一步并不是朝着圆心（即最小值）前进的，或者说法线方向不朝着圆心。需要花费更多时间纠正方向。标准化后各维度的量纲一致，就会好很多。对更高维度不太清楚。

问：归一化 （normalization） 和标准化 （standardization） 有什么区别？  
答：归一化将数据映射到一个区间内，一般为 $[0,1]$；标准化将数据映射到负无穷到正无穷的区间内，其均值为 0， 标准差为 1。 归一化消除量纲的影响；标准化改变数据分布，变为正态分布。

问：为什么要将数据映射为正态分布？对模型训练有什么好处？  
答：

问：如何选择合适的特征缩放方法？  
答：特征缩放方法主要分两种：归一化（例如 min-max scaling）和标准化（例如 Z-score normalization）。当数据对输出范围有要求或者数据较为稳定、不包含噪音时，可选择归一化；反之，选择标准化。

## 文本表示
问：词袋模型有什么适用场景？  
答：详见 TF-IDF 或其它算法。词袋模型实际上只是一种概念，词的权重可以由简单的 one-hot 编码或词频表示，也可以用更复杂的 TF-IDF 或者其它算法表示。

问：什么是 TF-IDF？  
答：TF-IDF 是用来衡量某个单词在一个文档中的重要性，表示为一个数字。通过 TF-IDF 算法可以得到一段文本的向量。与之不同，word2vec 中每个单词都有一个向量，一段文本则是一个矩阵。

问：TF-IDF 怎么计算文档的主题？  
答：计算出文档中每个词的权重后，进行降序排序，选择前几个作为关键词，即主题。

问：TF-IDF 怎么计算两个文档的相似度？  
答：如果是短文本，那么可以用 TF-IDF 将它们转换为两个向量，然后计算余弦相似度（也可以使用其他算法）；如果是长文本，例如文档，那么可以首先计算文档的主题（数量需要相等），再用 TF-IDF 将主题转化为向量，最后计算余弦相似度。  
还看到过[另一种](https://zhuanlan.zhihu.com/p/113017752)计算相似度的方法，遍历输入文本 $A$ 和待计算文本 $B$，如果 $A$ 中的单词出现在 $B$ 中，则累加单词对应的权重，否则跳过，即视其权重为 0。例如给定一个问题，从候选文档中挑选出一个最相似的问题（医疗咨询）。

### 词嵌入模型
问：word2vec 和 TF-IDF 在计算相似度上有什么区别？  
答：word2vec 主要计算单词之间的相似度，但是也可以计算文档的；TF-IDF 只能计算文档之间的相似度。

问：word2vec 和 NNLM 有什么区别？  
答：
1. 目的不同：word2vec 旨在训练词向量，而 NNLM 旨在训练神经语言模型，词向量是它的副产物。
2. 模型结构不同：基于 CBOW 的 word2vec 的输入是上下文每个词向量的累加，NNLM 是拼接。为了加速训练，word2vec 没有隐藏层，只有输入输出层；NNLM 有隐藏层。
3. 模型算法不同：word2vec 使用了 CBOW 和 skip-gram 两种算法；NNLM 作为语言模型使用的是条件概率。为了提供解码效率，word2vec 还使用了 hierarchical softmax 和 negative sampling；NNLM 则仅使用了 softmax。

问：讲讲 word2vec 的两种训练算法（CBOW 和 skip-gram）。  
答：CBOW 利用单词 $w_i$ 的上下文 $\text{context}(w_i)$ 预测 $w_i$ 的概率。在上下文各个词向量输入模型时，采用累加的方法。skip-gram 使用 $w_i$ 预测它的上下文，公式为 $p(\text{context}(w_i)| w_i) = \prod_{u \in \text{context}(w_i)} p(u|w_i)$。其中每个 $p(u|w_i)$ 可用 hierarchical softmax 或 negative sampling 解码，与 CBOW 类似。

问：讲讲 word2vec 的两种解码算法（hierarchical softmax 和 negative sampling）。  
答：**heirarchical softmax** 首先根据词表中每个单词的词频构造一棵哈夫曼树，每个叶结点分别代表词表中的各个单词，词频越高叶结点离根结点越近。然后，给每个**非叶结点**分配一个权重向量，用于执行二分类。模型优化的是哈弗曼编码。比如 1001 需要进行四次二分类。最后单词 $w_i$ 的条件概率 $p(w_i|\text{context}(w_i)) = \prod_d p(label_d)$，其中 $d$ 代表哈夫曼树的深度，该公式意味着将 $w_i$ 路径上的每个概率相乘。  
**negative sampling**

问：推导一遍基于 CBOW 的算法？解码方法用 heirarchical softmax。（从模型输入开始，推导到更新参数）   
答：

-----

问：讲讲 GloVe？  
答：GloVe 首先从语料库中统计单词 $j$ 出现在单词 $i$ 附近的次数 $X_{ij}$，然后计算单词 $i,j$ 的共现概率 $P_{ij} = \frac{X_{ij}}{X_i}$。其中 $X_i$ 代表任意单词出现在单词 $i$ 附近的总次数。最后通过探查词 $k$ 可以计算共现概率的比值 $r = \frac{P_{ik}}{P_{jk}}$，以此计算单词 $i,j$ 的关系。
基于这种可以从语料库统计出的已知关系，我们定义一个函数 $\hat{r} = f(i,j,k)$，使得函数的输出 $\hat{r}$ 尽可能的接近 $r$。可以使用均方差作为损失函数以此进一步优化。函数的输入就是三个词向量，在训练过程中不断地优化它们。

问：GloVe 和 word2vec 的区别、关联。跟 LSA 呢？  
答：

-----

问：讲讲 FastText？  
答：FastText 是一个简易的线性分类器，考虑到 BOW 没有词序信息，因此输入是 n 元词袋，具体来说是二元词袋。为了进一步加速模型训练和推理，FastText 还使用了 hierarchical softmax。此外，还使用了 hashing trick 减少存储 n-gram 的内存。

问：FastText 和 word2vec 有什么关联和区别？  
答：关联有：FastText 使用了 word2vec 的 CBOW 模型，都使用了 hierarchical softmax。区别有：FastText 的输入是文档的 n-grams，word2vec 的输入是中间词的上下文；FastText 预测的是文档的标签，word2vec 预测的是中间词。

## 语言模型

## 特征提取器

### RNN
问：simple RNN 有什么优缺点？  
答：优点是：1）能够处理序列数据；2）拥有记忆，能够捕捉到数据的先后顺序信息。缺点是：1）只有短期依赖；2）有梯度消失和梯度爆炸。

问：讲一下 RNN 的梯度消失和梯度爆炸？  
答：梯度消失和梯度爆炸产生的**基本原因**都是梯度的连乘。与其它的神经网络不同 ，RNN 的**根本原因**有两项：**一是（与其它神经网络一样的）激活函数导数的累乘，二是权重矩阵的累乘。**

问：对于 RNN 的梯度消失，你有没有想补充的？  
答：RNN 的短期依赖问题，[RNN 的梯度消失是个伪命题](https://www.zhihu.com/question/275856902)。在某个时间步对 RNN 求导，会得到一个比较复杂的式子。因为太复杂了，我举个例子。类似于 0.001 + 0.01 + 0.1 + 1。尽管有几项确实梯度消失了，但是总梯度并没有消失。那些梯度消失的项是离当前时间步很远的信息，这些信息无法反馈到当前时间步，所有 RNN 只有短期依赖。

补充：对 RNN 的权重求导之后，可以得到 ${S_t}' \cdot a_{t-1} + \sum^t_{i=2} (W_{aa})^{t-i+1} (\prod^{t}_{k=i-1} {S_k}') \cdot a_{i-2}$。可以看到 $t$ 时间步的梯度来自 $t$ 之前所有时间步的信息之和。在 $t$ 时，$W \cdot S'$ 的值并不会很大或很小，而当时间步为 1 时，$(W \cdot S')^t$ 有可能很大或很小，从而引发梯度爆炸或梯度消失。$W \cdot S'$ 本质上是 $\frac{\partial{a^t}}{\partial{a^i}}, i=0, \dots, t-1$。综上，想要处理梯度爆炸和梯度消失，可以从几个方向入手：1）将激活函数替换为 relu；2）权重矩阵的特征值最好在 1 附近，即将其初始化为正交矩阵；3）将 $\frac{\partial{a^t}}{\partial{a^i}}$ 分解，这是目前常用的做法，比如 LSTM、GRU。

问：怎么初始化 RNN 的权重？  
答：由于 $W$ 的连乘，导致我们必须小心地初始化 $W$。很自然的可以想到 $W$ 应该初始化在 1 附近。然而我看了 pytorch 的官方实现，他们用的是均匀分布。网上推荐使用正交分布，因为正交矩阵的特征值的绝对值是 1。[原文](https://smerity.com/articles/2016/orthogonal_init.html)，[中文翻译](https://zhuanlan.zhihu.com/p/28981495)，[提到了用正交初始化](https://www.zhihu.com/question/57828011/answer/155275958)。

> 正交矩阵有许多有意思的特性，最重要的就是正交矩阵的特征值绝对值等于 1。**这意味着，无论我们重复多少次矩阵乘法，矩阵的结果既不会爆炸也不会消失。**

问：RNN 为什么要用 tanh？能不能用 relu？  
答：第一个问题不知道。对于第二个，RNN 可以使用 relu，只不过在用的时候需要将权重初始化为单位矩阵，将偏差初始化为 0 [@Le2015]。如果不这么做在前向传播时会导致 $W$ 连乘导致梯度消失或爆炸。

问：tanh 对比 sigmoid 有什么优点？  
答：见《激活函数》一节。

问：LSTM 主要解决了 RNN 的什么问题？  
答：LSTM 缓解了梯度消失问题；优化了 memory 机制（使用加法）；相比于 RNN 具有更长的短期依赖。注意：说 LSTM 梯度消失问题指的是防止 RNN 梯度连乘，实际上 RNN 并没有广义上的梯度消失。此外，由于 LSTM 解决了梯度连乘问题，使其拥有更长的短期依赖。

问：LSTM 为什么没有解决梯度爆炸？  
答：在反向传播中，LSTM 将梯度的连乘分解为，几个门的梯度的和的连乘。这些梯度的值是不同的，可以一定程度地缓解梯度问题。而 RNN 中的这些梯度是 $Wf'$，其中 $W$ 是相同的，$f' \in [0, 1]$。（*无参考文献，个人理解*）

问：LSTM 的设计和残差网络有什么区别？残差网络可以解决前馈神经网络的梯度消失问题，为什么 RNN 不直接用残差网络？  
答：残差网络主要是用来解决网络退化问题的。RNN 暂时还没有这种问题，它的长期依赖问题都还没解决。残差连接一般用于训练更深的模型，RNN一般只有一层。参考[文章](https://zhuanlan.zhihu.com/p/80226180)

### Transformer
问：self-attention 的优缺点？  
答：优点：可以并行处理文本序列；attention 具有可解释性、可视化；缺点：缺少位置信息。

问：[为什么要使用多头机制？](https://www.zhihu.com/question/341222779)  
答：原文的回答是：多头机制允许模型关注来自不同表征子空间的信息 [[@vaswani2017attention]](#vaswani2017attention)。网上有些看法是，多头机制类似于集成学习（ensemble）。

问：为什么点积之前需要缩放？[参考1](https://www.zhihu.com/question/339723385)、[参考2](https://blog.csdn.net/qq_37430422/article/details/105042303)  
答：这是为了不让 attention score 在经过 softmax 之后得到的概率分布出现异常。如果出现异常可能会导致梯度消失。  
*不缩放可能会让点积结果变得很大或很小，无论哪种都会导致梯度趋于 0，从而造成梯度消失，难以训练网络。*  
**解释点积结果很大或很小**：假定 q 和 k 中的元素的均值为 0，方差为 1，那么 $qk^T$ 均值为 0，方差为 $d_k$，其中 $d_k$ 为向量维度。可以看出当向量维度很大时，点积结果的方差很大，也就是说部分元素会出现异常值。**注**：Layer Norm 将向量中的元素变为均值为 0，方差为 1 的分布。点积之后方差为 $d_k$ 是因为 $var(q \cdot k^T) = \sum_{i} var(q_i k_i) = d_k$，其中 $var(q_i k_i) = 1$ 是已知的。  
**解释无论哪种都会导致梯度趋于 0**：对 softmax 函数 $S(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$ 求导可得：

$$
\begin{aligned}
	\frac{\partial{S(x_i)}}{\partial{x_i}} & = S(x_i) \cdot (1 - S(x_i)), & \text{if} \ i = j \\
	\frac{\partial{S(x_i)}}{\partial{x_j}} & = -S(x_i) \cdot S(x_j), & \text{if} \ i \neq j \\
\end{aligned}
$$

softmax 会把向量中数量级较大的元素投影到 1 上，其余元素趋于 0。自己做个小实验就可以知道，softmax 对数量级大小很敏感，大约 10 来倍的差距就会发生以上现象。因此，当 $x_i$ 相对其它元素很大时，$S(x_i)$ 趋向于 1，上述二者都趋向于 0；当 $x_i$ 相对其它元素很小时，$S(x_i)$ 趋向于 0，二者还是趋向于 0。  
**综上所述**，很有必要让 q 和 k 中每个元素之间的差距不要大，也就是让它们的方差变小。Transformer 通过除以 $\sqrt{d_k}$ 将它们还原了。为什么要除以 $\sqrt{d_k}$？因为方差计算公式 $var(kx) = k^2 var(x)$。

问：[为什么 Transformer 要使用不同的 Q 和 K，相同不行吗？](https://www.zhihu.com/question/319339652)  
答：如果相同，那么 Q 在和所有向量计算时，大概率和自身点积的值是最大的，即得到一个斜对角元素最大的注意力矩阵。在归一化之后，会得到一个类似于单位矩阵的注意力矩阵。这显然与 attention 机制的初衷不符。与自身点积最大是因为，两个相同向量的夹角是 0，而与其它任意夹角的向量不太可能是 0。当向量夹角等于 0 时，向量内积最大。*KV 相同是可以的，最初的 attention 就是这样。*

问：Transformer 用位置编码有何意义？  
答：

问：Transformer 残差结构的意义？  
答：防止模型退化

问：讲讲 Transformer 的结构？  
答：

问：优化 self-attention？  
答：

问：layer norm 的放置位置？  
答：

问：Transformer 为什么用 Layer Normalization，而不是 Batch Normalization？  
答：1）Batch Norm 很难应用在 NLP 领域，因为一批文本输入，它的序列长度不一定等长。即使可以 padding，但这种特殊符号本身是没有意义的。2）Batch Norm 本身还有缺陷。它要求计算一批数据的均值和方差，当批次较小时，均值和方差不足以反映真实数据。3）对一批文本数据用 Batch Norm 不是很符合直觉，不同的序列之间没有什么关系。

问：如何处理大于 512 长度的输入？  
答：1）观察数据的特征，尝试截断头部或者截断尾部 [@Sun2019]；2）分段捕捉特征再拼接或接入其他神经网络；3）transformer-xl；4）LongFormer……

## 子词算法
1. BPE 算法有什么优缺点？
	- 优点：BPE 算法平衡了词表大小和每句所需编码符号的数量。1）与使用 unk 相比，略微增加了词表大小；与大型词表的做法相比，减少了词表数量；2）与使用 unk 相比，略微增加了序列长度；与 character-based 方法相比，减少了序列长度。
	- 缺点：BPE 基于贪婪且不可逆转的符号替换操作，无法提供多种分词的概率。
2. BPE 算法与 NMT-BPE 算法有什么区别？  
	- BPE 算法将常见的连续字节替换为一个不曾出现在数据中的字节，例如“aaabdaaabac” → “ZabdZabac” → “ZYdZYac” → “XdXac”。注意，BPE 算法合并的是字节，上面用字母只是用于举例。
	- NMT-BPE 算法首先将序列按字符切分，然后把在语料中最频繁出现的字符或者字符序列合并。
3. BPE 和 joint BPE 有什么区别？各自的优点是什么？  
> 1）后者将源词表和目标词表合并，而前者保持独立。  
> 2）前者的文本和词表容量更小，并且更能确保每个子词都与训练文本有关；后者能够提高源分词和目标分词的一致性。（如果独立地对两个词表使用 BPE 算法，那么同一个名字可能会有两种切分方法）
4. 为什么最终词表大小等于初始词表大小加合并操作次数？我觉得不一定，详见上文。
5. 如何选择合并操作次数这个超参数？  
6. WordPiece 算法有什么优缺点？
7. BPE、WordPiece、ULM 有什么区别？
	- BPE、WordPiece 的词表由小变大，而 ULM 预先建立词表，然后通过评估规则不断丢弃部分子词从而逐渐减小词表。
	- BPE 无法得到分词的概率

## 参考文献
<textarea id="bibtex_input" style="display:none;">
@Article{Le2015,
  author        = {Quoc V. Le and Navdeep Jaitly and Geoffrey E. Hinton},
  title         = {A Simple Way to Initialize Recurrent Networks of Rectified Linear Units},
  year          = {2015},
  month         = apr,
  abstract      = {Learning long term dependencies in recurrent networks is difficult due to vanishing and exploding gradients. To overcome this difficulty, researchers have developed sophisticated optimization techniques and network architectures. In this paper, we propose a simpler solution that use recurrent neural networks composed of rectified linear units. Key to our solution is the use of the identity matrix or its scaled version to initialize the recurrent weight matrix. We find that our solution is comparable to LSTM on our four benchmarks: two toy problems involving long-range temporal structures, a large language modeling problem and a benchmark speech recognition problem.},
  archiveprefix = {arXiv},
  eprint        = {1504.00941},
  groups        = {RNN},
  keywords      = {cs.NE, cs.LG},
  primaryclass  = {cs.NE},
  readstatus    = {skimmed},
  url           = {http://arxiv.org/pdf/1504.00941v2},
}
@InCollection{Sun2019,
  author    = {Chi Sun and Xipeng Qiu and Yige Xu and Xuanjing Huang},
  booktitle = {Lecture Notes in Computer Science},
  publisher = {Springer International Publishing},
  title     = {How to Fine-Tune {BERT} for Text Classification?},
  year      = {2019},
  pages     = {194--206},
  doi       = {10.1007/978-3-030-32381-3_16},
  groups    = {finetuning},
  printed   = {printed},
}
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  pages={5998--6008},
  year={2017}
}
</textarea>


