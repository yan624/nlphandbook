encoder-decoder 模型是一种范式，它主要由两个部分组成，即 encoder 和 decoder。encoder 负责提取输入的特征，decoder 负责运用提取出的特征并且生成（v.）任务所需要的目标。而在 NLP 领域之中，由于输入以及输出都是一段序列，所以又可以被称为 Sequence-to-Sequence 模型，简写为 Seq2Seq。所以 encoder-decoder 其实还可以有其他的变体，例如在图像描述（Image Caption）生成领域，encoder 就是一个 CNN，decoder 是一个 RNN。

## seq2seq<!-- {docsify-ignore-all} -->
问：讲讲 seq2seq 以及它的训练方法和解码方法。  
答：seq2seq 可以接收一个序列序号通过编码和解码生成另一个序列信号，应用广泛，例如机器翻译、智能对话等任务。**seq2seq 有两种简单的训练方法**，分别是 free-running 和 teacher-forcing。使用 free-running 训练时，解码器将上一个时间步的预测结果作为输入；使用 teacher-forcing 训练时，解码器将上一个时间步的目标单词作为输入。前者收敛缓慢、模型不稳定。后者使模型在训练时和推理时的分布不同，这被称为 Exposure Bias。一个简单的改进方法是 scheduled sampling。**seq2seq 最简单的解码方法**是贪婪搜索（greedy search），即每个时间步取概率最大的单词；穷举搜索 （exhaustive search）可以确保生成的结果是全局最优，但它的搜索空间十分庞大。一个简单的改进办法是集束搜索（beam search），即每个时间步取前 k 个最好的结果，在分别进行解码之后再取前 k 个最好的组合，以此类推直至 k 个可能都解码完毕，最后取最好的那个结果作为输出。

问：为什么 free-running 收敛缓慢、模型不稳定？  
答：收敛缓慢是因为：在训练初期时，模型的输入是一些随机的结果，这让模型需要花费更多的时间优化参数。*模型不稳定是因为：在训练时，模型的输入太过随机？*

问：展开讲讲 scheduled sampling？  
答：

问：seq2seq 有什么缺点？  
答：第一，seq2seq 要求预测结果与真实值完全相等，这在现实中是不现实的。第二，seq2seq 在训练时输入的是真实值，在推理时输入的是预测值，它们的分布不同，这被称为 exposure bias。目前的解决办法是使用 scheduled sampling。第三，普通的 seq2seq 无法处理长句，这是因为输入被编码进一个向量中，现在一般使用 attention 缓解这一问题。

## attention
问：attention 机制有什么作用？  
答：  
attention 可以从源端的大量信息中筛选出**少量**重要的信息并聚焦到这些信息上，聚焦体现在各个信息的权重上。权重越大说明该信息越重要。

假设有源句 $X=\{x_1, x_2, x_3\}$，目标句为 $Y=\{y_1, y_2, y_3, y_4\}$。如果现在生成 $y_3$，就可以使用概率公式表示为 $p(y_3 | X y_1 y_2)$。

假设 $y_3$ 与 $x_1$ 无关，那么它们可以看作是独立的，也就是 $p(y_3 | X y_1 y_2) = p(y_3 |x_2 x_3 y_1 y_2)$。普通的 seq2seq 只做到了等式的左边，还需要额外学习 $y_3$ 和 $x_1$ 之间的关系，这是无意义的，而 attention 做到了等式的右边。

问：有哪些常见的 attention？  
答：attention 可以根据对齐函数（alignment function）和评分函数（score function）划分。根据对齐函数，分为 global attention 和 local attention。global attention 在解码时与编码器的所有输入进行交互并计算权重，然后融合它们的信息；local attention 只进行局部的交互。[@luong2015effective] 的实验表明二者没什么太大区别，一般选择较容易实现的 global attention。原始的 attention 拥有四种评分函数，即 dot、general、concat 和 location。

问：能否手写一下 attention 以及四种评分函数（计算流程）？  
答：*自己手写一下，熟练点。*

问：讲讲 local attention？  
答：

问：讲讲 Bahdanau attention 和 Luong attention 的区别。  
答：

## PtrNets


## 参考文献
<textarea id="bibtex_input" style="display:none;">
@article{luong2015effective,
  title={Effective approaches to attention-based neural machine translation},
  author={Luong, Minh-Thang and Pham, Hieu and Manning, Christopher D},
  journal={arXiv preprint arXiv:1508.04025},
  year={2015}
}
</textarea>








