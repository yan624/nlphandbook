seq2seq 是自然语言生成任务中的一个重要结构，是编码器解码器架构中的一员，它可以应用在 NLP 的多种任务之上，例如语义解析、神经机器翻译、对话系统等。Seq2Seq 模型于 2014 年由 Bengio 团队[[@Cho_2014]](https://arxiv.org/pdf/1406.1078.pdf 'Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation')首先提出，同年，google 团队[[@sutskever2014sequence]](https://arxiv.org/pdf/1409.3215.pdf 'Sequence to Sequence Learning with Neural Networks')做其做出改进。二者结构如下图所示：

![Cho seq2seq](https://blog-content-1256924128.cos.ap-shanghai.myqcloud.com/zcy/深度学习算法（三）：RNN%20各种机制/Cho%20seq2seq.png 'Cho seq2seq')
![Sutskever seq2seq](https://blog-content-1256924128.cos.ap-shanghai.myqcloud.com/zcy/深度学习算法（三）：RNN%20各种机制/Sutskever%20seq2seq.png 'Sutskever seq2seq :size=50%')

**目前常用的是由 google 团队 Sutskever 等人提出的结构（右图）**。下面对比二者的区别。**从结构上看**，二者的 encoder 几乎一样，只是 decoder 有所不同。对于第一种结构，decoder 的每个时间步都接收 encoder 最后一个时间步的隐藏状态；对于第二种结构，每个时间步只是接收上个时间步的隐藏状态。**从编码器类型上看**，前者使用 simple RNN，后者使用 4-layer bi-LSTM。RNN 的缺点众所周知，好奇的是这篇论文在发表的时候，为什么不用 LSTM，那时 LSTM 应该已经流行了。

除了这些，观察 Cho seq2seq 和 attention 可以发现，**Cho seq2seq 和 attention 类似除了隐藏状态 $h$ 和输入值 $x$ 还需要一个上下文向量 $c$。只不过 Cho seq2seq 的 $c$ 是编码器的最后一个隐藏状态，而 attention 的 $c$ 是编码器各个时间步输出的加权平均**。如果移除两个模型的 $c$ 就变成了 Sutskever seq2seq。

本章只关注 Sutskever seq2seq。接下来，首先描述 seq2seq 的缺陷，然后介绍两种常见训练方式：free-running 和 teacher-forcing。scheduled sampling 是两种训练方式的改进版。最后，介绍 seq2seq 常用的解码方法，集束搜索（beam search）。

## Seq2Seq的缺陷
虽然 seq2seq 模型取得了显著的效果，但是它却迟迟无法达到人类的水平（*此处有待考证*）。其中有几个重要因素制约了 seq2seq 的性能：

1. seq2seq 模型在训练（training）时的输入和在推理（inference）时的输入是不一样的。这种问题被称为 **Exposure Bias**。
2. seq2seq 模型在训练时，要求预测结果必须与参考句一一对应【[3](https://www.jiqizhixin.com/articles/2019-08-10-2 'ACL2019 最佳论文冯洋：Teacher Forcing 亟待解决 ，通用预训练模型并非万能')】。这显然是不合理的。因为文字具有多样性，一词多义或者一义多词的情况比比皆是，甚至英语还具有时态变化。
	- *对于这点，原文想要表达的意思可能并不是我所说的意思，但是我觉得我所说的也是一个比较重要的问题。此外，原文中的第二个因素我觉得实际上与第一个是类似的*
3. 矫枉过正（overcorrect）
4. 无法捕获长句的特征，因为 decoder 的输入是 encoder 最后一个 RNN 的输出，上下文的信息被浓缩进这一个向量，这显然不合理，我们无法将希望寄托于这单独的一个向量上。（由 attention 解决）

目前 teacher-forcing 是亟待解决的问题。由于模型的训练方法和推理方法不同，因此会导致模型在推理时受到很大的影响。目前有一个不算解决办法的办法，就是 Beam Search，详见《[Beam Search](#Beam-Search)》。其次还有 scheduled sampling [@bengio2015scheduled]（一种 curriculum learning），Professor Forcing [@lamb2016professor]，curriculum learning [@bengio2009curriculum]，[@Zhang_2019]。

## free-running和teacher-forcing
按照国（Ge）际（Ren）惯例，结论写在前面。可以看完下面的文章之后，再回来看图，或者边看文章边看图。如下图所示，左边就是 free-running 形式的 seq2seq，是不是很熟悉？因为这是我们经常使用的结构。右图就是 teacher-forcing seq2seq，实际上二者并没有特别大的区别。
![free-running-seq2seq](https://blog-content-1256924128.cos.ap-shanghai.myqcloud.com/zcy/深度学习算法（三）：RNN%20各种机制/free-running-seq2seq.gif ':size=49%')
![teacher-forcing-seq2seq](https://blog-content-1256924128.cos.ap-shanghai.myqcloud.com/zcy/深度学习算法（三）：RNN%20各种机制/teacher-forcing-seq2seq.gif ':size=50%')

seq2seq 具有两种训练方式：1）free-running；2）teacher-forcing。**我们熟知的以及教程上讲的，通常是第一种 free-running**。思路是，将上一个时间步的预测结果输入进当前时间步的 RNN，以此循环往复，直至预测出结束符 `<EOS>` 或者循环到一个给定的次数（*例如解码 90 次*）。由于第一个时间步的特殊性，第一个时间步的输入通常是起始符 `<SOS>`。在推理时，我们用的都是 free-running。

然而，在训练时目标（target）是已知的，所以没必要等到预测出 `<EOS>` 才结束，我们完全可以根据 target 的长度进行控制，例如一句话 20 个字。此外，由于 free-running 的输入是上一个时间步的输出（预测值），可以用“一步错，步步错”来形容。因此使用 **free-running** 训练会产生以下几项问题【[2](https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/)】：

1. 收敛缓慢
2. 模型不稳定
3. 技能匮乏

**teacher-foring 就是为了解决这一问题而提出的**【[1](https://blog.csdn.net/qq_30219017/article/details/89090690)】。根据字面意思，这种方法就好像“老师在教导学生一样”，每个时间步的输入不再是上一个时间步的输出，而是真实的 target。例如 $t-1$ 步时，预测值为“i”，我们假设真实值为“we”，那么在 $t$ 步时，RNN 的输入不是“i”的词向量，而是直接输入“we”的词向量。这跟老师在纠正学生的错误一样。

teacher-forcing 是一个快速且高效的训练方式，但是当待生成的序列与模型在训练过程中看到的不同时，就会导致模型特别**脆弱**或者**有所限制**【[2](https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/)】。目前有一些解决办法：1）Beam Search；2）Curriculum Learning；3）……

## Scheduled Sampling
Scheduled Sampling 由 [@bengio2015scheduled] 首次提出，这是一种 Curriculum learning 策略。

序列预测任务的训练和推理之间的主要区别是：某个时间步的输入是上一个时间步的真实值 $y_{t-1}$，还是上一个时间步的估计值 $\hat{y}_{t-1}$。**scheduled sampling 机制就是在训练阶段，随机地决定使用 $y_{t-1}$ 还是 $\hat{y}_{t-1}$**。以掷硬币的方法来实现，当概率为 $\epsilon_i$ 则使用真实值 $y_{t-1}$，当概率为 $1- \epsilon_i$ 则使用模型的估计 $\hat{y}_{t-1}$。（*注意：在实验中，需要对每一个符号都掷一次硬币。论文中提出，为每个序列掷一次硬币的实验他们也测试过，但是结果更差*）

如果 $\epsilon_i$ 等于 1 就代表是 teacher-forcing 方法，反之就是 free-running 方法。此外，直观来说，当模型训练初期，模型会产生一些随机的符号，这是因为模型还训练得不是很好，如果使用 free-running 方法可能会导致收敛过慢，所以可能应该更多地选择 teacher-forcing。反之，当训练的末尾， $\epsilon_i$ 更应该倾向于 free-running。这样更符合模型的推理情况，同时也可以期望模型已经准备好能够处理未知情况以及采样出合理的符号。

最后一个问题就是，如上一段所述，如何动态地控制抛硬币的概率？其实跟 learning rete decay 是一样的。论文中建议使用一个时间表以此减少 $\epsilon_i$，时间表就是一个以 $i$ 为自变量的函数，并且提出三种方式：

1. Linear decay：$\epsilon_i = max(\epsilon, k - ci)$，其中 $k$ 和 $c$ 分别代表偏移量和斜率，这取决于观测到的收敛速度；
2. Expoential decay：$\epsilon_i = k^i$，其中 $k < 1$ 是一个常量，这取决于观测到的收敛速度；
3. Inverse sigmoid decay：$\epsilon = \frac{k}{k + exp(\frac{i}{k})}$，其中 $k \ge 1$ 取决于观测到的收敛速度。

以上的方法就被称为 **Scheduled Sampling**。由于 scheduled sampling 其实就是以某个概率动态地控制 teacher-forcing 和 free-running，所以 gif 就不做了，可以参考上面《[free-running和teacher-forcing](#free-running和teacher-forcing)》一节的 gif。

## Beam Search
国内关于 beam search 的资料比较少，我搜寻了一番大都数都在知乎，它上面有些许讲解。据 [@Freitag_2017] 论文中所述，beam search 首先由 [@graves2012sequence; @boulanger2013audio] 为 seq2seq 模型提出，此后在机器翻译领域由[@sutskever2014sequence]首次提出。下面给出 wikipedia 上的描述，这是[原链接](https://en.wikipedia.org/wiki/Beam_search)（*需番蔷*）有兴趣可以去网站上看全文。

> In computer science, beam search is a heuristic search algorithm that explores a graph by expanding the most promising node in a limited set. Beam search is an optimization of best-first search that reduces its memory requirements. Best-first search is a graph search which orders all partial solutions (states) according to some heuristic. But in beam search, only a predetermined number of best partial solutions are kept as candidates.[1] It is thus a greedy algorithm.
> 
> The term "beam search" was coined by Raj Reddy of Carnegie Mellon University in 1977.[2]

相较于 beam search 算法，还有两类算法，即贪婪搜索（greedy search）和穷举搜索（exhaustive search）。**穷举搜索可以确保生成的结果是全局最优的**，但是它的搜索空间非常大，可能是个天文数字。*Q：为什么是全局最优的？A：穷举每一个可能，当然是全局最优的。*而**贪婪搜索只能确保局部最优**，它的思想是，在每一个解码步，只取概率最大的那个字。

需要注意的是，每次取概率最大的字并不是最优的。举个简单的例子。**1）**使用贪婪搜索，取 A，B 两个字，概率分别为 0.5 和 0.6，则 AB 的概率为 0.3；**2）**现在使用 beam search，取 C，D 两个字，概率分别为 0.4 和 0.9，则 AB 的概率为 0.36。可以看到在使用 beam search 算法时，第一个字并没有选择概率最大的 A，而是选择了第二大的 C，但是最终结果却是 beam search 要好。因为第二个字的概率不同。*注意：AC两字的概率之和小于 1，BD 两字的概率之和大于 1。这是因为 BD 二字不在同一个 beam 上，而 AC 在。*

### 实战感悟
<del>我个人在实验之中使用 beam search 算法，发现并没有任何效果，甚至结果更差。</del>

2020.06.22 更新：上一段写下约半个礼拜之后，我又去想了一下，如果别人大规模应用这个算法，没道理没效果。所以经过一番代码检查后，我发现了问题。原来之前的代码对于隐藏状态的处理有问题，我将 beam size 大小的预测结果视作了线性的，而不是固定它们的隐藏状态。

具体来说，例如时间步 $t$ 的预测结果为单词 $I$，并且隐藏状态为 $h_I$。接下来假设我们已知下一个时间步的预测结果为 $\{love, hate, told\}$（注：单词预测结果由 Beam Search 产生，并非词表大小只是 3），则它们的隐藏状态输入都应该是 $h_I$ 才是正确的，但是我将它们的隐藏状态输入分别设置为了 $h_I, h_{love}, h_{hate}$，其中 $h_{love}, h_{hate}$ 是将 love 和 hate 输入 LSTM 后得到的隐藏状态。

所以简单来说就是代码写错了……在循环遍历的时候写错了……实际上大概可以提升 3 左右。

## 参考文献
1. [一文弄懂关于循环神经网络 (RNN) 的 Teacher Forcing 训练机制](https://blog.csdn.net/qq_30219017/article/details/89090690)（算是 free-running 以及 teacher-forcing 的一篇综述文章）
2. [teacher forcing for recurrent neural networks](https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/)（上一篇综述部分内容的英语原文）
3. [ACL2019 最佳论文冯洋：Teacher Forcing 亟待解决 ，通用预训练模型并非万能](https://www.jiqizhixin.com/articles/2019-08-10-2)
3. [seq2seq中的beam search算法过程](https://zhuanlan.zhihu.com/p/28048246)
4. [十分钟读懂Beam Search 1：基础](https://zhuanlan.zhihu.com/p/114669778)
20. *[吴恩达李宏毅综合学习笔记：RNN入门](https://yan624.github.io/posts/5e27260b.html#seq2seq)*

<textarea id="bibtex_input" style="display:none;">
@Article{Cho_2014,
  author    = {Cho, Kyunghyun and van Merrienboer, Bart and Gulcehre, Caglar and Bahdanau, Dzmitry and Bougares, Fethi and Schwenk, Holger and Bengio, Yoshua},
  journal   = {Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  title     = {Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation},
  year      = {2014},
  doi       = {10.3115/v1/d14-1179},
  groups    = {enc2dec},
  publisher = {Association for Computational Linguistics},
  url       = {http://dx.doi.org/10.3115/v1/D14-1179},
}
@Misc{sutskever2014sequence,
  author        = {Ilya Sutskever and Oriol Vinyals and Quoc V. Le},
  title         = {Sequence to Sequence Learning with Neural Networks},
  year          = {2014},
  archiveprefix = {arXiv},
  eprint        = {1409.3215},
  groups        = {enc2dec},
  primaryclass  = {cs.CL},
}
@Article{bengio2015scheduled,
  author        = {Samy Bengio and Oriol Vinyals and Navdeep Jaitly and Noam Shazeer},
  title         = {Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks},
  year          = {2015},
  month         = jun,
  abstract      = {Recurrent Neural Networks can be trained to produce sequences of tokens given some input, as exemplified by recent results in machine translation and image captioning. The current approach to training them consists of maximizing the likelihood of each token in the sequence given the current (recurrent) state and the previous token. At inference, the unknown previous token is then replaced by a token generated by the model itself. This discrepancy between training and inference can yield errors that can accumulate quickly along the generated sequence. We propose a curriculum learning strategy to gently change the training process from a fully guided scheme using the true previous token, towards a less guided scheme which mostly uses the generated token instead. Experiments on several sequence prediction tasks show that this approach yields significant improvements. Moreover, it was used successfully in our winning entry to the MSCOCO image captioning challenge, 2015.},
  archiveprefix = {arXiv},
  eprint        = {1506.03099},
  groups        = {enc2dec},
  keywords      = {cs.LG, cs.CL, cs.CV},
  primaryclass  = {cs.LG},
  url           = {http://arxiv.org/pdf/1506.03099v3},
}
@Article{lamb2016professor,
  author        = {Alex Lamb and Anirudh Goyal and Ying Zhang and Saizheng Zhang and Aaron Courville and Yoshua Bengio},
  title         = {Professor Forcing: A New Algorithm for Training Recurrent Networks},
  year          = {2016},
  archiveprefix = {arXiv},
  eprint        = {1610.09038},
  groups        = {enc2dec},
  primaryclass  = {stat.ML},
}
@inproceedings{bengio2009curriculum,
  title={Curriculum learning},
  author={Bengio, Yoshua and Louradour, J{\'e}r{\^o}me and Collobert, Ronan and Weston, Jason},
  booktitle={Proceedings of the 26th annual international conference on machine learning},
  pages={41--48},
  year={2009}
}
@Article{Zhang_2019,
  author    = {Zhang, Wen and Feng, Yang and Meng, Fandong and You, Di and Liu, Qun},
  journal   = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  title     = {Bridging the Gap between Training and Inference for Neural Machine Translation},
  year      = {2019},
  doi       = {10.18653/v1/p19-1426},
  groups    = {enc2dec},
  publisher = {Association for Computational Linguistics},
  url       = {http://dx.doi.org/10.18653/V1/P19-1426},
}
@Article{Freitag_2017,
  author    = {Freitag, Markus and Al-Onaizan, Yaser},
  journal   = {Proceedings of the First Workshop on Neural Machine Translation},
  title     = {Beam Search Strategies for Neural Machine Translation},
  year      = {2017},
  doi       = {10.18653/v1/w17-3207},
  groups    = {enc2dec},
  publisher = {Association for Computational Linguistics},
  url       = {http://dx.doi.org/10.18653/v1/W17-3207},
}
@Article{graves2012sequence,
  author        = {Alex Graves},
  title         = {Sequence Transduction with Recurrent Neural Networks},
  year          = {2012},
  month         = nov,
  abstract      = {Many machine learning tasks can be expressed as the transformation---or \emph{transduction}---of input sequences into output sequences: speech recognition, machine translation, protein secondary structure prediction and text-to-speech to name but a few. One of the key challenges in sequence transduction is learning to represent both the input and output sequences in a way that is invariant to sequential distortions such as shrinking, stretching and translating. Recurrent neural networks (RNNs) are a powerful sequence learning architecture that has proven capable of learning such representations. However RNNs traditionally require a pre-defined alignment between the input and output sequences to perform transduction. This is a severe limitation since \emph{finding} the alignment is the most difficult aspect of many sequence transduction problems. Indeed, even determining the length of the output sequence is often challenging. This paper introduces an end-to-end, probabilistic sequence transduction system, based entirely on RNNs, that is in principle able to transform any input sequence into any finite, discrete output sequence. Experimental results for phoneme recognition are provided on the TIMIT speech corpus.},
  archiveprefix = {arXiv},
  eprint        = {1211.3711},
  keywords      = {cs.NE, cs.LG, stat.ML},
  primaryclass  = {cs.NE},
  url           = {http://arxiv.org/pdf/1211.3711v1},
}
@inproceedings{boulanger2013audio,
  title={Audio Chord Recognition with Recurrent Neural Networks.},
  author={Boulanger-Lewandowski, Nicolas and Bengio, Yoshua and Vincent, Pascal},
  booktitle={ISMIR},
  pages={335--340},
  year={2013},
  organization={Citeseer}
}
</textarea>