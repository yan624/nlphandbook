?> 本文首先以 BERT 为例，描述其📔输入与嵌入表示、🧬模型结构、⚽优化目标、🍺如何使用 BERT 的隐藏状态以及🐞改进与缺陷。最后阐述对 BERT 的疑问（🚀自问自答），然后对本文进行总结。  
[BERT 论文地址](https://arxiv.org/pdf/1810.04805.pdf)，[预训练模型地址](https://github.com/google-research/bert#pre-trained-models)

## 🔖引 言
BERT 共分为两步：预训练（pre-training）和微调（fine-tuning）。预训练阶段，BERT 使用无标签数据对不同的任务进行预训练。在微调阶段，BERT 模型首先加载预训练参数，然后使用下游任务的标签数据微调所有参数。每种任务都有独立的微调模型，即使它们使用相同的预训练参数[[@devlin2018bert]](#devlin2018bert)。这里的微调模型应该指的是下游模型架构，比如分类模型使用 [CLS] 进行线性分类，序列标注使用序列的状态输出。

BERT 的特性是：对于不同任务，预训练结构与最终下游任务结构只具有微小的差异，但是**整体上几乎是统一的**。

BERT 并不是万能的，它无法完成自然语言处理中所有任务，其适用的任务为：自然语言理解（分类、推理）、问答（span-based or not）、SWAG 逻辑连续性推理。

## 💉输入以及嵌入的表示
为了处理多样的下游任务，BERT 的输入“语句”可以是一段任意跨度的连续文本，并非必须为一条真实的语句。“语句”可以表示为单条语句或一对语句的语素序列。

BERT 的嵌入与 RNN 系列模型有所不同，共由三部分组成，分别为词嵌入（word embedding）、位置嵌入（position embedding）以及片段嵌入（segment embedding）。这是因为 BERT 无法捕获序列的语序信息，需要使用位置嵌入表示单词顺序，使用片段嵌入表示句子顺序。在计算时，BERT 采用将三者直接相加的方式从而产生一个全新的嵌入。下面将分别介绍三种嵌入。

?> 相较于 BERT，RNN 系列模型具有捕获序列语序信息的天然优势，它们会顺序地处理单词并生成隐藏状态，其中每个隐藏状态都包含了时序信息。如果句中单词的顺序发生变化，那么单词的隐藏状态也会有所变化。

**词嵌入**有两种初始化方式：1）随机初始化并对其优化；2）使用预训练的词向量，例如 GloVe、word2vec。无论采用以上哪种方式，输入序列均为各单词的 one-hot 编码，以查表的方式从 BERT 的嵌入层中得到词向量。BERT 采用词表大小为 30000 的 WordPiece embeddings 初始化所有词嵌入。

**位置嵌入**：由于 BERT 天然不具有捕获序列顺序信息的能力，因此如何表示位置嵌入就显得至关重要。其具有多种表示方法：Transformer 主要尝试两种，即训练好的嵌入以及正弦位置嵌入。BERT 采用预先训练的方式。

> 更多关于位置嵌入的讨论详见 Transformer 一章。

**片段嵌入**：BERT 使用一个预先训练好的嵌入[[@devlin2018bert]](#devlin2018bert)。

## 🧬模型结构
BERT 的模型结构使用多层双向的 Transformer encoder，论文中使用 [tensor2tensor](https://github.com/tensorflow/tensor2tensor) 库进行实现，他们发布了多个版本的 BERT，其中 $BERT_{BASE}$ 的参数与 GPT 一致，主要为了与其对比。

## ⚽优化目标
与 ELMo 和 GPT 不同，BERT 没有使用传统的从左至右或者从右至左的语言模型。而是使用两种其他的无监督任务，分别为 Masked Language Model（MLM） 以及  Next Sentence Prediction（NSP）。

### Masked LM
直觉上来讲，有理由相信**深层双向模型**比**从左至右**或者**浅层的从左至右以及从右至左的拼接模型**要好。**不幸的是，标准的语言模型只能做到捕获一个方向的语言特征**，*原因如下，但是我认为这两句话没有因果关系吧？*

> Unfortunately, standard conditional language models can only be trained left-to-right or right-to-left, since bidirectional conditioning would allow each word to indirectly “see itself”, and the model could trivially predict the target word in a multi-layered context.

为了训练一个深层的双向语言模型，BERT 以一定概率（论文中为 15%）随机地掩盖输入语句中的部分语素（语素指 wordpiece token），然后预测这些语素。这被称为 MLM，在文学领域这被称为完形填空（Cloze）。基于该做法，被掩盖语素的最终隐藏向量被输入一个基于词表大小的 softmax，这与标准的语言模型相同。与 denoising auto-encoders (Vincent et al., 2008) 相比，BERT 只是预测被掩盖的单词，而不是重构整个输入。

尽管基于MLM的BERT拥有了强大的性能，然而却引入了一个缺点，即**预训练和微调阶段的不一致性，这是因为 `[MASK]` 没有出现在微调阶段**。为了缓解这一问题，BERT 不总是使用 `[MASK]` 替换单词。训练数据生成器以 15% 的概率选择一个语素位置，在进行掩盖时分为以下三种情况：（1）80% 替换；（2）10% 替换成一个随机的语素；（3）10% 不做变化。然后 $T_i$ 将被用于预测真实语素，其基于交叉熵损失函数。以上三种方式各种变体的对比详见附录 C.2。

### NSP
许多重要的下游任务都需要理解两个句子之间的关系，例如 QA 和 NLI，这并不是语言模型能捕捉到的。为了使模型能够理解句子之间的联系，BERT 还额外训练了 NSP 任务。具体来讲，对于句子 A 和 B，50% 的情况下 B 就是 A 的下一句（标签为 `IsNext`）；50% 的情况下 A 的下一句是一条随机语句（标签为 `NotNext`）。

## 🍺如何使用隐藏状态
BERT 使用的框架分为两步：预训练和微调，这与 ELMo 不同，它使用的是预训练和基于特征（feature-based）。所以 ELMo 需要决定使用哪层的隐藏状态作为下游任务的输入，而 BERT 只需要在模型的顶部加上下游任务的模型，然后继续微调即可。

## 🐞改进与缺陷
改进：  
1. 使用双向 Transformer
2. 引入 MLM
3. GELU

缺陷：  
1. 输入是 wordpiece，这让基于 span 的任务很难办。对于命名体识别来说，每一个单词都有对应的标签，如果用 wordpiece 算法会打乱标签。BERT 的做法是只使用 sub-word 的第一个部分的隐藏状态，详情可参考[issue1](https://github.com/huggingface/transformers/issues/323)。需要注意，BERT 的论文应该做过修改，issue1 中提到的段落，现在应该在 5.3。原文我放在下面。然而这只能解决命名体识别任务，无法解决基于位置信息的任务，例如基于 index-based Ptr 模型做的任务。
	- > We use the representation of the first sub-token as the input to the token-level classifier over the NER label set.
1. 最大长度限制为 512，无法处理长文本。即使能处理也是狗尾续貂，有些论文中使用额外的技巧进行处理，但是这样就比较强行。
2. 提高预测准确率到底来源于预训练？还是数据量大？还是依靠的底层模型？
3. 只能输入两句话？
4. 由于 Transformer 完全没有语序的概念，所以需要用位置编码。
5. 任务方面：无法完成语言生成任务
7. 硬件方面：模型过于庞大，如果使用 BERT 提取特征，接下来的分类层只能设计小一点

## 📔总结
1. BERT 由预训练以及微调两步组成，预训练步骤使用大规模的无标签文本数据优化参数，微调步骤使用标注数据优化预训练过的参数。
2. BERT 的嵌入由三部分组成，分别为语素、位置以及片段。在模型进行计算时，将三者相加。
3. 模型的结构采用多层双向的 Transformer Encoder，而非ELMo 使用的多层“双向” LSTM 或者 GPT 的多层单向 Transformer Decoder。这是因为从直觉上来讲，深层双向模型可以获得更好的性能表现。ELMo 的双向模型只是将两个单向模型进行拼接。不幸的是，传统的语言模型只能捕获一个反向的信息，为此引入 Masked Language Model（MLM）。
4. 但是 MLM 又导致训练阶段和测试阶段数据的不一致性。这是因为在训练阶段，BERT 使用 `[MASK]` 特殊符号去替换输入序列中的一个随机语素，而在测试阶段却移除了该特殊符号。为此进行了特殊的处理：1）80% 的概率执行替换；2）10% 的概率使用一个随机单词替换；3）10% 的概率保持不变。
5. 为了使 BERT 能够适应更多的下游任务，还引入了 NSP 任务。

## 📚参考文献
1. [BERT 的演进和应用](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650411744&idx=2&sn=1db39446e4e91299f9ba8c1d4eeb5983&chksm=becd94ba89ba1dac221e2092cb1a12ecb1a5a704b6ae5cebd2649adc1a903355b95567ab484e&mpshare=1&scene=1&srcid=&sharer_sharetime=1572502447054&sharer_shareid=68f8b84d7a46cc216b0afdc45278d6be&key=8a4bbb55c6c79ce6c9104a6cfe5de2a3d1b8fa801c35e22e74b62948f50b6684c3f06195815e8712080977db6cec80fca5adfc95c9bc6fa848b8e68b41df13d8610e8d6c283ee2392b30de5cdae504bb&ascene=1&uin=MTQxMTUzMzk2MA%3D%3D&devicetype=Windows+10&version=62070152&lang=en&pass_ticket=pZijWLQmmCpNBDcjO4cUImTRWv1ZWLG4JENv1zUqjhXnUnShPGofPjjR%2Bkv1cozV)
2. [万字长文带你纵览 BERT 家族](https://zhuanlan.zhihu.com/p/145119424)
3. [A Survey on Contextual Embeddings](https://arxiv.org/pdf/2003.07278.pdf)
4. [Pre-trained Models for Natural Language Processing: A Survey](https://arxiv.org/pdf/2003.08271.pdf)
5. [从 Word Embedding 到 Bert 模型 — 自然语言处理中的预训练技术发展史](https://zhuanlan.zhihu.com/p/49271699)

<textarea id="bibtex_input" style="display:none;">
@article{devlin2018bert,
  title={Bert: Pre-training of deep bidirectional transformers for language understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
</textarea>



