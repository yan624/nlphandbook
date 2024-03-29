## ELMo <!-- {docsify-ignore-all} -->
问：ELMo 为什么要用 BiLM 而不是 BiLSTM？  

问：ELMo 的缺点是什么？  
答：1）训练词向量最好需要上下文信息，语言模型只能得到上文信息，双向语言模型也只能近似地得到上下文信息。2）无法自定义词表，BERT 可以。

## BERT
BERT 采用的是 Transformer encoder 结构，因此二者问题有部分重复，以下就不重复叙述了。

问：[为什么三个嵌入可以直接相加？](https://www.zhihu.com/question/374835153)  
答：暂时没有看到严格的证明。三个嵌入相加其实等价于将三个 one-hot 编码拼接然后通过一个全连接层（视作嵌入层）。例如一个单词的语素、位置、片段三种 one-hot 表示为 $[1, 0, 0, 0, 0], [1, 0, 0], [0, 1]$，则融合之后的 one-hot 编码为 $[1, 0, 0, 0, 0, 1, 0, 0, 0, 1]$。将 one-hot 编码进行拼接的方式可以看作是在一维空间中融合了三种表示方法，特征融合。  

问：如何处理序列长度大于 512 的输入？  
答：见[基础知识_特征提取器_Transformer](基础知识/index#transformer)

问：位置编码？  
答：主要涉及相对位置编码和绝对位置编码的问题，详见 Transformer 一章。

问：为什么 BERT 使用 MLM 和 NSP，而不使用传统的语言模型？  
答：考虑到传统的语言模型只能捕获单向的信息，所以转用 MLM。NSP 是为了适配多样的下游任务，主要考虑到一些基于两条语句的任务，例如关系推理。

问：[NSP 任务真得有用吗？](https://www.zhihu.com/question/331076024)  
答：

问：MLM 有什么缺点？  
答：第一，导致训练阶段与测试阶段输入的不一致性；第二，只会 mask 语素，导致 MLM 任务过于简单。例如将词语“琵琶”拆成“琵”和“琶”，相对来说比较简单，这是因为这两个字的组合比较少。解决办法为：Whole Word Masking（WWM）

问：论文中将 MLM 与 denoising auto-encoders 做了对比，但是我认为 MLM 与 CBOW 也很像，却没有与它对比。问题：三种的区别是什么？  
答：

问：为什么要捕获双向的语言信息？  
答：原文只是说了从直觉上来讲，这么做要更好。

问：BERT 为什么有效？与其他模型相比？  
答：


