## LayerNorm的放置位置
layer normalization 有两种方式与残差网络结合，分别称为：**Post-Norm** 和 **Pre-Norm** [@wang-etal-2019-learning-deep]。前者放在残差网络之后，后者放在 multi-head attention 或者 FFN 之前，也就是残差网络之内。前者属于原始 Transformer 的做法。

两种做法都是对 Transformer 不错的实现。@wang-etal-2019-learning-deep 的实验显示对于 6 层编码器的系统，两种做法的 BLEU 指标类似。然而，当转用更深的模型后，情况就变得不一样了。他们发现如果使用更深的模型，在训练时 pre-norm 比 post-norm 更有效。这可以通过观察反向传播去解释，详见原文 2.2 节。**pre-norm 的优点是误差梯度不取决于 Transformer 堆叠的深度，注意式(6)右侧括号中有一个 1，这起码保证了梯度不会消失**（与 RNN 的梯度消失类似）。而式(5)右侧没有这种结构，有的只是很多乘积项，这可能会引发梯度消失或爆炸。

@pmlr-v119-xiong20b 从理论上研究了 Transformers 的梯度，并且发现靠近 post-norm Transformer 输出层的梯度在训练初期很大。这也是为什么 post-norm Transformer 没有 warm-up 训练策略导致其训练不稳定。他们因此推断然后实验验证了 pre-norm Transfromer 可以移除 warm-up 阶段。[@Lin2021]

综上所述，当模型需要叠更多层时，可以选择 pre-norm。结果显示当层数达到 20 时，已经无法训练原始的 post-norm Transformer。*注：@wang-etal-2019-learning-deep 表格中还汇报了其他研究人员用一些方法训练到了 16 层。*此外，在使用 pre-norm 时，不需要 warm-up 阶段。

## LayerNorm的替代品

## Normalization-free Transformer
ReZero

## 参考文献
<textarea id="bibtex_input" style="display:none;">
@inproceedings{wang-etal-2019-learning-deep,
    title = "Learning Deep Transformer Models for Machine Translation",
    author = "Wang, Qiang  and
      Li, Bei  and
      Xiao, Tong  and
      Zhu, Jingbo  and
      Li, Changliang  and
      Wong, Derek F.  and
      Chao, Lidia S.",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P19-1176",
    doi = "10.18653/v1/P19-1176",
    pages = "1810--1822",
    abstract = "Transformer is the state-of-the-art model in recent machine translation evaluations. Two strands of research are promising to improve models of this kind: the first uses wide networks (a.k.a. Transformer-Big) and has been the de facto standard for development of the Transformer system, and the other uses deeper language representation but faces the difficulty arising from learning deep networks. Here, we continue the line of research on the latter. We claim that a truly deep Transformer model can surpass the Transformer-Big counterpart by 1) proper use of layer normalization and 2) a novel way of passing the combination of previous layers to the next. On WMT{'}16 English-German and NIST OpenMT{'}12 Chinese-English tasks, our deep system (30/25-layer encoder) outperforms the shallow Transformer-Big/Base baseline (6-layer encoder) by 0.4-2.4 BLEU points. As another bonus, the deep model is 1.6X smaller in size and 3X faster in training than Transformer-Big.",
}
@InProceedings{pmlr-v119-xiong20b,
  author    = {Xiong, Ruibin and Yang, Yunchang and He, Di and Zheng, Kai and Zheng, Shuxin and Xing, Chen and Zhang, Huishuai and Lan, Yanyan and Wang, Liwei and Liu, Tieyan},
  booktitle = {Proceedings of the 37th International Conference on Machine Learning},
  title     = {On Layer Normalization in the Transformer Architecture},
  year      = {2020},
  editor    = {III, Hal Daumé and Singh, Aarti},
  month     = {13--18 Jul},
  pages     = {10524--10533},
  publisher = {PMLR},
  series    = {Proceedings of Machine Learning Research},
  volume    = {119},
  abstract  = {The Transformer is widely used in natural language processing tasks. To train a Transformer however, one usually needs a carefully designed learning rate warm-up stage, which is shown to be crucial to the final performance but will slow down the optimization and bring more hyper-parameter tunings. In this paper, we first study theoretically why the learning rate warm-up stage is essential and show that the location of layer normalization matters. Specifically, we prove with mean field theory that at initialization, for the original-designed Post-LN Transformer, which places the layer normalization between the residual blocks, the expected gradients of the parameters near the output layer are large. Therefore, using a large learning rate on those gradients makes the training unstable. The warm-up stage is practically helpful for avoiding this problem. On the other hand, our theory also shows that if the layer normalization is put inside the residual blocks (recently proposed as Pre-LN Transformer), the gradients are well-behaved at initialization. This motivates us to remove the warm-up stage for the training of Pre-LN Transformers. We show in our experiments that Pre-LN Transformers without the warm-up stage can reach comparable results with baselines while requiring significantly less training time and hyper-parameter tuning on a wide range of applications.},
  groups    = {Transformer},
  pdf       = {http://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf},
  url       = {https://proceedings.mlr.press/v119/xiong20b.html},
}
@Article{Lin2021,
  author        = {Tianyang Lin and Yuxin Wang and Xiangyang Liu and Xipeng Qiu},
  title         = {A Survey of Transformers},
  year          = {2021},
  month         = jun,
  abstract      = {Transformers have achieved great success in many artificial intelligence fields, such as natural language processing, computer vision, and audio processing. Therefore, it is natural to attract lots of interest from academic and industry researchers. Up to the present, a great variety of Transformer variants (a.k.a. X-formers) have been proposed, however, a systematic and comprehensive literature review on these Transformer variants is still missing. In this survey, we provide a comprehensive review of various X-formers. We first briefly introduce the vanilla Transformer and then propose a new taxonomy of X-formers. Next, we introduce the various X-formers from three perspectives: architectural modification, pre-training, and applications. Finally, we outline some potential directions for future research.},
  archiveprefix = {arXiv},
  eprint        = {2106.04554},
  groups        = {Transformer},
  keywords      = {cs.LG, cs.AI, cs.CL},
  primaryclass  = {cs.LG},
  url           = {http://arxiv.org/pdf/2106.04554},
}
</textarea>