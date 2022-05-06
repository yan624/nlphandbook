?> 2021.1.14：本文暂时仅为学习记录

> <span class="badge badge-success">图类型</span> <span class="badge badge-warning">训练方法</span> <span class="badge badge-primary">传播步骤</span>

目前，2021 年，图神经网络正处于发展阶段，网络上暂未出现高质量的参考资料。因此个人认为正确的学习方法是 1）在国内外的论坛中查找 GNN 相关的讨论，了解其基本概念；2）查阅几篇综述论文，了解其发展历程以及知识脉络；3）如果对数学方面的知识不够了解，应该及时学习；4）根据自己的研究方向着重查阅相关论文。

经本人多方查找，可以参考[知乎——图神经网络入门](https://zhuanlan.zhihu.com/p/105605774)了解 GNN 的基本概念；可在[清华大学的 github 仓库](https://github.com/thunlp/GNNPapers)找寻所需的论文（首先看综述）。


## GRNN
1

## GCN
图卷积（Graph Convolution）被划分为基于空域以及基于频域两大类[@estrach2014spectral]。值得注意的是，这两种图卷积方式可以重叠[@zhang2020deep]。

本节以 GCN 的发展作为主线，讲述其基础算法。为了保持文章的简洁，在讲述过程中不会提供算法细节的解释或推导，而是在读者可能会产生疑惑地方提出问题。这些问题将在文章的末尾进行统一解答。

> 问题：**什么是傅里叶变换？有什么用？**  
> 问题：**什么是拉普拉斯矩阵？**  
> 问题：**什么是卷积？CNN 中的卷积呢？卷积和傅里叶变换的关系？**

### spectral CNN
卷积是 CNNs 最基本的操作。然而，用于图像或者文本的标准卷积操作无法直接适用于图（graph），这是因为它缺少了网格（grid）结构[@shuman2013emerging]。[@estrach2014spectral]首次提出使用图拉普拉斯（laplacian）矩阵 $L$[@belkin2001laplacian]从频域中对图数据卷积（同时他们也提出了基于空域的方法）。

> 问题：为什么非要对图数据进行卷积操作？为什么要定义图卷积？**为什么不能直接在邻接矩阵上卷积？（邻接矩阵不就是欧几里得数据？）**

图卷积操作 $*_G$ 被定义为如下形式：

$$
x_1 *_G x_2 = U \cdot ((U^T x_1) \odot (U^T x_2)) \tag{1}
$$

其中，$x_1, x_2 \in \mathbb{R}^N$ 是两个定义在结点上的信号，$U$ 是 $L$ 所有的特征向量，$\odot$ 代表哈达玛积（Hadamard product），即向量元素对位相乘。该定义的有效性基于卷积定理（convolution theorem），即两个信号卷积操作的傅里叶变换是其傅里叶变换的元素对位相乘。

> 问题：**为什么图卷积被定义成这种形式（这跟 $L$ 的特征向量有什么关系，甚至是跟 $L$ 有什么关系，这公式怎么这么复杂）？卷积定理在图上有效是否只是一个假设？**

那么，一个信号 $x$ 可以使用以下公式进行过滤：

$$x' = U \Theta U^T x \tag{2}
$$

其中 $x'$ 是输出信号，$\Theta = \Theta(\Lambda) \in \mathbb{R}^{N \times N}$ 是一个可学习的对角矩阵，$\Lambda$ 是 $L$ 所有的特征值。通过对不同的输入输出信号应用不同的过滤器，可以定义一个卷积层，其公式如下所示：

$$x^{l+1}_j = \rho(\Sigma^{f_l}_{i=1} U \Theta^l_{i,j} U^T x^l_i) \quad j = 1, \cdots, f_{l+1} \tag{3}
$$

其中 $l$ 代表某层，$x^l_j \in \mathbb{R}^N$ 代表结点在 $l$ 层的第 $j$ 维隐藏表征（即信号），$\Theta^l_{i,j}$ 是可学习的参数。公式(3)背后的思想与传统的卷积类似：输入信号通过一个可学习的过滤器集合，从而聚合信息，最后再通过一些非线性变换实现特征的提取。将结点特征 $F^V$ 作为输入层，堆叠多层卷积层，其总体架构类似于一个 CNN。

> 问题：(1) 是 tmd 怎么到 (2) 的？？？什么是第 j 维隐藏表征（$x \in \mathbb{R}^N$，N 不是结点数量吗）？

spectral CNN 主要有三大缺陷：1）参数量极大；2）时间复杂度极高（包括特征分解和矩阵相乘）；3）无法局部化。spectral CNN 的卷积核参数为结点数量 $N$，这在数以亿计的社交网络中，是个不可容忍的数字；式(2)需要计算$U \Theta U^T$，至少需要 $O(N^2)$ 的时间复杂度，更不要说特征分解的 $O(N^3)$；观察式(3)可以发现 spectral CNN 在卷积时直接与 $Q$ 相乘，而 $Q$ 来自 $L$，相当于信号 $x$ 在与整个图卷积。事实上，[@estrach2014spectral]在论文中解决了第一个缺陷。

### ChebNet
为了克服 spectral CNN 的效率问题，[@defferrard2016convolutional]设计了与之不同的多项式卷积核，公式如下所示：

$$\Theta(\Lambda) = \Sigma^K_{k=0} \theta_k \Lambda^k \tag{4}
$$

其中 $\theta_0, \cdots, \theta_K$ 是可学习参数，$K$ 是多项式的阶。为了解决矩阵连乘的高额时间复杂度，作者使用 Chebyshev 展开式[@hammond2011wavelets]重写式(4)为以下形式：

$$\Theta(\Lambda) = \Sigma^K_{k=0} \theta_k \mathcal{T}_k(\tilde{\Lambda}) \tag{5}
$$

其中， $\tilde{\Lambda} = 2 \Lambda / \lambda_{max} - I$ 是缩放后的特征值，$\lambda_{max}$ 是最大的特征值，$I \in \mathbb{R}^{N \times N}$ 是单位矩阵，$\mathcal{T}_k(x)$ 是 $k$ 阶 Chebyshev 多项式。利用性质 $L^k=U \Lambda^k U^T$，可以用拉普拉斯矩阵的多项式来充当特征值的多项式，那么式(2)可以被重写为：

$$
\begin{aligned}
	x' = U \Theta(\Lambda) U^T x = & \sum^K_{k=0} \theta_k U \mathcal{T}_k(\tilde{\Lambda}) U^T x \tag{6}\\
	= & \sum^K_{k=0} \theta_k \mathcal{T}_k(\tilde{L}) x = \sum^K_{k=0} \theta_k \bar{x}_k
\end{aligned}
$$

其中 $\bar{x}_k = \mathcal{T}_k(\tilde{L}) x$，$\tilde{L} = 2 L / \lambda_{max} - I$。

> 问题：为什么能克服 spectral CNN 的效率问题？为什么要缩放特征值？

经过巧妙的设计，ChebNet 不再需要拉普拉斯的特征向量以及特征值，移除了时间复杂度极高的特征分解步骤。由于 CheNnet 借助了 Chebyshev 多项式，很容易看出它也可以局部化在 $K$ 步邻近结点之内。

> 问题：为什么 ChebNet 可以实现局部化？

### GCN
通过只使用一阶邻居，[@kipf2016semi]进一步简化了过滤器：

$$H^{l+1} = \rho(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^l \Theta^l)  \tag{7}
$$

其中 $\tilde{A} = A + I$，即增加了自连接。

至此，spatial-based 与 spectral-based 两种方法终于殊途同归。此后的大部分论文都基于 spatial-based 方法。

### 其他spectral-based方法

### Multiple Graphs <!-- {docsify-ignore} -->

### Frameworks <!-- {docsify-ignore} -->

### 数学问题 <!-- {docsify-ignore} -->
- 线性代数
	- [如何计算向量 a 在向量 b 上的投影向量？](https://tieba.baidu.com/p/4202479773)
	- [向量与分量的关系](https://www.zhihu.com/question/278189638/answer/508028323)
	- [如何理解矩阵特征值？](https://www.zhihu.com/question/21874816/answer/19742536)

### 问题汇总

#### 为什么非要对图数据进行卷积操作？
图是一种十分普遍的数据结构。然而由于图结构处于非欧几里得空间（non-Euclidean space）及其不具有网格结构，导致无法使用 RNNs 或者 CNNs 提取其特征。由于许多应用都存在一种图结构，挖掘其潜在价值至关重要，图领域亟需一个能够处理图数据的特征提取器。

图卷积操作的灵感来源于 CV 领域的卷积操作。不过，实际上处理图数据的方法不止这一种。你还可以使用 Graph RNNs、Graph Autoencoders、Graph RL 以及 Graph Adversarial Methods[@zhang2020deep]。

#### 为什么要定义图卷积？
卷积操作不适用于图数据。根据卷积定理：两个信号卷积操作的傅里叶变换是其傅里叶变换的元素对位相乘，因此我们可以先做傅里叶变换再卷积。

#### 为什么图卷积被定义成这种形式？
[@shuman2013emerging]

#### 卷积定理在图上有效是否只是一个假设？
暂时不知。

#### (1)是tmd怎么到(2)的？？？
已知式 (1)：

$$x_1 *_G x_2 = U \cdot ((U^T x_1) \odot (U^T x_2))
$$

将式 (1) 中的向量 $U^T x_1$ 记为可学习的对角矩阵 $\Theta(\Lambda)$，$U^T x_2$ 记为 $U^T x$。可得式 (2)：

$$
\begin{aligned}
	x' & = U \Theta(\Lambda) U^T x \\
	& = U 
	\begin{pmatrix}
		\Theta(\lambda_1) & & \\
		& \ddots & \\
		& & \Theta(\lambda_N) \\
	\end{pmatrix}
	U^T x \\
\end{aligned}
$$

值得注意的是，$(U^T x_1) \odot (U^T x_2)$ 中使用了哈达玛积，而 $\Theta(\Lambda) U^T x$ 中使用的是矩阵乘法。这使得两式是等价的。

那么为什么可以将 $U^T x_1$ 视作一个可学习的参数呢？我们可以暂时将 $x_1$ 看作 CV 领域的卷积核，$x_2$ 为输入的图片（image）。类比到图数据上，在频域中，将 $U^T x_1 = \Theta(\Lambda)$ 视为卷积核，将 $U^T x_2$ 视为输入的图（graph）。由于我们只拥有图数据，而卷积核是未知的，那何不如将 $\Theta(\Lambda)$ 随机初始化，让神经网络更新其中的值，使其达到最佳的卷积效果。$\Theta(\Lambda)$ 就是卷积核。以上就是[@estrach2014spectral]的思想。

#### 什么是第j维隐藏表征？
GCN 与 CNN 和 RNN 不同，它的运算基于图的信号（标量），而非隐藏状态。GCN 每次只计算结点特征 $F$ 中的一个元素/维度（即信号）。请注意公式(3)，$x^l_j \in \mathbb{R}^N$，其维度为结点数量 $N$，而非隐藏状态的维度。简单来说，假设一个结点特征的维度为 768 维，不妨将所有的结点特征视为一个**图**的 768 **种**信号。你可以在脑海中想象由 768 个图堆叠而成的三维结构，每个结点上都有一个矩形，其长度代表信号强弱（即一个图代表一种信号表示）。不过在实际计算中，GCN 与其他神经网络没什么区别，因为可以向量化计算。

#### 为什么能克服 spectral CNN 的效率问题？
将式(4)带入式(2)，得：

$$
x' = U (\Sigma^K_{k=0} \theta_k \Lambda^k) U^T x = U 
\begin{pmatrix}
\Sigma^K_{k=0} \theta_k \lambda^k_1 & & \\
 & \ddots & \\
 & & \Sigma^K_{k=0} \theta_k \lambda^k_N
\end{pmatrix}
U^T x
$$

已知性质 $L^k=U \Lambda^k U^T$，则

$$
\begin{aligned}
x' = & U (\Sigma^K_{k=0} \theta_k \Lambda^k) U^T x \\
= & \Sigma^K_{k=0} \theta_k L^k x
\end{aligned}
$$

由于 spectral CNN 需要 $N$ 个可学习参数，上述公式只需要 $K$ 个，一般而言 $N \gg K$，因此简化了大量的可学习参数，也提升了计算效率。此外，上述公式利用了矩阵的性质，致使神经网络不再需要时间复杂度为 $O(N^3)$ 的特征分解。最后，如前所述，$U \Theta U^T$ 的时间复杂度至少为 $O(N^2)$，虽然上述做法只需计算离散矩阵连乘 $L^k$，时间复杂度为 $O(|E|^2) \ll O(N^2)$，但依旧颇高。传统上讲可以使用多种算法对其优化。不管怎样，[@defferrard2016convolutional]使用 Chebyshev 展开式 $\mathcal{T}_k(x) = 2x\mathcal{T}_{k-1}(x) - \mathcal{T}_{k-2}(x)$ 改写式(4)，其中 $\mathcal{T}_0 = 1, \mathcal{T}_1(x) = x$。可以观察到，我们可以递推地计算出 $\mathcal{T}_k$ 的值，那么整个过滤操作花费 $O(K|E|) \ll O(|E|^2) \ll O(N^2)$。

#### 为什么要缩放特征值？
Chebyshev 多项式的输入要求在 $[-1, 1]$ 之间。由于拉普拉斯矩阵是半正定矩阵，即 $\lambda \ge 0$，则 $\tilde{\Lambda} \in [-1, 1]$。

#### 为什么 ChebNet 可以实现局部化？
[@defferrard2016convolutional]第三页，公式(3)下面的解释。

[邻接矩阵连乘代表什么](https://www.baidu.com/s?wd=邻接矩阵连乘)

## GAE

## Graph RL

## Graph Adversarial Methods

## Readout Operations

## 改进方向
CSDN 中的一篇[文章](https://blog.csdn.net/yyl424525/article/details/100058264)介绍了许多改进的 GNN。

## 参考文献
<bibtex src="基础知识/特征提取器/gnn.bib"></bibtex>
