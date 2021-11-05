建议翻阅以下文献，建议按顺序阅读。

1. [CRF Layer on the Top of BiLSTM - 1](https://createmomo.github.io/2017/09/12/CRF_Layer_on_the_Top_of_BiLSTM_1/)
2. 《统计学习方法	》
3. [CRF 条件随机场的原理、例子、公式推导和应用](https://zhuanlan.zhihu.com/p/148813079)（图解）
4. [机器学习-白板推导系列(十七)-条件随机场CRF（Conditional Random Field）](https://www.bilibili.com/video/BV19t411R7QU?p=5)
5. *[An Introduction to Conditional Random Fields](https://arxiv.org/abs/1011.4088)*
6. 补充
	1. [二维随机变量的边缘分布是什么意思（前向后向算法相关）](https://www.bilibili.com/video/BV14b411M7Jn?p=7)
	2. [贝叶斯公式和全概率公式](https://www.bilibili.com/video/BV1a4411B7B4/)

预测：

1. [关于 LogSumExp](https://zhuanlan.zhihu.com/p/153535799)
2. [viterbi 算法](https://www.zhihu.com/question/20136144)

实战：

1. [pytorch 官方提供的简易例子](https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html#advanced-making-dynamic-decisions-and-the-bi-lstm-crf)
2. [pytorch-crf 实现](https://github.com/kmkurn/pytorch-crf)

以下是笔记、公式仅供个人查阅。

## 概率无向图模型
设有联合概率分布 $P(Y)$，由无向图 $G=(V,E)$ 表示，在图 $G$ 中，结点表示**随机变量**，边表示随机变量之间的依赖关系。如果联合概率分布 $P(Y)$ 满足**成对、局部或全局马尔可夫性**，就称此联合概率分布为概率无向图模型（probabilistic undircted graphical model），或马尔可夫随机场（Markov random field）。

### 概率无向图模型的因子分解
首先给出无向图中团与最大团的定义。

无向图 $G$ 中任何两个结点均有边连接的结点子集称为**团**（clique）。若 $C$ 是无向图 $G$ 的一个团，并且不能再加紧任何一个 $G$ 的结点使其成为一个更大的团，则称此 $C$ 为**最大团**（maximal clique）。

将概率无向图模型的联合概率分布表示为其最大团上的随机变量的函数的乘积形式的操作，称为概率无向图模型的因子分解（factorization）。

**Hammersley-Clifford 定理**（《统计学习方法》p218）

概率无向图模型 的联合概率分布 $P(Y)$ 可以表示为如下形式：

$$P(Y) = \frac{1}{Z} \prod_C \Psi_C (Y_C) \tag{1} 
$$

其中，$C$ 是无向图的最大团，$Y_C$ 是 $C$ 的结点对应的随机变量，$\Psi_C (Y_C)$ 称为势函数（potential function），是 $C$ 上定义的严格正函数，乘积是在无向图所有的最大团上进行的。$Z$ 是规范化因子（normalization factor）

$$Z = \sum_Y \prod_C \Psi_C (Y_C) \tag{2}
$$

规范化因子保证 $P(Y)$ 构成一个概率分布。通常 $\Psi_C (Y_C)$ 被定义为指数函数：

$$\Psi_C (Y_C) = exp\{-E(Y_C)\} \tag{3}
$$

## 条件随机场的定义与形式
条件随机场（conditional random field）是给定随机变量 $X$ 的条件下，随机变量 $Y$ 的马尔可夫随机场。以下主要介绍定义在线性链上的特殊的条件随机场，称为线性链条件随机场（linear chain conditional random field）。

**定理**：线性链条件随机场的参数化形式（《统计学习方法》p220）

设 $P(Y|X)$ 为线性链条件随机场，则在随机变量 $X$ 取值为 $x$ 的条件下，随机变量 $Y$ 取值为 $y$ 的条件概率具有如下形式（博主注：也就是说以下形式只是其中一种情况 $X=x, Y=y$，事实上 $Z(x)$ 代表了所有可能）：

$$P(y|x) = \frac{1}{Z(x)} exp\{\sum_{i,k} \lambda_k t_k(y_{i-1}, y_i, x, i) + \sum_{i,l} \mu_l s_l(y_i, x, i)\} \tag{4}
$$

其中，

$$Z(X) = \sum_y exp\{\sum_{i,k} \lambda_k t_k(y_{i-1}, y_i, x, i) + \sum_{i,l} \mu_l s_l(y_i, x, i)\} \tag{5}
$$

式中， $t_k$ 和 $s_l$ 是特征函数，$\lambda_k$ 和 $\mu_l$ 是对应的权值。$Z(x)$ 是规范化因子，求和是在所有可能的输出序列上进行的。（博主注：这里的求和应该指的是 $\sum_y$）具体来说，$t_k$ 是定义在边上的特征函数，称为转移特征，依赖于当前和前一个位置；$s_l$ 是定义在结点上的特征函数，称为状态特征，依赖于当前位置。

## 条件随机场的简化形式
注意到条件随机场式 (4) 中**同一特征**在各个位置都有定义，可以对同一个特征在各个位置求和，将**局部特征函数**转化为一个**全局特征函数**，这样就可以将 条件随机场写成**权值向量和特征向量的内积形式**，即条件随机场的简化形式。

为方便起见，将转移特征和状态特征及其权值用统一的符号表示。设有 $K_1$ 个转移特征，$K_2$ 个状态特征，$K = K_1 + K_2$，记

$$
f_k(y_{i-1}, y_i, x, i) =
\begin{cases}
	t_k(y_{i-1}, y_i, x, i), & k = 1, 2, \cdots, K_1 \tag{6} \\
	s_l(y_i, x, i),          & k = K_1 + l; l = 1, 2, \cdots, K_2
\end{cases}
$$

然后，对转移与状态特征在各个位置 $i$ 求和，记作

$$f_k(y, x) = \sum^n_{i=1} f_k(y_{i-1}, y_i, x, i), \quad k = 1, 2, \cdots, K \tag{7}
$$

用 $w_k$ 表示特征 $f_k(y, x)$ 的权值，即

$$
w_k = 
\begin{cases}
	\lambda_k, & k = 1, 2, \cdots, K_1 \tag{8} \\
	\mu_l,     & k = K_1 + l; l = 1, 2, \cdots, K_2
\end{cases}
$$

于是，条件随机场 (4)~(5) 可表示为

$$
P(y|x) = \frac{1}{Z(x)} exp\{\sum^K_{k=1} w_k f_k(y, x)\} \tag{9}
$$
$$Z(x) = \sum_y exp\{\sum^K_{k=1} w_k f_k(y, x)\} \tag{10}
$$

经过简化之后，可以发现 $exp()$ 函数中的值是累加的，可以表示为向量的内积形式。若以 $w$ 表示权值向量，即

$$w = (w_1, w_2, \cdots, w_K)^T \tag{11}
$$

以 $F(y, x)$ 表示全局特征向量，即

$$F(y, x) = (f_1(y, x), f_2(y, x), \cdots, f_K(y, x))^T \tag{12}
$$

则条件随机场可以写成向量 $w$ 与 $F(y, x)$ 的内积形式：

$$P_w(y|x) = \frac{exp\{w \cdot F(y, x)\}}{Z_w(x)} \tag{13}
$$

其中，

$$Z_w(x) = \sum_y exp\{w \cdot F(y, x)\} \tag{14}
$$

## 条件随机场的矩阵形式
条件随机场还可以由矩阵表示。

对观测序列 $x$ 的每一个位置 $i = 1, 2, \cdots, n+1$，由于 $y_{i-1}$ 和 $y_i$ 在 $m$ 个状态（标记）中取值，可以定义一个 $m$ 阶矩阵随机变量

$$
M_i(x) =
\begin{bmatrix}
	M_i(y_{i-1}, y_i | x) \tag{15}
\end{bmatrix}
$$

矩阵随机变量的元素为

$$M_i(y_{i-1}, y_i | x) = exp\{W_i(y_{i-1}, y_i | x)\} \tag{16}
$$

$$W_i(y_{i-1}, y_i | x) = \sum^K_{k=1} w_k f_k(y_{i-1}, y_i, x, i) \tag{17}
$$

这里 $w_k$ 和 $f_k$ 分别由式 (8) 和式 (6) 给出，$y_{i-1}$ 和 $y_i$ 是状态（标记）随机变量 $Y_{i-1}$ 和 $Y_i$ 的取值。

这样，给定观测序列 $x$，相应的标记序列 $y$ 的非规范化概率可以通过该序列 $n+1$ 个矩阵的**适当元素**的乘积 $\prod^{n+1}_{i=1} M_i(y_{i-1}, y_i | x)$ 表示。注意，由于引进了 $y_0=\text{start}, y_{n+1} = \text{stop}$，因此是 $n+1$ 个矩阵。于是，条件概率 $P_w(y|x)$ 是

$$P_w(y|x) = \frac{1}{Z_w(x)} \prod^{n+1}_{i=1} M_i(y_{i-1}, y_i | x) \tag{18}
$$

其中，$Z_w(x)$ 是规范化因子，是 $n+1$ 个矩阵的乘积的 $(\text{start}, \text{stop})$ 元素，即

$$
Z_w(x) = 
\begin{bmatrix}
	M_1(x) M_2(x) \cdots M_{n+1}(x) \tag{19}
\end{bmatrix}_{\text{start}, \text{stop}}
$$

注意，$y_0=\text{start}, y_{n+1} = \text{stop}$ 分别表示开始状态和终止状态，规范化因子 $Z_w(x)$ 是以 $\text{start}$ 为起点 $\text{stop}$ 为终点，通过状态的**所有路径** $y_1 y_2 \cdots y_n$ 的非规范化概率 $\prod^{n+1}_{i=1} M_i(y_{i-1}, y_i | x)$ 之和。简单来说，就是所有的可能路径的概率。

*2021.08.29：没看懂 $Z_w(x)$ 这个矩阵的下标 $\text{start}, \text{stop}$ 是什么意思*

2021.08.30：懂了。$\begin{bmatrix} M_1(x) M_2(x) \cdots M_{n+1}(x) \end{bmatrix}$ 的结果是一个稀疏矩阵，只有第一行第一列有值，其余位置均为 0。自己手算一下《统计学习方法》p223 的例子就知道了。第一行第一列位置的值恰好就是从 $\text{start}$ 到 $\text{stop}$ 的所有路径的非规范化概率之和，即规范化因子 $Z_w(x)$。

随机变量矩阵 $M_i$ 中的元素由式 (16) 和式 (17) 给出，由于存在指数运算，矩阵相乘等价于指数上的相加。以《统计学习方法》p223为例，矩阵乘积的第一行第一列为

$$a_{01}b_{11}c_{11} + a_{02}b_{21}c_{11} + a_{01}b_{12}c_{21} + a_{02}b_{22}c_{22} + \cdots
$$

其中 $a_{01}b_{11}c_{11}$ 等元素代表一条路径的概率 $exp\{\sum^K_{k=1} w_kf_k(y, x)\} = exp\{w \cdot F(y, x)\}$。所有元素之和代表所有路径的非规范化概率之和。

## 条件随机场各个参数的意义
上式中有太多变量，稍微解释一下是什么意思，以下基于自己的理解。$i$ 代表输入长度维度，$k$ 和 $l$ 分别代表转移特征维度和状态特征维度，$m$ 代表状态数量维度。对标深度学习，$i$ 可以理解为输入长度，$m$ 可以理解为标签数量。$k, l$ 比较难理解，它们分别是转移函数和状态函数的数量，其中 $K \le M^2, L \le M$。

以下图为例，输入序列 $x = [w_1, w_2, w_3]$，输出序列 $y = [y_1, y_2, y_3]$，并且有两个状态 $\{\text{O}, \text{B\_person}\}$，分别用 $\{1, 2\}$ 表示，另外引入起始状态 $\text{stop}$ 和终止状态 $\text{stop}$。如下图所示，$t_k$ 是定义在边上的特征函数，$s_l$ 是定义在结点上的特征函数。最上面的状态路径就是 $y_i$ 所有可能的取值的路径。

![CRF 参数含义解释](/img/CRF参数含义解释.jpg)




