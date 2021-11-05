## simple RNN
此处介绍 [RNN 的发展历史](https://www.jianshu.com/p/d35a2fd593eb)。暂时略。

李宏毅老师在视频中提到 **[Jordan Network 的性能可能会更好](https://yan624.github.io/posts/5e27260b.html#Elman-Network-amp-Jordan-Network '《吴恩达李宏毅综合学习笔记：RNN入门#Elman Network & Jordan Network》')**。不过现在最常用的还是 Elman Network（即我们熟知的 simple RNN）。

本节首先介绍 RNN 的输入，然后介绍二者输入的区别，最后根据一个简单的例子介绍 BPTT。

### simple RNN 的输入
simple RNN 的结构如下所示，其中一个矩形中包含四个圆的结构可以被称为记忆细胞（memory cell）。实际上将图片往左旋转 90 度，看起来就像 $L$ 层深的前馈神经网络，其中 $L$ 是输入文本的长度。这里的记忆细胞被称为 RNN cell。还有其他的记忆细胞，例如 LSTM cell、GRU cell 等。
![吴恩达深度学习中的RNN示意图](https://blog-content-1256924128.cos.ap-shanghai.myqcloud.com/深度学习算法（二）：simple%20RNN%20推导与理解/吴恩达深度学习中的RNN示意图.jpg ':size=60%')

仔细观察上面的图片可以发现其实 RNN 有两个输入。一个是输入值 $x$，一个是记忆 $a$。下图是 simple RNN 的一个记忆细胞。接下来以多种形式展示 RNN **如何处理 $x$ 和 $a$ 两个输入**。

![simple rnn cell](https://blog-content-1256924128.cos.ap-shanghai.myqcloud.com/深度学习算法（二）：simple%20RNN%20推导与理解/simple%20rnn%20cell.png ':size=60%')

#### 以一个记忆细胞为例
设词向量的维度为 300，一个记忆细胞的 units 为 32，并且进行的是 18 元分类问题。

以**一个**样本为例，分类问题的计算过程可以表示为：
$$
\begin{aligned}
	A_{next} & = tanh(W^{32 \times 300}_x X^{300 \times 1} + W^{32 \times 32}_a A^{32 \times 1}_{\text{prev}} + b^{32 \times 1}_a) \\
	y & = softmax(W^{18 \times 32}_y A_{\text{next}} + b^{18 \times 1}_y)
\end{aligned}
$$

以 **128 个**样本为例：
$$
\begin{aligned}
	A_{next} & = tanh(W^{32 \times 300}_x X^{300 \times 128} + W^{32 \times 32}_a A^{32 \times 128}_{\text{prev}} + b^{32 \times 128}_a)\\
	y & = softmax(W^{18 \times 32}_y * A_{\text{next}} + b^{18 \times 128}_y)
\end{aligned}
$$

#### 以多个记忆细胞为例
计算过程与上面一致，唯一的区别是输入值 $X$。因为有了多个记忆细胞，所以X变成了3维，第3维是 timestep（时间步）。假设有 6 个时间步，伪代码如下：
```
fun rnn_cell(A_prev, Xt, hp):
	do上面的操作

timesteps = 6
X = rand((300, 128, timesteps))
A0 = rand((32, 128))
hp = 初始化所有参数
A = A0
for ts in range(timesteps):
	y, A = rnn_cell(A, X[:, :, ts], hp)
```
其实就是遍历每一个时间步而已。
![RNN](https://blog-content-1256924128.cos.ap-shanghai.myqcloud.com/深度学习算法（二）：simple%20RNN%20推导与理解/RNN.png ':size=60%')


### simple RNN 和 simple NN 的输入对比
首先回顾一下 simple NN 的结构，如下图所示，一个圆圈代表一个神经元。输入值 $x_1, x_2$ 是数字，而不是向量，向量表示为 $[x_1, x_2]$。

![带参数的前馈神经网络模版](https://blog-content-1256924128.cos.ap-shanghai.myqcloud.com/深度学习算法（二）：simple%20RNN%20推导与理解/带参数的前馈神经网络模版.svg ':size=40%')

在经典的线性回归或者逻辑回归中，一项数据就是一个向量。例如在天气预测任务中一条样本可能是 $x = [22.5, 42, 53]$，分别代表温度、湿度和 PM2.5。**因此 $x$ 是一个向量**。

然而在 NLP 领域一项数据是一条语句。由于数学公式无法计算文字，我们通常将每个单词表示为一个唯一的数字（一般表示为 one-hot 编码），然后查找对应的词向量。**也就是说 RNN 的输入 $x$ 是一个矩阵**。

显然 simple NN 很难处理序列数据。如果非要用 simple NN 来做自然语言处理任务，该怎么做？

假定输入“我 是 一名 学生”，将其表示为：
$$
\begin{pmatrix}
	x_1 = \text{我}\\
	x_2 = \text{是}\\
	x_3 = \text{一名}\\
	x_4 = \text{学生}\\
\end{pmatrix}
$$

在训练时，一般使用小批量梯度下降算法，如果再输入一条样本“今天 天气 好像 不错”，那么两条样本就可以表示为：
$$
\begin{pmatrix}
	x_1 = \text{我} & x_5 = \text{今天}\\
	x_2 = \text{是} & x_6 = \text{天气}\\
	x_3 = \text{一名} & x_7 = \text{好像}\\
	x_4 = \text{学生} & x_8 = \text{不错}\\
\end{pmatrix}
$$

**假设词向量已知**，现在的问题是 simple NN 如何处理三维数据 ($B, L, H$)，其中 $B, L, H$ 分别是批次大小，序列长度，隐藏状态维度。有一个折中的方法。将每个词的词向量加起来，再除以词的个数，即：
$$
\frac{(v_{\text{我}} + v_{\text{是}} + v_{\text{一名}} + v_{\text{学生}})}{4} =  v_{\text{我是一名学生}}
$$

其中 $v_{w}$ 代表某个词对应的词向量。将 4 个向量合并成 1 个向量之后，simple NN 就可以处理这种一维数据了。不过这样太勉强了。

总结一下，假设只看一个样本。simple NN 的输入是一个向量 $x = [x_1, x_2, x_3] = [22.5, 42, 53]$，虽然 simple RNN 的输入也是一个向量 $x = [x_1, x_2, x_3, x_4] = [\text{我}, \text{是}, \text{一名}, \text{学生}]$，但是由于文字无法被计算，因此我们使用词向量表示文字，那么输入就转化为 $x = [[0.321, 0.004, \cdots], \cdots]$，它是一个二维的矩阵。

### 反向传播（BPTT）
假设序列长度设为 3，则公式为如下所示，其中 S 代表 sigmoid 函数，下标用于区分不同复合函数，注意 w 和 b 是共享的，没有加额外的下标。为了方便计算，我们选择 sigmoid 作为 RNN 中的激活函数，而非 tanh。实际上二者差别不大，$sigmoid' = s (1 - s)$，$tanh' = 1 - tanh^2$
$$
\begin{aligned}
	a_0 & = 0 \\
	a_1 & = S_1(\underbrace{W_{aa} \cdot a_0 + W_{ax} \cdot x_1 + b}_{f_1}) \\
	a_2 & = S_2(\underbrace{W_{aa} \cdot a_1 + W_{ax} \cdot x_2 + b}_{f_2}) \\
	a_3 & = S_3(\underbrace{W_{aa} \cdot a_2 + W_{ax} \cdot x_3 + b}_{f_3}) \\
\end{aligned}
$$

对 $W_{aa}$ 的求导结果 $\Delta W_{aa}$ 如下所示，其中 $\frac{\partial{a_0}}{\partial{W_{aa}}} = 0$，$\frac{\partial{a_1}}{\partial{W_{aa}}} = 0$，这是因为 $a_0 = 0$（*a 代表 RNN 的记忆，初始记忆是 0*）。此外，为了使式子更清晰，$S_t$ 并没有替换成其输出 $a_t$。
$$
\begin{aligned}
	\frac{\partial{a_3}}{\partial{W_{aa}}} 
	& = \frac{\mathrm{d} a_3}{\mathrm{d} f_3} \frac{\partial{f_3}}{\partial{W_{aa}}} + \frac{\mathrm{d} a_3}{\mathrm{d} f_3} \frac{\partial{f_3}}{\partial{a_2}} \frac{\partial{a_2}}{\partial{W_{aa}}} \\
	& = \frac{\mathrm{d} a_3}{\mathrm{d} f_3} (\frac{\partial{f_3}}{\partial{W_{aa}}} + \frac{\partial{f_3}}{\partial{a_2}} \frac{\partial{a_2}}{\partial{W_{aa}}} ) \\
	& = \underbrace{S_3 \circ (1 - S_3)}_{\frac{\mathrm{d} a_3}{\mathrm{d} f_3} = {S_3}'} \circ 
		\{ \overbrace{a_2}^{\frac{\partial{f_3}}{\partial{W_{aa}}}} + \overbrace{W_{aa}}^{\frac{\partial{f_3}}{\partial{a_2}}} \cdot \underbrace{
				S_2 \circ (1 - S_2)[
					a_1 + W_{aa} \cdot \underbrace{
							S_1 \circ (1 - S_1) \circ (a_0 + W_{aa} \cdot \frac{\partial{a_0}}{\partial{W_{aa}}})
					}_{\frac{\partial{a_1}}{\partial{W_{aa}}}}
				]
			}_{\frac{\partial{a_2}}{\partial{W_{aa}}}}
		\} \\
	\frac{\partial{a_2}}{\partial{W_{aa}}}
	& = \frac{\mathrm{d} a_2}{\mathrm{d} f_2} (\frac{\partial{f_2}}{\partial{W_{aa}}} + \frac{\partial{f_2}}{\partial{a_1}} \frac{\partial{a_1}}{\partial{W_{aa}}} ) \\
	& \cdots
\end{aligned}
$$

实际上，很容易发现上式是一个嵌套表达式，可以写作以下形式，其中 $t$ 代表第几个激活值，也可以理解为第几个时间步。
$$
\begin{aligned}
	\frac{\partial{a_t}}{\partial{W_{aa}}}
	& = \frac{\mathrm{d} a_t}{\mathrm{d} f_t} (\overbrace{\frac{\partial{f_t}}{\partial{W_{aa}}} + \frac{\partial{f_t}}{\partial{a_{t - 1}}} \frac{\partial{a_{t - 1}}}{\partial{W_{aa}}}}^{\frac{\partial{f_t}}{\partial{W_{aa}}}}) \\
	& = S_t \circ (1 - S_t) \circ (a_{t - 1} + W_{aa} \cdot \frac{\partial{a_{t - 1}}}{\partial{W_{aa}}})
\end{aligned}
$$

$\frac{\partial{a_t}}{\partial{W_{ax}}}$ 和 $\frac{\partial{a_t}}{\partial{b}}$ 的求法与上类似。值得注意的是，虽然上标中的 $\frac{\partial{f_t}}{\partial{W_{aa}}}$ 和下面括号中的那个在形式上看起来完全一样，但是它们不是同一个意思。前者指的是将 $f_t$ 看成是 $W_{aa}, W_{ax}$ 的函数，即 $a_{t,\dots,1}$ 是一个复合函数而不是变量；后者指的是将 $f_t$ 看成是 $W_{aa}, W_{ax}, a_t$ 的函数，即 $a_{t,\dots,1}$ 是一个变量而不是复合函数。在求**偏导**时，前者应该先消除 $a_{t,\dots,1}$ 再计算，后者应该将 $a_{t,\dots,1}$ 视为常量。具体可以参考《[多元复合函数求导](https://www.bilibili.com/video/BV1Eb411u7Fw?p=89)》（18:30）。

在上述的公式中，$a_t$ 对 $W_{aa}$ 求偏导实际上使用了链式法则，即 $a_t$ 先对 $f_t$ 求导，然后 $f_t$ 再分别对 $W_{aa}$ 和 $a_{t-1}$ 求偏导。为了方便表示，可以忽视中间的过程。给定以下函数：

$$a_t = S_t(W_{aa} \cdot a_{t-1} + W_{ax} \cdot x_t + b)
$$

$a_t$ 对 $W_{aa}$ 求偏导可以表示为：

$$
\begin{aligned}
	\frac{\partial{a_t}}{\partial{W_{aa}}} & = \frac{\partial{a_t}}{W_{aa}} + \frac{\partial{a_t}}{\partial{a_{t-1}}} \frac{\partial{a_{t-1}}}{W_{aa}} \text{, where} \\
	\frac{\partial{a_t}}{W_{aa}} & = \frac{\mathrm{d} a_t}{\mathrm{d} f_t} \frac{\partial{f_t}}{\partial{W_{aa}}} \\
	\frac{\partial{a_t}}{\partial{a_{t-1}}} & = \frac{\mathrm{d} a_t}{\mathrm{d} f_t} \frac{\partial{f_t}}{\partial{a_{t - 1}}}
\end{aligned}
$$

!> 注意，你可能会发现第一个等式的左右两边都有一个 $\frac{\partial{a_t}}{\partial{W_{aa}}}$，实际上两个符号所代表的意义完全不同。上面也已经说过这个问题了。如果无法理解，可以将右边的 $\frac{\partial{a_t}}{\partial{W_{aa}}}$ 视为 $\frac{\partial{S_t}}{\partial{W_{aa}}}$。而 $\frac{\partial{a_{t-1}}}{W_{aa}}$ 的意义和等式左边 $\frac{\partial{a_t}}{\partial{W_{aa}}}$ 的意义是相同的。

可以发现简化后的公式仍旧是嵌套结构，其余超参数的求导与之类似。在知乎上一些讨论 rnn 梯度消失问题的时候，经常看见以下公式 $\frac{\partial{h_t}}{\partial{h_{t-1}}}$，其中 $h=a$。可能会疑惑这个求导是在干嘛？从上述的推导就可以发现很有用，因为它是求 $\frac{\partial{h_t}}{\partial{W_{aa}}}$ 中最重要的一环。具体可以参考[RNN的分析](#RNN的分析)一节。

### 优化 memory 机制
[RNN 中学习长期依赖的三种机制](https://zhuanlan.zhihu.com/p/34490114)

## GRU

## LSTM
该[博客](https://yan624.github.io/posts/5e27260b.html#长短期记忆——Long-Short-term-Memory-LSTM)中描述了一个 LSTM 的例子，已经把大部分的东西概括了。不过，今天看了别人的代码，第一次见到代码形式的 LSTM，感觉还是有些地方有问题。以下就记录这些问题。

下图是吴恩达深度学习第五周作业中的图片，是一个 LSTM 单元。**与李宏毅老师做的图有略微不同：1）将 input gate 称为 update gate。2）在他所提供的图片中，g(z) 指的是 sigmoid 函数，而这里是 tanh，即下图 update gate 旁边的函数**。3）为了简便，李宏毅老师并没有使用上个时间步的激活值。

![LSTM cell](https://blog-content-1256924128.cos.ap-shanghai.myqcloud.com/zcy/深度学习算法（三）：RNN%20各种机制/LSTM%20cell.jpg ':size=60%')

1. 首先是**输入的问题**。一般来说一个 LSTM 的输入是前一**个** LSTM 的输出值 $a$ 以及输入值 $x$（对于双层 LSTM，其输入值就是前一**层**的输出值）。但是众所周知，**LSTM 每个门的输入肯定只有一个向量，$a$ 和 $x$ 是两个向量，那么如何处理呢？** 
	- 图中使用了 $[a^{<t-1>},x^{<t>}]$ 向量拼接操作。
	- 在我看的代码中直接使用了加法进行相加，代码[在这](https://github.com/Alex-Fabbri/lang2logic-PyTorch/blob/master/seq2seq/atis/lstm/main.py)，但是代码量太大了，随便看看就行了（**2020.2.25 更新**：该代码使用了加法是基于一种较为特殊的情况，即 lstm 的隐藏状态维度等于词向量维度，所以正好可以使用加法，但是还有可能它们的维度不相同，所以**只能使用拼接的方式**）。
2. 之前说过 update gate 就是 input gate，它的输出 $\Gamma^{<t>}_u$ 实际上也是一个向量，而 $\tilde{c^{<t>}}$ 就是输入向量。$\Gamma^{<t>}_u$ 的意思就是限制 $\tilde{c^{<t>}}$ 的信息进入 memory，试想 $\Gamma^{<t>}_u$ 的输出值范围为 (0, 1)，这不就是在说 $\Gamma^{<t>}_u$ 将 $\tilde{c^{<t>}}$ 的每个元素都按其比例进行调整？就类似于将 $\tilde{c^{<t>}}$ 中的信息丢失一部分。如果 $\tilde{c^{<t>}}$ 的输出全是 1，就代表 $\tilde{c^{<t>}}$ 中的信息我全都要。如果 $\tilde{c^{<t>}}$ 的输出全是 0，就代表 $\tilde{c^{<t>}}$ 中的信息我全都不要。
3. 问：由于第一个 LSTM 不存在前一个LSTM，那么它的输入值怎么处理？答：**暂且使用随机初始化，具体还要补充**。
<!-- more -->
4. 记忆单元（下图中的 c，也可以称作 memory(m)）中的数据也可以随机初始化或者直接为 0。
5. 每一层的 LSTM 都权重共享。意思是每一层都有多个 LSTM，里面的权重值其实是同一份。

下图是多个 LSTM 运行的示意图。

![多个 LSTM](https://blog-content-1256924128.cos.ap-shanghai.myqcloud.com/zcy/深度学习算法（三）：RNN%20各种机制/多个%20LSTM.jpg ':size=60%')

下图是 LSTM 的反向传播，被称为 BPTT（backpropagation through time）。

![LSTM反向传播](https://blog-content-1256924128.cos.ap-shanghai.myqcloud.com/zcy/深度学习算法（三）：RNN%20各种机制/LSTM反向传播.jpg ':size=60%')

### LSTM是如何缓解梯度消失的
参考[文章](https://zhuanlan.zhihu.com/p/137427454)。

LSTM 将 $\frac{\partial{a^t}}{\partial{a^i}}$ 分解了。

## RNN的分析
Simple RNN 具有以下优点：1）能够处理序列数据；2）拥有记忆，能够捕捉到数据的先后顺序信息。也就是说RNN 在每个时间步都会把激活值存储起来，然后将其与输入值一起输入下一个时间步，其中激活值被称为记忆（memory）。它的缺点是只有短期依赖。

对于下面两条语句，它们只有复数形式上的不同，但是开头的名词影响到了最后面的 be 动词。simple RNN 无法很好地处理这种长期依赖问题，这可能是由梯度消失或梯度爆炸导致的。
- The cat, which already ate..., was full.
- The cats, which already ate..., were full.

### 梯度消失和梯度爆炸
RNN 所利用的 memory 实际上是上一个时间步的输出，这导致产生了一种复合函数结构。因此在反向传播时需要链式求导，即 $f(g(x)) = f'(g(x))·g'(x)$。这种梯度与梯度的连乘容易造成**梯度消失**和**梯度爆炸**。

具体来说，我们将之前提到的 $\frac{\partial{a_t}}{\partial{W_{aa}}}$ 展开，大致可以得到
 
$$
\begin{aligned}
	\frac{\partial{a_t}}{\partial{W_{aa}}} 
	& = \frac{\partial{a_t}}{W_{aa}} + \frac{\partial{a_t}}{\partial{a_{t-1}}} \frac{\partial{a_{t-1}}}{W_{aa}} \\
	& = \frac{\partial{a_{t}}}{\partial{W_{aa}}} + \sum^t_{i=1} (\prod^{t}_{k=i} \frac{\partial{a_k}}{\partial{a_{k - 1}}}) \cdot \frac{\partial{a_{i-1}}}{\partial{W_{aa}}}\text{,} & \text{where} \ \frac{\partial{a_0}}{W_{aa}} = 0\\
	& = \frac{\mathrm{d} a_{t}}{\mathrm{d} f_{t}} \cdot \frac{\partial{f_{t}}}{\partial{W_{aa}}} + \sum^t_{i=2} (\prod^{t}_{k=i} \frac{\mathrm{d} a_k}{\mathrm{d} f_k} \frac{\partial{f_k}}{\partial{a_{k - 1}}}) \cdot \frac{\mathrm{d} a_{i-1}}{\mathrm{d} f_{i-1}} \cdot \frac{\partial{f_{i-1}}}{\partial{W_{aa}}} \text{,} & \text{expand} \\
	& = {S_t}' \cdot a_{t-1} + \sum^t_{i=2} (\prod^{t}_{k=i} {S_k}' W_{aa}) \cdot {S_{i-1}}' \cdot a_{i-2} \\
	& = {S_t}' \cdot a_{t-1} + \sum^t_{i=2} (W_{aa})^{t-i+1} (\prod^{t}_{k=i-1} {S_k}') \cdot a_{i-2} \text{,} & \text{merge} \\
\end{aligned}
$$

特别地，$a_0 = 0 \text{ or random}$。注意，第二行右端的 $\frac{\partial{a_{i-1}}}{\partial{W_{aa}}}$ 指的是 $\frac{\partial{S_{i-1}}}{\partial{W_{aa}}}$，它等于一个确定的值 ${S_{i-1}}' \cdot a_{i-2}$。

从上式可以轻易地观察到，RNN 的梯度爆炸和梯度消失的问题来自于梯度累乘。**与其他神经网络不同的是，RNN 受到两方面的影响，即激活函数导数的累乘和超参数 $W$ 的累乘。**然而，细心的你可能会发现 $\frac{\partial{a_t}}{\partial{W_{aa}}}$ 展开后它是一个累加的形式。当 $i$ 接近 $t$ 的时候，梯度不会受到累乘的影响，况且最前面还有一项 ${S_t}' \cdot a_{t-1}$。这么说 RNN 其实不会梯度消失（梯度爆炸还是会的）？实际上确实是这样，详见下面的[RNN真正的梯度消失](#RNN真正的梯度消失)一节。

关于梯度爆炸，可以使用简单的梯度裁剪解决。

### RNN真正的梯度消失
之前说到 RNN 不会因为梯度连乘而造成梯度消失，那么其梯度消失究竟是怎么产生的？**实际上，RNN 的梯度消失等价于其无法捕捉到长期依赖**。假设将最后一个时间步的输出作为 xx 分类模型的输入，我们可以计算其损失值并且求得梯度。公式已由上一节给出。

我们将每个 $(W_{aa})^{t-i+1} (\prod^{t}_{k=i-1} {S_k}') \cdot a_{i-2}$ 看作是第 $i \in [1, L]$ 个时间步的特征对第 $t$ 个时间步**总梯度**的贡献，其中 $L$ 是输入序列的总长度。可以发现由于第 $t$ 个时间步不会受到 $W$ 连乘的影响，因此其贡献最大，主导了总梯度。随着时间回溯，每个时间步的贡献越来越少。换言之，第 $t$ 个时间步没有得到远处语句片段的贡献，即其无法捕获长期依赖。在反向传播时，$\frac{\partial{a_t}}{\partial{W_{aa}}}$ 会更新 $W_{aa}$，由于其没有捕获到长期依赖，因此模型也无法捕获到长序列的信息。

综上所示，由于 RNN 梯度累乘导致的梯度消失，使其不具有捕获长期依赖的能力。改进方向有多种，例如优化激活函数使其导数不过小也不过大（relu）、优化算法使权重不会连乘等。

---

下面是看了知乎某篇回答写下的。大概是一年前写的，比较乱。该回答的思想与我的思想类似，不同的是他从损失值累加的角度进行解释。而我上面给出的例子是比较常见的例子，即一个普通的分类问题。

虽然链式求导导致的梯度相乘大致已经可以解释梯度消失的问题，但是如果仔细想想就会发现盲点。

在此之前，我想先说明 RNN 家族的反向传播路径与其他的神经网络不同，它的 loss 值是每一个 timestep 的真实值 y 与输出值 的 loss 之和。（[此视频](https://mooc.study.163.com/learn/2001280005?tid=2001391038&_trace_c_p_k2_=72573d316c3441869416d70899cdf382#/learn/content?type=detail&id=2001770031) 大致讲明白了这个总 loss 值到底由哪些 loss 相加得到）

知道了上面的前提条件，就可以很简单的理解这个盲点了（**参考资料 1 大致解释了这一问题**，这一段可能比较绕）。**在反向传播时，后面的时间步（比如下图中 $loss_4$）会出现梯度消失的现象（注意 RNN 每个 timestep 的 W 都是一样），这是因为在求梯度时，函数已经复合了好几层**。而对 $loss_1$ 求 W 的导数时，由于它本身就在序列的前面，函数还没有复合，还不会出现梯度消失。根据之前所说的前提，**在计算总 loss 时，是将各个阶段的梯度加起来**，即使后面的 loss 会得到一个很小的的梯度 $\Delta W$，但由于 $loss_1$ 的原因，并不会发生梯度消失！

然而，事实上会发生，那么梯度消失从何而来呢？这是因为在求序列前几个单词的梯度时，你需要从 $loss_4$ 开始计算（*当然其他的 $loss_3$ 也要计算，但是原理是一样*），由于 $loss_4$ 中复合了好几层函数，导致诸如 $x_1$所对应的 RNN 的梯度很小，从而产生了**信息丢失**。信息丢失就是 RNN 的梯度消失。

你可能会想这不还是梯度连乘导致的？确实，但是有一点需要考虑，RNN 在反向传播时，是需要传播到输入值 x 的，即词向量。而在计算梯度时，$x_4$ 所对应 RNN 肯定拥有不是很小的梯度 $\Delta W$，这是由于此时它还没有嵌套函数，所以信息无问题。但是当反向传播到 $x_1$ 时，梯度已经很小了，由于小梯度导致 $x_1$ 无法得到很好的更新，于是产生了信息丢失，也就是说长期记忆没有回传给 $x_1$。

$loss_3$ 以及 $loss_2$ 以此类推，不过对于 $loss_1$ 并无问题，因为它没有复合函数。

注：**上两段，还参考了参考资料 3，个人认为参考资料 1 中内容并不是很完整**。

![total loss](https://blog-content-1256924128.cos.ap-shanghai.myqcloud.com/zcy/深度学习500问笔记/total%20loss.jpg)

而在 LSTM 中，是使用加和的计算方式（博主注：**由于我没有计算过，所以我也不是很肯定**），所以大致解决了梯度消失的问题。

### 梯度裁剪
理论上，梯度爆炸也同样糟糕。但在实践上，其实我们可以直接砍一刀（原话：it turns out we can actually have a hack），这由 [Thomas Mikolov](http://proceedings.mlr.press/v28/pascanu13.pdf) 首次提出。在某种程度上是不精确的，比如说“**现在你有一个很大的梯度 100，让我们把它限制在 5 吧**”。这方法就结束了。虽然不是一个数学方法，但结果表明在实践中效果不错。

如下图所示，其中有一个像峭壁一样的东西，它就是罪魁祸首。当我们将一个小球往前移动时，有时候正好迈过峭壁，小球得以正常移动。但是当小球碰到峭壁时，小球就会被反弹回去，导致 loss 发生剧烈变化。
![the error surface is rough](https://blog-content-1256924128.cos.ap-shanghai.myqcloud.com/zcy/深度学习500问笔记/the%20error%20surface%20is%20rough.jpg)

从数学角度来看，那个峭壁就是梯度。**需要注意的是**，由于图中 z 轴标注的是 total loss，所以第一印象感觉峭壁代表 total loss，但是**峭壁代表的是梯度**。根据参数更新公式 $w -= \alpha * \Delta w$，其中 $\Delta w$ 代表 w 的梯度，所以 w 的更新方向其实与梯度直接相关。**当 w 不幸到达某个值时，遇到梯度极大的情况，那么不管梯度是正还是负，都会将 w 更新到一个相对很大的值，从而 loss 值也会跟着改变。注：这里其实也与 learning rate 有关，因为原本的梯度都很小，所以我们初始设置的 lr 都很大。突然梯度增大，而 lr 没有适应，一个大的梯度乘上一个大的 lr，那就更大了**。

## 参考文献
- GRU
	1. [人人都能看懂的GRU](https://zhuanlan.zhihu.com/p/32481747)
	2. *[吴恩达李宏毅综合学习笔记：RNN入门#GRU单元——Gate Recurrent Unit](https://yan624.github.io/posts/5e27260b.html#GRU单元——Gate-Recurrent-Unit)*
- LSTM
	1. [文章](https://zhuanlan.zhihu.com/p/137427454)
- simple RNN 的分析
	1. [LSTM如何解决梯度消失问题](https://zhuanlan.zhihu.com/p/28749444)
	2. [深入理解lstm及其变种gru](https://zhuanlan.zhihu.com/p/34203833)
	3. [LSTM如何来避免梯度弥散和梯度爆炸？](https://www.zhihu.com/question/34878706/answer/665429718)
	4. [人人都能看懂的LSTM介绍及反向传播算法推导（非常详细）](https://zhuanlan.zhihu.com/p/83496936)
	5. [改善深层神经网络：超参数调试、正则化以及优化](https://mooc.study.163.com/learn/2001281003?tid=2001391036&_trace_c_p_k2_=1f1aedd0dd9f431da0ce64963f010916#/learn/content?type=detail&id=2001702118&cid=2001699114)或[Tips for Deep Learning](https://www.bilibili.com/video/av10590361/?p=18)（08:37）
	6. [Recurrent Neural Network (Part II)](https://www.bilibili.com/video/av10590361/?p=37)，13:50~18 左右
	7. [【高等数学】多元复合函数求导的基本方法](https://zhuanlan.zhihu.com/p/61585348)
	8. [《高等数学》同济版 全程教学视频（宋浩老师）](https://www.bilibili.com/video/BV1Eb411u7Fw?p=89)，17:34 开始