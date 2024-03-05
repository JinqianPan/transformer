# Transformer中的Positional Encoding
这篇笔记1️以 [详解Transformer中的Positional Encoding](https://blog.csdn.net/qq_40744423/article/details/121930739#:~:text=%2Dpositional%2Dencoding%2F-,%E4%B8%80%E3%80%81%E4%B8%BA%E4%BB%80%E4%B9%88%E8%A6%81%E6%9C%89Positional%20Encoding%EF%BC%9F,Encoding%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81%E2%80%9D%E7%9A%84%E6%A6%82%E5%BF%B5%E3%80%82) 和 [Transformer学习笔记一：Positional Encoding（位置编码）](https://zhuanlan.zhihu.com/p/454482273) 为基础。

## 什么是 Positional Encoding
Positional Encoding 为`没有循环以及卷积结构`的 transformer 提供 self-attention 能够利用`位置信息`。

> [!NOTE]
> Positional encoding 和 input embedding 具有`同样的维度`，这样使得 positional encoding 和 input embedding 可以直接相加，结果作为 Encoder 和 Decoder 的底部输入。

## 怎么定义Positional Encoding呢
这一部分，两篇文章给出了几种可以使用的 positional encoding，并且给出了他们的优缺点，从而解释为什么论文中作者会给出三角函数型的 positional encoding 公式。

### (1) 单调递增--直接编号
Assume 给定一个长度为 $T$ 的序列，token在序列中的位置记作 $pos$，那么 token 的位置编码
$$PE = pos = 0, 1, 2, \dots, T-1$$

> [!IMPORTANT]
> 但是这就有个问题，**如果有一段很长的序列**（假如为1000），那么**最后一个token的位置编码就是1000**，这就会产生 bias：
>
> 前后位置编码相差巨大会导致，出现特征在数值上的倾斜，从而对模型产生干扰。
>
> 那么，`这就需要位置编码最好有一定的取值范围`。

### (2) 单调递增进阶--对每个位置 $pos$ 作归一化
$$PE = \frac{pos}{T-1}, pos \in \{0, 1, 2, \dots, T-1 \}$$

这样使得所有位置编码都落入区间 $[ 0, 1 ]$，但是问题也是显著的：

> [!IMPORTANT]
> 不同长度序列的位置编码的步长是不同的，`在较短序列中相邻的两个token的位置编码的差异，会比长序列中相邻的两个token的位置编码差异更小`。如果使用这种方法，那么在长文本中相对次序关系会被“稀释”。

> [!NOTE]
> 所以，position encoding 的定义要满足下列需求：
> 1. 每个位置有一个唯一的 positional encoding；
> 2. 最好具有一定的值域范围；
> 3. 在序列长度不同的情况下，不同序列中token的相对位置/距离也要保持一致

### (3) 将单一维度的位置信息扩展到多维度
考虑到位置信息作用在input embedding上，因此比起用单一的值，更好的方案是用一个和 input embedding 维度一样的向量来表示位置。
这时我们就很容易想到二进制编码。如下图，假设d_model = 3，那么我们的位置向量可以表示成：

<p align="center">
  <img src="./img/009.png" width="700">
</p>

这下所有的值都是有界的（位于0，1之间），且 transformer 中的 $d_{model}$ 本来就足够大，基本可以把我们要的每一个位置都编码出来了。

> [!IMPORTANT]
> 但是这种编码方式也存在问题：这样编码出来的位置向量，处在一个离散的空间中，不同位置间的变化是不连续的。

### (4) 在多维度的基础上使用有界的周期性函数
既然要需要解决 (3) 中空间离散的问题，我们可以使用有界又连续的函数来代替0和1。
三角函数就可以满足这一点。

Assume $\omega_i = \frac{1}{2^i}, i \in {0, 1, ..., d_{model}-1}$
```math
PE(pos) = \left[
\begin{array}{c}
\sin(\omega_0 \cdot pos) \\
\sin(\omega_1 \cdot pos) \\
\sin(\omega_2 \cdot pos) \\
\sin(\omega_3 \cdot pos) \\
\vdots \\
\sin(\omega_4 \cdot pos ) \\
\sin(\omega_5 \cdot pos ) \\
\end{array}
\right]_{d_{model} \times 1}
```

结合下图，来理解一下这样设计的含义。
图中每一行表示一个 $PE(pos)$，每一列表示 $PE(pos)$ 中的第 $i$ 个元素。旋钮用于调整精度，越往右边的旋钮，需要调整的精度越大，因此指针移动的步伐越小。
每一排的旋钮都在上一排的基础上进行调整（函数中t的作用）。
通过频率 $\frac{1}{2^i}$ 来控制 $sin$ 函数的波长，频率不断减小，则波长不断变大，此时 $sin$ 函数对 $t$ 的变动越不敏感，以此来达到越向右的旋钮，指针移动步伐越小的目的。 这也类似于二进制编码，每一位上都是0和1的交互，越往低位走（越往左边走），交互的频率越慢。

<p align="center">
  <img src="./img/010.png" width="300">
  <img src="./img/011.png" width="106">
</p>

在前面的两种方法中，我们为了体现某个字在句子中的绝对位置，使用了一个单调的函数，使得任意后续的字符的位置编码都大于前面的字。
如果我们放弃对绝对位置的追求，转而要求位置编码仅仅关注一定范围内的相对次序关系，那么使用一个sin/cos函数就是很好的选择，因为sin/cos函数的周期变化规律非常稳定，所以编码具有一定的平移不变性。

$$PE(pos) = \sin(\omega \cdot pos)$$

其中，$\omega$ 越小，波长越长，即相邻的 token 的位置编码之间的差异越小。

> [!IMPORTANT]
> 但这样也存在一些问题：
> 1. 如果 $\omega$ 比较大，相邻 token 之间的位置差异不明显；
> 2. 如果 $\omega$ 比较小，在长序列中可能会有一些不同位置的token的位置编码一样，这是因为PE的值域 $[-1, 1]$ 的表现范围有限。

## 论文的选择
所以作者并没有使用单一维度的三角函数，而是将维度拓展到 512维。使用512的理由是 word embedding 的维度是 512，这样方便让 positional embedding 和 word embeeding 的结果相加。

$$PS(pos, 2i) = \sin (\frac{1}{1000^{\frac{2i}{d_{model}}}} \cdot pos)$$
$$PS(pos, 2i+1) = \cos (\frac{1}{1000^{\frac{2i}{d_{model}}}} \cdot pos)$$

where $pos$ is the position, $i$ is the dimension, and $d_{model} = 512$.

将公式给拆开，我们就会得到：
```math
PE(pos) = \left[
\begin{array}{c}
\sin(\omega_0 \cdot pos) \\
\cos(\omega_0 \cdot pos) \\
\sin(\omega_1 \cdot pos) \\
\cos(\omega_1 \cdot pos) \\
\vdots \\
\sin\left(\omega_{\frac{d_{model}}{2}-1} \cdot pos\right) \\
\cos\left(\omega_{\frac{d_{model}}{2}-1} \cdot pos\right) \\
\end{array}
\right]_{d_{model} \times 1}
= \left[
\begin{array}{c}
\sin\left(\frac{pos}{10000^{2 \times \frac{0}{512}}}\right) \\
\cos\left(\frac{pos}{10000^{2 \times \frac{0}{512}}}\right) \\
\vdots \\
\sin\left(\frac{pos}{10000^{2 \times \frac{255}{512}}}\right) \\
\cos\left(\frac{pos}{10000^{2 \times \frac{255}{512}}}\right) \\
\end{array}
\right]_{d_{model} \times 1}
\approx
\left[
\begin{array}{c}
\sin(pos) \\
\cos(pos) \\
\vdots \\
\sin(0.0010 \cdot pos) \\
\cos(0.0010 \cdot pos) \\
\end{array}
\right]_{d_{\text{model}} \times 1}
```

不过这样定义positional encoding，仍会陷入循环, 这里人为地将最大不重复序列长度限制为 512。例如，在 BERT 中就是这样做的（尽管值得一提的是他们使用了学习位置嵌入，但那是另一回事了）。 如果不这样做，模型确实无法区分序列中的第一个token和第513个token的位置编码。

## Coding
```Python
import numpy as np
import seaborn as sns

PE = np.zeros([512, 512])
d_model = 512

for pos in range(d_model):
    for i in range(int(d_model / 2)):
        PE[pos][2 * i] = np.sin(pos / (1000 ** (2 * 2 * i / d_model)))
        PE[pos][2 * i + 1] = np.cos(pos / (1000 ** (2*(2 * i + 1) / d_model)))

sns.heatmap(data=PE,vmin=-1,vmax=1)
```

<p align="center">
  <img src="./img/008.png" width="700">
</p>

可以发现，由于sin/cos函数的性质，位置向量的每一个值都位于 $[-1, 1]$ 之间。
同时，纵向来看，图的右半边几乎都是红色的，这是因为越往后的位置，频率越小，波长越长，所以不同的t对最终的结果影响不大。
而越往左边走，颜色交替的频率越频繁。