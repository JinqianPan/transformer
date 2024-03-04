# Transformer中的Positional Encoding
这篇 Readme 在 [原文](https://blog.csdn.net/qq_40744423/article/details/121930739#:~:text=%2Dpositional%2Dencoding%2F-,%E4%B8%80%E3%80%81%E4%B8%BA%E4%BB%80%E4%B9%88%E8%A6%81%E6%9C%89Positional%20Encoding%EF%BC%9F,Encoding%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81%E2%80%9D%E7%9A%84%E6%A6%82%E5%BF%B5%E3%80%82) 的基础上加入了我自己的理解。

## 为什么要有Positional Encoding？
由于Transformer中`没有循环以及卷积结构`，为了让模型能够利用`时序`，作者们插入了一些关于 tokens 在序列中相对或绝对位置的信息。因此，作者们提出了“Positional Encoding位置编码”的概念。

Positional encoding 和 words embedding 具有`同样的维度`，positional encoding 和 words embedding 可以直接相加，结果作为 Encoder 和 Decoder 的底部输入。

## 怎么定义Positional Encoding呢？
这一部分，[原文](https://blog.csdn.net/qq_40744423/article/details/121930739#:~:text=%2Dpositional%2Dencoding%2F-,%E4%B8%80%E3%80%81%E4%B8%BA%E4%BB%80%E4%B9%88%E8%A6%81%E6%9C%89Positional%20Encoding%EF%BC%9F,Encoding%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81%E2%80%9D%E7%9A%84%E6%A6%82%E5%BF%B5%E3%80%82) 给出了几种可以使用的 positional encoding，并且给出了他们的优缺点，从而解释为什么论文中作者会给出三角函数型的 positional encoding 公式。

### 方式1: 表格型--直接编号
Assume 给定一个长度为 $T$ 的序列，token在序列中的位置记作 $pos$，那么 token 的位置编码
$$PE = pos = 0, 1, 2, \dots, T-1$$

> [!IMPORTANT]
> 但是这就有个问题，**如果有一段很长的序列**（假如为1000），那么**最后一个token的位置编码就是1000**，这就会产生 bias：
>
> 前后位置编码相差巨大会导致，出现特征在数值上的倾斜，从而对模型产生干扰。
>
> 那么，`这就需要位置编码最好有一定的取值范围`。

### 方式2：表格型--对每个位置 $pos$ 作归一化
$$PE = \frac{pos}{T-1}, pos \in \{0, 1, 2, \dots, T-1 \}$$

这样使得所有位置编码都落入区间 $[ 0, 1 ]$，但是问题也是显著的：

> [!IMPORTANT]
>不同长度序列的位置编码的步长是不同的，`在较短序列中相邻的两个token的位置编码的差异，会比长序列中相邻的两个token的位置编码差异更小`。如果使用这种方法，那么在长文本中相对次序关系会被“稀释”。

> [!NOTE]
> 所以，position encoding 的定义要满足下列需求：
> 1. 每个位置有一个唯一的 positional encoding；
> 2. 最好具有一定的值域范围；
> 3. 需要体现一定的相对次序关系，并且在一定范围内的编码差异不应该依赖于文本长度，具有一定 translation invariant 平移不变性。

