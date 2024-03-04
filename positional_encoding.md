# [详解Transformer中的Positional Encoding](https://blog.csdn.net/qq_40744423/article/details/121930739#:~:text=%2Dpositional%2Dencoding%2F-,%E4%B8%80%E3%80%81%E4%B8%BA%E4%BB%80%E4%B9%88%E8%A6%81%E6%9C%89Positional%20Encoding%EF%BC%9F,Encoding%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81%E2%80%9D%E7%9A%84%E6%A6%82%E5%BF%B5%E3%80%82)

## 为什么要有Positional Encoding？
由于Transformer中`没有循环以及卷积结构`，为了使模型能够`利用序列的顺序`，作者们插入了一些关于tokens在序列中相对或绝对位置的信息。因此，作者们提出了“Positional Encoding位置编码”的概念。

Positional Encoding 和 token embedding 具有`同样的维度`，Positional Encoding 和 token embedding 可以直接相加，结果作为 Encoder 和 Decoder的底部输入。

## 怎么定义Positional Encoding？
现在知道我们需要Positional Encoding，那怎么定义它呢?

### (1)直接编号
假设给定一个长度为 $T$ 的序列，token在序列中的位置记作 $pos$，那么token的位置编码
$$PE = pos = 0, 1, 2, \dots, T-1$$

> [!IMPORTANT]
> 但是这就有个问题，**如果有一段很长的序列**（假如为1000），那么**最后一个token的位置编码非常大**，这是很不合适的：
> 1. 它比第一个token的编码大太多，和token embedding合并以后难免会出现特征在数值上的倾斜；
> 2. 它比一般的token embedding的数值要大，模型可能会把它当作主要信息，对模型可能有一定的干扰。
>
> 那么，`位置编码最好具有一定的值域范围`。

### (2) 对每个位置 $pos$ 作归一化
我们可以使用序列长度 $T$ 对每个位置 $pos$ 作归一化，也就是:
$$
PE = \frac{pos}{T-1}, pos \in \{0, 1, 2, \dots, T-1 \}
$$

上面两种方法都是建立一个长度为 $T$ 的词表，按词表的长度来分配 position encoding，这两个方法都属于表格型。

这样固然使得所有位置编码都落入区间 $[ 0, 1 ]$ ，但是问题也是显著的：

> [!IMPORTANT]
>不同长度序列的位置编码的步长是不同的，在较短序列中相邻的两个token的位置编码的差异，会比长序列中相邻的两个token的位置编码差异更小。如果使用这种方法，那么在长文本中相对次序关系会被“稀释”。

> [!NOTE]
> 总结一下，position encoding的定义要满足下列需求：
> 1. 每个位置有一个唯一的positional encoding；
> 2. 最好具有一定的值域范围，否则它比一般的字嵌入的数值要大，难免会抢了字嵌入的「风头」，对模型可能有一定的干扰；
> 3. 需要体现一定的相对次序关系，并且在一定范围内的编码差异不应该依赖于文本长度，具有一定 translation invariant 平移不变性。

