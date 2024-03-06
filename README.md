Edit by `Mar. 3, 2024`

# Transformer

## 一些小小的理论知识
> A transformer is a deep learning architecture based on the multi-head attention mechanism, proposed in a 2017 paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762). -- Wiki

### High level review
从 high level 层面上来看，transformer其实就是一个 `black box`。而这个黑匣子由 `Encoders` 和 `Decoders` 两个部分组成。

<p align="center">
  <img src="./img/001.png" width="700">
</p>

`Encoders` 和 `Decoders` 其实是由多个个数相同的 encoders 和 decoders 组成的。论文里使用了6个，但这是超参，可以进行调试。

<p align="center">
  <img src="./img/003.png" width="700">
</p>

每一个 encoder 都有`相同的结构`，但他们`不 share weights`。Decoders 也一样。

### 过程

<p align="center">
  <img src="./img/004.png" width="300">
</p>

#### Input
在 transformer中，如上图所示，有两个 input part。而这两个 input part 由 **input embedding (word embedding)** 和 **positional encoding** 组成。
1. Input embedding: 比较常见的input embedding 有 One-hot encoding 和 Word embedding 两种。Input embedding 的本质是将输入输入成一组数字。
    * One-hot encoding
    * Word embedding 在嵌入空间中生成**语义上相似 和 相关单词的类似位置**。

<p align="center">
  <img src="./img/005.png" width="300">
  <img src="./img/006.png" width="300">
</p>

2. Positional encoding 为`没有循环以及卷积结构`的 transformer 提供 self-attention 能够利用`位置信息`。(详见[positional_encoding.md](positional_encoding.md))

<p align="center">
  <img src="./img/007.png" width="700">
</p>

#### Encoder side


## Coding

## Reference:
1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
2. [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
3. [Transformer模型详解（图解最完整版）](https://zhuanlan.zhihu.com/p/338817680)
4. [详解Transformer中的Positional Encoding](https://blog.csdn.net/qq_40744423/article/details/121930739#:~:text=%2Dpositional%2Dencoding%2F-,%E4%B8%80%E3%80%81%E4%B8%BA%E4%BB%80%E4%B9%88%E8%A6%81%E6%9C%89Positional%20Encoding%EF%BC%9F,Encoding%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81%E2%80%9D%E7%9A%84%E6%A6%82%E5%BF%B5%E3%80%82)
5. [Transformer论文逐段精读](https://www.youtube.com/watch?v=nzqlFIcCSWQ)

