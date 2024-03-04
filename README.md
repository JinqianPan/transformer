Edit by `Mar. 3, 2024`

# Transformer

## Theoretical knowledge
> A transformer is a deep learning architecture based on the multi-head attention mechanism, proposed in a 2017 paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762). -- Wiki

### High level review
The transformer could be regarded as a `black box` in high level. This black box is composed by `Encoders` and `Decoders`.

<p align="center">
  <img src="./img/001.png" width="700">
</p>

Encoders and decoders are stacked by multiple encoders and decoders of the same number. The simple transformer seems to use 6 encoders and decoders.

<p align="center">
  <img src="./img/003.png" width="700">
</p>

`Each encoder` has `same structure`, but they `do not share weights`. (Same as decoders)

### Processing

<p align="center">
  <img src="./img/004.png" width="300">
</p>

#### Input part
In the transformer, the the input part is **input embedding (word embedding)** and **positional encoding**.
1. Input embedding: usually input embedding have two chooses, one-hot encoding and word embedding. Its essence is to make input into a set of numbers.
    * One-hot encoding
    * Word embedding generates *similar positional positioning of semantically similar and related words* in the Embedding space.

<p align="center">
  <img src="./img/005.png" width="300">
  <img src="./img/006.png" width="300">
</p>

2. Positional encoding: Since there are **no loops or convolutional structures** in Transformer, in order to **enable the model to take advantage of the sequence**, authors insert some information about the **relative or absolute position** of tokens in the **sequence**.[4]

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

