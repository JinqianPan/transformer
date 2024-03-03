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

### More details

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

2. Positional encoding: 

## Coding

## Reference:
1. Attention Is All You Need: https://arxiv.org/abs/1706.03762
2. The Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/
3. Transformer模型详解（图解最完整版）: https://zhuanlan.zhihu.com/p/338817680

