Edit by `Mar. 10, 2025`

# Why Input embedding + Positional encoding  not concat(Input embedding, Positional encoding)

紧跟着 embedding 的其实是 $W^q, W^k, W^v$，需要矩阵相成

OK，了解这一点后，我们开始尝试使用 concat 的方式在原始输入中加入位置编码：

给每个位置的 $x^i \in R^{(d, 1)}$ [input embedding] concat 上一个代表位置信息的向量 $p^i \in R^{(N, 1)}$ [positional encoding] 形成 $x_{p}^{i} \in R^{(d+N, 1)}$， 它也可以表示为 $[[x^i]^T, [p^i]^T]^T$ 这个形式。

接着对这个新形成的向量做线性变换。记变换矩阵 $W \in R^{(a, d+N)}$, a 是 out 的 size, 它也可以表示为 $[W^x, W^p]$, 其中 $W^x \in R^{(a, d)}, W^p \in R^{(a, N)}$。

$W \cdot x_{p}^{i} = [W^x, W^p] \cdot [[x^i]^T, [p^i]^T]^T = W^x \cdot [x^i]^T +  W^p \cdot [p^i]^T = word^i + pos^i$

由变换结果可知，在原输入上 concat 一个代表位置信息的向量在经过线性变换后 等同于 将原输入经线性变换后直接加上位置编码信息。


Reference:
1. [Transformer 修炼之道（一）、Input Embedding](https://zhuanlan.zhihu.com/p/372279569)