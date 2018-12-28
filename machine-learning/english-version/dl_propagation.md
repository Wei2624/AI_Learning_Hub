---
layout: single
mathjax: true
toc: true
toc_sticky: true
category: Machine Learning
tags: [notes]
qr: machine_learning_notes.png
title: Nerual Networks
share: true
permalink: /MachineLearning/dl_propagtion/
sidebar:
  nav: "MachineLearning"
---

# 1 Forward Propagation

This is more like a summary section. 

We set $a^{[0]} = x$ for our input to the network and $\ell = 1,2,\dots,N$ where N is the number of layers of network. Then, we have

$$z^{[\ell]} = W^{[\ell]}a^{[\ell-1]} + b^{[\ell]}$$

$$a^{[\ell]} = g^{[\ell]}(z^{[\ell]})$$

where $g^{[\ell]}$ is the same for all the layers except for last layer. For the last layer, we can do:

&nbsp;&nbsp;&nbsp;&nbsp;1 regression then $g(x) = x$

&nbsp;&nbsp;&nbsp;&nbsp;2 binary then $g(x) = sigmoid(x)$

&nbsp;&nbsp;&nbsp;&nbsp;3 multi-class then $g(x) = softmax(x)$

Finally, we can have the output of the network $a^{[N]}$ and compute its loss. 

For regression, we have:

$$\mathcal{L}(\hat{y},y) = \frac{1}{2}(\hat{y} - y)^2$$

For binary classification, we have:

$$\mathcal{L}(\hat{y},y) = -\bigg(y\log\hat{y} + (1-y)\log (1-\hat{y})\bigg)$$

For multi-classification, we have:

$$\mathcal{L}(\hat{y},y) = -\sum\limits_{j=1}^k\mathbb{1}\{y=j\}\log\hat{y}_j$$


Note that for multi-class, if we have $\hat{y}$ as a k-dimensional vector, we can calculate its cross-entropy for its loss:

$$\mathcal{L}(\hat{y},y) = -\sum\limits_{j=1}^ky_j\log\hat{y}_j$$

# 2 Backpropagation

We define that:

$$\delta^{[\ell]} = \triangledown_{z^{[\ell]}}\mathcal{L}(\hat{y},y)$$

So we have three steps for computing the gradient for any layer:

1 For output layer N, we have:

$$\delta^{[N]} = \triangledown_{z^{[N]}}\mathcal{L}(\hat{y},y)$$

For softmax function, since it is not performed element-wise, so you can directly caculate it as a whole. For sigmoid, it is applied element-wise, so we need to:

$$\triangledown_{z^{[N]}}\mathcal{L}(\hat{y},y) = \triangledown_{\hat{y}}\mathcal{L}(\hat{y},y)\circ (g^{[N]})^{\prime}(z^{[N]})$$

Note this is element-wise operation.

2 For $\ell = N-1,N-2,\dots,1$, we have:

$$\delta^{[\ell]} = (W^{[\ell+1]T}\delta^{[\ell+1]})\circ g^{\prime}(z^{[\ell]})$$

3 For each layer, we have:

$$\triangle_{W^{[\ell]}}J(W,b) = \delta^{[\ell]}a^{[\ell]T}$$

$$\triangle_{b^{[\ell]}}J(W,b) = \delta^{[\ell]}$$

This can be directly used in coding, which acts like a formula. 
