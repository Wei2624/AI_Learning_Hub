---
layout: single
mathjax: true
toc: true
toc_sticky: true
category: Machine Learning
tags: [notes]
qr: machine_learning_notes.png
title: Online Learning and Perceptron Algorithm
share: true
permalink: /MachineLearning/sv_online_learning_perceptron/
sidebar:
  nav: "MachineLearning"
---

We have talked about the learning paradigm where we feed a batch of training data to train a model. This is called **batch learning**. In this section, we think about the scenario where the model has to make prediction while it is continously learning on the go. This is called **online learning**.

In this scenario, we have a sequnce of examples $(x^{(1)},y^{(1)}),(x^{(2)},y^{(2)}),\dots,(x^{(n)},y^{(n)})$. What online learning does is to first feed $x^{(1)}$ to the model and ask model to predict, and then show $y^{(1)}$ to the model to let the model perform learning process on it. We do this for one pair of training samples at a time. Eventually, we can come up with a model which has gone through the training dataset. What we are interested in is how many errors this model makes while in online learning process. This is heavily related to the knowledge from learning theory we have discussed before. 

Now, we can take perceptron algorithm as an example. We define $y\in\\{-1,1\\}$ for the label classes. Perceptron algorithm makes prediction based on:

$$h_{\theta}(x) = g(\theta^{T}x)$$

where:

$$g(z) = \begin{cases} 1  \text{, if } z \geq 0 \\ -1  \text{, otherwise} \\ \end{cases}$$

Then the model makes the update to its parameters as:

$$\theta_t = \theta_{t-1} + (h_{\theta}-y)x$$

We can see that if the prediction is correct, we make no change to the parameters. Then, we have the following theorem for the bound on the number of errors made in the online process. 

**Theorem** Let a sequence of examples $(x^{(1)},y^{(1)}),(x^{(2)},y^{(2)}),\dots,(x^{(n)},y^{(n)})$ be given. Suppose that $\lvert\lvert x^{(i)}\rvert\rvert\leq D$ for all i, and further that there exists a unit-length vector u ($\lvert\lvert u\rvert\rvert_2=2$) such that $y^{(i)}(u^Tx^{(i)}\geq \gamma$ for all examples in the sequence(i.e., $u^Tx^{(i)}\geq \gamma$ if $y^{(i)}=1$ and $u^Tx^{(i)}\leq -\gamma$ if $y^{(i)}=-1$ so that u separates the data with the margin at least $\gamma$). Then the total number of mistakes that the perceptron algorithm makes on this sequnece is at most $O(D/\gamma)^2$.

**Proof**. Perceptron is an online learning algorithm. That means it will feed one pair of samples at a time. We also know that perceptron algorithm only updates its parameters when it makes a mistake. Thus, let $\theta^k$ be the weights that were being used for k-th mistake. We initialize from zero vector. Thus, $\theta^1 = \overrightarrow{0}$. In addition, when we make a mistake on i-th iteration, then $g((x^{(i)})^T\theta^k)\neq y^{(i)}$. This is saying:

$$(x^{(i)})^T\theta^k y^{(i)} \leq 0$$

The update rule is $\theta^{k+1} = \theta^k + y^{(i)}x^{(i)}$. We can multiply it by u to have:

$$(\theta^{k+1})^Tu = (\theta^k)^Tu + y^{(i)}(x^{(i)})^Tu \geq (\theta^k)^Tu + \gamma$$

This triggers inductive calculation, which says:

$$(\theta^{k+1})^Tu \geq k\gamma$$

On the other hand, we have:

$$\begin{align}
\lvert\lvert \theta^{k+1}\rvert\rvert^2 &= \lvert\lvert \theta^k + y^{(i)}x^{(i)}\rvert\rvert^2\\
&=  \lvert\lvert\theta^k\rvert\rvert^2 + 2y^{(i)}(x^{(i)})^T\theta^k + \lvert\lvert x^{(i)}\rvert\rvert^2\\
&\leq \lvert\lvert\theta^k\rvert\rvert^2 + \lvert\lvert x^{(i)}\rvert\rvert^2 \\
&\leq \lvert\lvert\theta^k\rvert\rvert^2 + D^2
\end{align}$$

The third step is because last term in step 2 is a negative. Similarly, we can apply induction here to get:

$$\lvert\lvert \theta^{k+1}\rvert\rvert^2 \leq kD^2$$

Now, we combine everything to get:

$$\begin{align}
\sqrt{k}D &\geq \lvert\lvert \theta^{k+1}\rvert\rvert\\
&\geq  (\theta^{k+1})^Tu\\
&\leq k\gamma
\end{align}$$

We have second step because u is unit length vector so the product of the norms is greater than the dot product of the two. This means $k\leq (\frac{D}{\gamma})^2$. Note that this bound does not involve in the number of training samples. So the number of mistakes perceptron made is only bounded by D and $\gamma$. 