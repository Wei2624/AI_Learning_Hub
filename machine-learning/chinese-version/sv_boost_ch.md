---
published: true
layout: single
mathjax: true
toc: true
toc_sticky: true
category: Machine Learning
tags: [notes]
excerpt: "This post is a translation for one of Wei's posts in his machine learning notes."
title: Ensembling Methods Chinese Version
share: true
author_profile: true
permalink: /MachineLearning/sv_boost_ch/
---


# 简介

在决策树章节中，我们讨论了如何在回归和分类任务中应用决策树，以及如何构建决策树。正如该章节中所述，决策树模型能力有限，过拟合问题难以解决，我们很难训练一个在一般情况下表现良好的决策树。因此，该章节中提出了使用决策树的集成算法。简而言之，多个训练模型的表现比单个模型的表现会更好。

我们有n个独立同分布的随机变量$X_i$，其中$0 \leq i \leq n$，并假设所有$X_i$有$Var(X_i) = \sigma^2$。那么，我可以得到$X_i$均值的方差为：

$$Var(\bar{X}) = Var(\frac{1}{n}\sum\limits_i X_i) = \frac{\sigma^2}{n}$$

如果我们删除$X_i$独立的假设，则随机变量间是彼此相关的。

$$\begin{align}
Var(\bar{X})&=Var(\frac{1}{n}\sum\limits_i X_i) \\
&= \frac{1}{n^2}\sum\limits_{i,j}Cov(X_i,X_j) \\
&= \frac{n\sigma^2}{n^2} + \frac{n(n-1)p\sigma^2}{n^2} \\
& = p\sigma^2 + \frac{1-p}{n}\sigma^2
\end{align}$$

其中p是皮尔逊相关系数 $p_{X,Y} = \frac{Cov(X,Y)}{\sigma_x\sigma_y}$。我们知道 Cov(X,X) = Var(X)。

**数学**: 以下证明有助于理解上述步骤。

$$\begin{align}
Var(\frac{1}{n}\sum\limits_i X_i) &= \frac{1}{n^2} Var(\sum\limits_i X_i) \\
&= \mathbb{E}[(\sum\limits_i X_i)^2] - (\mathbb{E}[\sum\limits_i X_i])^2 \\
&=\mathbb{E}[\sum\limits_{i,j}X_i X_j] - (\mathbb{E}[\sum\limits_i X_i])^2  \\
&=\sum\limits_{i,j}\mathbb{E}[X_iX_j]- (\mathbb{E}[\sum\limits_i X_i])^2 \\
&= \sum\limits_{i,j}\mathbb{E}[X_iX_j] - \sum\limits_{i,j}\mathbb{E}[X_i] \mathbb{E}[X_j] \\
&= \sum\limits_{i,j} \mathbb{E}[X_iX_j] - \mathbb{E}[X_i] \mathbb{E}[X_j] \\
&= \sum\limits_{i,j} Cov(X_i,X_j)
\end{align}$$

**返回主题**：现在，如果我们将每个随机变量视为一个训练模型的误差，我们可以通过以下方式减少此方差：

1, 增加随机变量（即模型数量）n的数量以式子后半部分变小

2，减少每个随机变量之间的相关性，使第一项变小，使其更靠近独立同分布状态

问题是，我们如何实现这些目标呢？在此章节中，我们将介绍**Bagging**和**Boosting**。

# Bagging

## Bootstrap

简单来讲，Bootstrap是一种重新采样技术，它可以用于改进数据的estimator。在该算法中，我们从数据的经验分布中不断采样，最后得到数据的统计值。

假设我们有一个经过训练的estimator E，这个estimator可以预测数据的中位数。我们想知道这个estimator估算的置信度有多高，以及它与真实数据的差异有多大。这里我们可以使用bootstrap来进行测评。在bootstrap算法中，我们可以：

**1,** Bootstrap样本$\mathbb{B}_1,\dots,\mathbb{B}_B$，其中$\mathbb{B}_b$，是通过从数据为n的数据集中**有放回**的抽取样本而生成的。

**2,** 得到每个Bootstrap $\mathbb{B}_b$的estimator为：

$$E_b = E(\mathbb{B}_b)$$

**3,** 计算E的均值与方差:

$$\mu_B = \frac{1}{B}\sum\limits_{n=1}^B E_b,   \sigma_B^2 = \frac{1}{B}\sum\limits_{b=1}^B (E_b - \mu_B)^2$$

这可以让我们了解estimator在估算数据中值时的表现如何。

## Bagging

Bagging使用bootstrap的概念进行回归或分类，它代表着**Bootstrap聚合**。

算法如下：

对于$b=1,\dots,B$，

**1,** 从训练数据集中提取大小为n的bootstrap$\mathbb{B}_b$
Draw a bootstrap $\mathbb{B}_b$ of size n from training dataset

**2,** 对bootstrap $\mathbb{B}_b$，训练决策树分类器或决策树回归模型$f_b$。

要预测新数据点$x_0$，我们需要计算：

$$f(x_0) = \frac{1}{B} \sum\limits_{b=1}^B f_b(x_0)$$

对于回归问题，我们只需要计算出所有分类器的预测平均值即可。对于分类任务，我们可以使用投票机制来获得最终结果。

假设在二元分类中，有一个输入特征$x\in \mathbb{R}^5$。如下所示，我们可以使用bootstrap来训练多个分类器：

![Bagging Examples](https://raw.githubusercontent.com/Wei2624/AI_Learning_Hub/master/machine-learning/images/cs229_trees_15.png)

Let's back to the equation:

$$Var(\bar{X}) = p\sigma^2 + \frac{1-p}{n}\sigma^2$$

As we talked about, one way to reduce the variance is that we have less correlation on each trained model. Bagging achieves this by training on different datasets. One might concert about the fact that this will increase the bias since each bootstrap does not take the full training samples from the original dataset. However, it turns out that the decrease in variance is more than the increase in bias. Also, we can keep reducing variance by introducing more models (i.e. increase M). This will not lead to overfitting because $p$ is insensitive to M. Thus, overall variance can only decrease.

However, there are two key points that should be emphasized:

**1,** With bagging,each tree does not need to be perfect. "Ok" is fine.

**2,** Bagging often improves when the function is non-linear.

### Out-of-bag estimation

In each bootstrap, we can only contain a portion of original dataset. Let's assume that we sample it from uniform distribution with replacement. As dataset size $n\rightarrow \infty$, we have the probability of a sample not being selected as:

$$\begin{align}
\lim\limits_{n\rightarrow \infty} (1-\frac{1}{n})^n &= \lim\limits_{n\rightarrow \infty} \exp(n\log(1-\frac{1}{n})) \\
&\approx \exp(-n\frac{1}{n}) \\
&= \exp(-1)
\end{align}$$

This is roughly one third. That means one third of data not being selected in a single bootstrap. To test our bagging-trained models, for i-th sample, we can ask those models (approxmiately M/3 models) that are not trained on this sample to make prediction. By doing this over the entire dataset, we can obtain the out-of-bag error estimation. In the extreme case where $M\rightarrow\infty$, those models that are not trained on i-th sample are trained on all other samples. This gives us the same results as leave-one-out corss-validation does.

## Random Forests

There are drawbacks of Bagging. The bagged tress trained from bootstrap is related because bootstraps are correlated. This is unwanted because we want to have less correlation. The bagging will not be able to have the best performance. Thus, random forest is proposed. The modification is small but works. Instead of growing a tree on all dimensions, random forests propose to grow a tree on randomly selected subset of dimensions. In details,

For $b=1,\dots,B$,

**1,** Draw a bootstrap $\mathbb{B}_b$ of size n from training dataset

**2,** Train a tree classifier on $\mathbb{B}_b$. For each training, we randomly select a predefined m dimensions of d ($m \approx \sqrt(d)$). For each bootstrap, we have different m dimensions.

# Boosting

We know that bagging is to reduce the variance from a single tree. Boosting is, on the other hand, to reduce bias. In bagging, we generate bootstrap sample to train each individual model. In boosting, we re-weighted each sample in the bootstrap after every training iteration. Graphically, we have:

![Bagging Boost Examples](https://raw.githubusercontent.com/Wei2624/AI_Learning_Hub/master/machine-learning/images/cs229_boost_1.png)

Formally, we have the AdaBoost Algorithm as:

**1,** Initialize $w_i \leftarrow \frac{1}{N}$ for $i=1,2,\dots,n$ and it is binary classification.

**2,** For m=0 to M:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Sample a bootstrap dataset $B_m$ of size n according to distribution $w_t(i)$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Fit the model $F_m$ on bootstrap $B_t$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Set $\epsilon_m = \sum_{i=1}^n w_m(i) \mathbb{1}[y_i\neq F_m(x_i)] $ and $\alpha_m = \frac{1}{2}\ln\frac{1-\epsilon_m}{\epsilon_m}$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Scale $\bar{w}_{m+1}(i) = w_m(i)\exp(-\alpha_m y_i F_m(x_i))$ and normalize $w_{m+1}(i) = \frac{\bar{w}_{m+1}(i)}{\sum_j \bar{w}_{m+1}(j)}$

**3,** The classification rule is $f_{boost}(x_0) = sign(\sum_{m=1}^M \alpha_m)$

In each iteration, the misclassified samples are up-weighted cumulatively. The final prediction is weighted by weighted error. The summation allows additive terms for adding more modeling capability but will result in a high variance since each trained model is dependent. Thus, increasing M will increase variance as well.

## Analysis of boosting

it is worth talking about the accuracy of boosting on training data. This is purely theoretic, and you can skip it if you want.

**Theorem**: With AdaBoost algorithm, if $\epsilon_m$ is the weighted error of classifier $f_m$, then the final classification $f_{noost}(x_0)=sign(\sum_{m=1}^M \alpha_mf_m(x_0))$. The training error can be bounded as:

$$\frac{1}{n}\sum\limits_{i=1}^n \mathbb{1}[y_i\neq f_{boost}(x_i)] \leq \exp(-2\sum\limits_{m=1}^M (\frac{1}{2}-\epsilon_m)^2)$$

What it means that even though each $\epsilon_m$ is just a little better than random guessing, the sum over M models can produce a large negative value in the exponent when M is larger. Thus, we have a small upper bound.

**Proof**:

To prove this, we want to find an intermediate value as the stepping stone. That is, we find a < b and b < c, then a < c.

Recall that:

$$\bar{w}_{m+1}(i) = w_m (i) \exp(-\alpha_m y_i F_m(x_i))$$

$$w_{m+1}(i) = \frac{\bar{w}_{m+1}(i)}{\sum_j \bar{w}_{m+1}(j)}$$

Let's define:

$$Z_m = \sum_j \bar{w}_{m+1}(j)$$

Now, we can re-write:

$$w_{m+1}(i) = \frac{1}{Z_m} w_m(i)\exp(-\alpha_m y_i F_m(x_i))$$

We can use this to re-write:

$$\begin{align}
w_{M+1}(i) &= w_1(i)\frac{\exp(-\alpha_1 y_i F_1(x_i))}{Z_1} \times \frac{\exp(-\alpha_2 y_i F_2(x_i))}{Z_2}  \\
&\dots\times \frac{\exp(-\alpha_M y_i F_M(x_i))}{Z_M}
\end{align}$$

We know that $w_1(i) = \frac{1}{n}$ since I initialized this way. So we have:

$$w_{M+1}(i) = \frac{1}{n}\frac{\exp(-y_i\sum_{m=1}^M \alpha_m F_m(x_i))}{\prod_{m=1}^M Z_m} = \frac{1}{n}\frac{\exp(-y_i h_M(x_i))}{\prod_{m=1}^M Z_m}$$

where we define $h_M(x) = \sum_{m=1}^M \alpha_m F_m(x)$. And $\prod_{m=1}^M Z_m$ is our "b" above. Next, we can re-write the weights as:

$$w_{T+1}(i) \prod_{m=1}^M Z_m = \frac{1}{n} \exp(-y_i h_M(x_i))$$

Then, we can plug our training error back. Note that $0 < \exp(z_1), 1<\exp(z_2)$ for any $z_1 <0< z_2$. We have:

$$\begin{align}
\frac{1}{n}\sum\limits_{i=1}^n \mathbb{1}[y_i\neq f_{boost}] &\leq \frac{1}{n}\sum\limits_{i=1}^n \exp(-y_i h_M(x_i)) \\
&= \sum\limits_{i=1}^n w_{M+1}(i)\prod_{m=1}^M Z_m = \prod_{m=1}^M Z_m
\end{align}$$

We have shown that the training error is less or equal to an intermediate value "b". Then, we work on a single $Z_m$:

$$\begin{align}
Z_m &= \sum\limits_{i=1}^n w_m(i)\exp(-y_i\alpha_m F_m(x_i)) \\
&= \sum\limits_{i:y_i=F_m(x_i)} \exp(-\alpha_m w_m(i) + \sum\limits_{i:y_i\neq F_m(x_i)} \exp(\alpha_m)w_m(i) \\
&= \exp(-\alpha_m)(1 - \epsilon_m) + \exp(\alpha_m)\epsilon_m
\end{align}$$

where $\epsilon_m = \sum_{i:y_i\neq F_m(x_i)} w_m(i)$. If we minimize $Z_m$ with respect to $\alpha_m$, we can get:

$$\alpha_m = \frac{1}{2}\ln (\frac{1 - \epsilon_m}{\epsilon_m})$$

This is exactly what we have set up at the beginning.

We can plug this back to find out:

$$Z_m = 2\sqrt{\epsilon_m(1-\epsilon_m)} = \sqrt{1 - 4(\frac{1}{2} - \epsilon_m)^2}$$

We know that $1 - x \leq \exp(-x)$. Then, we can say:

$$Z_m = (1 - 4(\frac{1}{2} - \epsilon_m)^2)^{\frac{1}{2}} \leq (\exp(-4(\frac{1}{2} - \epsilon_m)^2))^{\frac{1}{2}} = \exp(-2(\frac{1}{2} - \epsilon_m)^2)

For all $Z_m$, we can have:

$$\prod_{m=1}^M Z_m \leq \exp(-2\sum_{m=1}^M (\frac{1}{2}-\epsilon_m)^2)$$

## Forward Stagewise Additive Modeling

Before talking about a new boosting algorithm, it is worth talking about the general framework of ensembling. It is called **Forward Stagewise Additive Modeling**. In details, we have

**Input**: Labeled training data $(x_1,y_1),\dot,(x_N,y_N)$

**Output**: Ensemble classifier f(x)

1, Initialize $f_0(x) = 0$

2, for m=1 to M do:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Compute $(\beta_m,\gamma_m) = \arg\min_{\beta,\gamma}\sum_{i=1}^N L(y_i,f_{m-1}(x_i) + \beta G(x_i;\gamma))$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Set $f_m(x) = f_{m-1}(x) + \beta_m G(x;\gamma_m)$

3, Output $f(x) = f_m(x)$

In each iteration, we fix the weights and parameters of all the trained models from previous steps. We have a weak learner G(x) parameterized by $\gamma$. We can show that Adaboost is a special case of this formulation of binary setting and exponential loss:

$$L(y,\bar{y}) = \exp(-y\bar{y})$$

Also, we can also show that if we plug in a squared loss, then:

$$L=\sum\limits_{i=1}^N (y_i-(f_{m-1}(x_i) + G(x_i)))^2 = ((y_i-f_{m-1}(x_i)) - G(x_i))^2$$

It means squared loss in this formulation is equally saying that we are fitting the individual classifier to the residual $(y_i-f_{m-1}(x_i)$. This just opens a short introduction to stagewise additive learning, if you want to see more, you should check more on textbooks.

## Gradient boosting

Boosting is used in many areas. It is also one of examples in stagewise additive modeling. The core idea is that every iteration we learn a weak learner. That is, we just need each one to perform a little better than random guess. At the end, we can aggregate all weak learns together to form a strong one. In Adaboost, for every iteration, we want the new model to focus on the re-weighted data samples. For gradient boosting, the core idea is that we want the new model to focus on gradients from biased predictions.

There are several steps to follow:

1, initialize $f_0(x) = c$

2, At i-th iteration, for sample $j=1,\dots,N$, we compute:

$$g_{ij} = \frac{\partial L(y_i,f_{i-1}(x_i))}{\partial f_{i-1}(x_i)}$$

At this point, we have new pairs $(x_1,g_{1i}),\dots,(x_N,g_{Ni})$ for i-th iteration.

3, Fit a new decision or regression tree on the new pairs $(x_1,g_{1i}),\dots,(x_N,g_{Ni})$ for i-th iteration. That is,

$$\gamma_i = \arg\min_{\gamma}\sum\limits_{j=1}^N (g_j-G(x_j;\gamma))^2$$

4, We set

$$f_i(x) = f_{i+1}(x) + G(x;\gamma_i)$$

We can do this M iterations to get $f_M(X)$, which is out final output model.

Again, this is just a short introduction to **Gradient Boosting**. More can be found on textbooks. There are two links that I found very helpful:

[Tutorial from Northeastern University by Prof. Cheng Li](http://www.chengli.io/tutorials/gradient_boosting.pdf)

[Top voted answer from Quora](https://www.quora.com/What-is-an-intuitive-explanation-of-Gradient-Boosting)