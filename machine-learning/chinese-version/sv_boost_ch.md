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

让我们回到等式：

$$Var(\bar{X}) = p\sigma^2 + \frac{1-p}{n}\sigma^2$$

正如我们所讨论的，减少误差的一种方法是使每个训练模型上的相关性变小。 Bagging可以通过对不同的数据集训练，实现这一目标。我们无法否认的是，由于每个bootstrp从原始数据集中只获取部分训练样本，这可能会使偏差加大。然而事实证明，由此带来的误差的减少将大于偏差的增加。此外，我们可以通过引入更多模型（即增加M）不断减少误差。这并不会导致过拟合，因为$p$对M不敏感，所以整体误差只会减少。

以下两点需要强调：

**1,** 用bagging时，每个决策树并不需要做到完美，OK就差不多了。

**2,** Bagging在非线性数据中表现更好。

### Out-of-bag estimation

在每个bootstrap中，我们只选择原始数据集的一部分。让我们假设我们均匀分布中对其进行**有放回**采样。由于数据集大小为$n\rightarrow \infty$，对于某一个样本，它未被选择的概率为：

$$\begin{align}
\lim\limits_{n\rightarrow \infty} (1-\frac{1}{n})^n &= \lim\limits_{n\rightarrow \infty} \exp(n\log(1-\frac{1}{n})) \\
&\approx \exp(-n\frac{1}{n}) \\
&= \exp(-1)
\end{align}$$

这大约是三分之一，这意味着一个bootstrap中约有三分之一的原始数据未被选中进行训练。为了测试我们的bagging训练模型，对于第i个样本，我们可以用未经过该样本训练过的那些模型（大约M / 3模型）在此样本上进行预测。通过在整个数据集中执行此操作，我们可以获得out-of-bag（词如其名，bagging外的误差）误差估计。在$M\rightarrow\infty$的极端情况下，未对第i个样本进行训练的模型，对所有其他样本进行了训练，这个效果与交叉验证的留一法相同。

## 随机森林

不过Bagging也是存在缺点的。从bootstrap训练的决策树是彼此相关的，因为bootstrap之间是相关的。这是我们不想见到的，我们只希望减少相关性。这样的bagging将无法获得最佳性能。因此，有人就提出了随机森林，这种方法，修改虽小但却很有效。bagging是在所有维度上生长决策树，随机森林则是在随机选择的维度子集中生长决策树，详细为：

对 $b=1,\dots,B$,

**1,** 从训练数据集中提取大小为n的bootstrap$\mathbb{B}_b$

**2,** 对于每次训练，我们从d维度中随机选择m维($m \approx \sqrt(d)$)。对于每个bootstrap，我们有不同的维度m。

# Boosting

我们现在知道了bagging是为了减少使用决策树时的误差，而Boosting则是为了减少偏差。在bagging中，我们生成bootstrap样本训练每个模型。在boosting中，我们在每次训练迭代后对bootstrap中的每个样本进行重新加权。如图所示：

![Bagging Boost Examples](https://raw.githubusercontent.com/Wei2624/AI_Learning_Hub/master/machine-learning/images/cs229_boost_1.png)

定义上讲，Adaboost为:

**1,** Initialize $w_i \leftarrow \frac{1}{N}$ for $i=1,2,\dots,n$ and it is binary classification.

**2,** 对 m=0 到 M:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Sample a bootstrap dataset $B_m$ of size n according to distribution $w_t(i)$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Fit the model $F_m$ on bootstrap $B_t$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Set $\epsilon_m = \sum_{i=1}^n w_m(i) \mathbb{1}[y_i\neq F_m(x_i)] $ and $\alpha_m = \frac{1}{2}\ln\frac{1-\epsilon_m}{\epsilon_m}$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Scale $\bar{w}_{m+1}(i) = w_m(i)\exp(-\alpha_m y_i F_m(x_i))$ and normalize $w_{m+1}(i) = \frac{\bar{w}_{m+1}(i)}{\sum_j \bar{w}_{m+1}(j)}$

**3,** The classification rule is $f_{boost}(x_0) = sign(\sum_{m=1}^M \alpha_m)$

In each iteration, the misclassified samples are up-weighted cumulatively. The final prediction is weighted by weighted error. The summation allows additive terms for adding more modeling capability but will result in a high variance since each trained model is dependent. Thus, increasing M will increase variance as well.

## Boosting分析

值得一谈的是Boosting训练的准确性。这部分是纯粹理论，如果你愿意可以跳过它。

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