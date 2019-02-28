---
published： true
layout： single
mathjax： true
toc： true
toc_sticky： true
category： Machine Learning
tags： [notes]
excerpt： "This post is a translation for one of Wei's posts in his machine learning notes."
title： Ensembling Methods Chinese Version
share： true
author_profile： true
permalink： /MachineLearning/sv_boost_ch/
---


# 简介

在决策树章节中，我们讨论了如何在回归和分类任务中应用决策树，以及如何构建决策树。正如决策树章节中所述，决策树模型能力有限，过拟合问题难以解决，我们很难训练一个在一般情况下表现良好的决策树。因此，该章节中提出了使用决策树的集成算法。简而言之，多个训练模型的表现比单个模型的表现会更好。

我们有n个独立同分布的随机变量$X_i$，其中$0 \leq i \leq n$，并假设所有$X_i$有$Var(X_i) = \sigma^2$。那么，我可以得到$X_i$均值的方差为：

$$Var(\bar{X}) = Var(\frac{1}{n}\sum\limits_i X_i) = \frac{\sigma^2}{n}$$

如果我们删除$X_i$独立的假设，则随机变量间是彼此相关的。

$$\begin{align}
Var(\bar{X})&=Var(\frac{1}{n}\sum\limits_i X_i) \\
&= \frac{1}{n^2}\sum\limits_{i，j}Cov(X_i,X_j) \\
&= \frac{n\sigma^2}{n^2} + \frac{n(n-1)p\sigma^2}{n^2} \\
& = p\sigma^2 + \frac{1-p}{n}\sigma^2
\end{align}$$

其中p是皮尔逊相关系数 $p_{X,Y} = \frac{Cov(X,Y)}{\sigma_x\sigma_y}$。我们知道 Cov(X,X) = Var(X)。

**数学**： 以下证明有助于理解上述步骤。

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

**1,** Bootstrap样本$\mathbb{B}\_1,\dots,\mathbb{B}\_B$，其中$\mathbb{B}\_b$，是通过从数据为n的数据集中**有放回**的抽取样本而生成的。

**2,** 得到每个Bootstrap $\mathbb{B}\_b$的estimator为：

$$E_b = E(\mathbb{B}\_b)$$

**3,** 计算E的均值与方差：

$$\mu_B = \frac{1}{B}\sum\limits_{n=1}^B E_b,   \sigma_B^2 = \frac{1}{B}\sum\limits_{b=1}^B (E_b - \mu_B)^2$$

这可以让我们了解estimator在估算数据中值时的表现如何。

## Bagging

Bagging使用bootstrap的概念进行回归或分类，它代表着**Bootstrap聚合**。

算法如下：

对于$b=1,\dots,B$，

**1,** 从训练数据集中提取大小为n的bootstrap数据$\mathbb{B}\_b$

**2,** 对bootstrap数据$\mathbb{B}\_b$训练决策树分类器或决策树回归模型$f_b$。

要预测新数据点$x_0$，我们需要计算：

$$f(x_0) = \frac{1}{B} \sum\limits_{b=1}^B f_b(x_0)$$

对于回归问题，我们只需要计算出所有分类器的预测平均值即可。对于分类任务，我们可以使用投票机制来获得最终结果。

假设在二元分类中，有一个输入特征$x\in \mathbb{R}^5$。如下所示，我们可以使用bootstrap算法来训练多个分类器：

![Bagging Examples](https：//raw.githubusercontent.com/Wei2624/AI_Learning_Hub/master/machine-learning/images/cs229_trees_15.png)

让我们回到等式：

$$Var(\bar{X}) = p\sigma^2 + \frac{1-p}{n}\sigma^2$$

正如我们所讨论的，减少误差的一种方法是使每个训练模型上的相关性变小。 Bagging可以通过对不同的数据集训练，实现这一目标。我们无法否认的是，由于每个bootstrp从原始数据集中只获取部分训练样本，这可能会使偏差加大。然而事实证明，由此带来的误差的减少将大于偏差的增加。此外，我们可以通过引入更多模型（即增加M或者在等式中的n）不断减少误差。这并不会导致过拟合，因为$p$对M不敏感，所以整体误差只会减少。

