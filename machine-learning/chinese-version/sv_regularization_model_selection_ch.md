---
published: true
layout: single
mathjax: true
toc: true
toc_sticky: true
category: Machine Learning
tags: [notes,chinese]
excerpt: "This post is a translation for one of Wei's posts in his machine learning notes."
title: Regularization Model Selection Chinese Version
share: true
author_profile: true
permalink: /MachineLearning/sv_regularization_model_selection_ch/
---

This Article is a Chinese translation of a study note by Wei. Click [here](https://wei2624.github.io/MachineLearning/sv_regularization_model_selection/) to see the original English version in Wei's homepage. I will continue to update Chinese translation to sync with Wei's notes.

请注意: 本文是我翻译的一份学习资料，英文原版请点击[Wei的学习笔记](https://wei2624.github.io/MachineLearning/sv_regularization_model_selection/)。我将不断和原作者的英文笔记同步内容，定期更新和维护。

正则化与模型选择
在选择模型时，如果我们在一个模型中有k个参数，那么问题就是这k个参数应该是什么值？哪些值可以给出最佳偏差-方差权衡呢。其中，我们从有限集合的模型 $\mathcal{M} = \{M_1,M_2,\dots,M_d\}$ 中来选取最佳模型。在集合中，我们有不同的模型，或者不同的参数。

# 1 交叉验证(Cross Validation)

想象一下，给定数据集S与一系列的模型，我们很容易想到通过以下方式来选择模型：

1从S集合训练每个模型$M_i$ ，并得到相应的假设$h_i$

2选取最小训练误差的模型

这个想法不能达到目的因为当我们选择的多项指数越高时，模型会更好的拟合训练数据集。然而，这个模型将会在新的数据集中有很高的统一化误差，也就是高方差。

在这个情况中，** 交叉验证（hold-out cross validation）**将会做得更好：

1 以70%和30%的比例将S随机分成训练数据集$S_{tr}$和验证数据集$S_{cv}$ 

2 在StrStr在中训练每一个 $M_i$ 以学习假设 $h_i$

3 选择拥有最小**经验误差(empirical error)**的模型 $S_{cv}$，我们将它标记为
$\hat{\varepsilon}\_{S_{cv}}(h_i)$

通过以上几步，我们试图通过测试模型在验证集上的表现以估计真实统一化误差。在第3步中，在选择最优模型后，我们可以用整个数据集来重复训练模型来得到最佳假设模型。然而，即使我们可以这样做，我们仍然选择的是基于70%数据集来训练模型。当数据少的时候这是很糟糕的。

因此，我们引出**K层交叉验证(K-fold cross validation)**：

1随机将S分成k个分离的子集，每个子集有m/k个样本，记为$S_1,S_2,\dots,S_k$

2 对于每个模型$M_i$，我们排除一个子集并标记为j，然后我们用其余的样本训练模型以得到$H_{ij}$。我们在$S_j$上测试模型，并且得到 $\varepsilon_{S_j}(h_{ij})$。我们这样遍历每一个j。最后，我们获取统一化误差除以j的平均。

3我们选择有最小平均统一误差的模型

通常我们取k为10。虽然这样算术很复杂，但是它会给我们很好的结果。如果数据很少，我们也可能设k=m。在这种情况下，我们每一次除去一个样本，这种方法叫**除一交叉验证(leave-one-out cross validation)**。

# 2 特征选择(Feature Selection)

如果我们有n个特征，m个样本，其中$n \gg m$ (VC 维度is O(n)),我们可能会过度拟合。在这种情况下，你想选择最重要的特征来训练。在暴力算法中，我们会有用$2^n$ 个特征组合，我们会有$2^n$ 个可能的模型，这处理起来会很费力。因此我们可以选择用**向前搜索算法(forward search algorithm)**:

1 我们初始化为$\mathcal{F} = \emptyset$

2 重复：(a)for $i =1,\dots,n$ 如果$i\notin\mathcal{F}$, 让$\mathcal{F}_i = \mathcal{F}\cup\{i\}$ 并且使用交叉验证算法来估计$\mathcal{F}_i$. (b)设置$\mathcal{F}$作为(a)中的最佳特征子集

3 从以上选择最佳特征子集。

你可以通过设置期望的特征数量来终止循环。相反地，在特征选择中我们也可以使用**向后搜索算法(backward search)**，这于离格算法类似。然而，因为这两种算法的时间复杂度都是$O(n^2)$ ，它们训练起来都会比较慢。

然而，我们也可以使用**过滤特征选择(Filter feature selection) **。它的原理是对于标签y，我们会根据每一个特征提供了多少信息来给它打分，然后挑选出最佳者。
一个容易想到的方法是根据xixi 和标签y的相关性打分。实际中，我们将分数设为**相互信息(mutual information)**:

$$MI(x_i,y) = \sum\limits_{x_i\in\{0,1\}}\sum\limits_{y\in\{0,1\}} p(x_i,y)\log\frac{p(x_i,y)}{p(x_i)p(y)}$$

其中我们假设每个特征和标签都是二元值，并且求和包括所有变量。每一个可能性都会从训练数据集中计算。为了进一步理解，我们知道：

$$MI(x_i,y) = KL(p(x_i,y)\lvert\lvert p(x_i)p(y))$$

其中KL是**相对熵(Kullback-Leibler divergence)**。它计算了竖线两边变量分布的差异。如果$x_i$和 $y$ 是独立的，那么 KL 是0。这代表着特征和标签直接没有任何关系。然而如果MI很高，那么这个特征和标签有强相关性。

# 3 贝叶斯统计与正则化(Bayesian Statistics and regularization)

在前面一章我们讨论了**最大似然法(maximum likelihood (ML) algorithm)**是如何训练模型参数的：

$$\theta_{ML} = \arg\max\prod_{i=1}^m p(y^{(i)}\lvert x^{(i)},\theta)$$

在这种情况下，我们视$\theta$ 为未知参数，它已经存在而恰好未知。我们的任务是找到未知参数并计算它的值。
同时$\theta$也是随机的，因此我们设置一个先验值，称它为**先验分布(prior distribution)**。基于先验分布，我们可以用S数据集来计算后验分布：

$$p(\theta\lvert S) = \frac{p(S\lvert\theta)p(\theta)}{p(S)} = \frac{\prod_{i=1}^m p(y^{(i)}\lvert x^{(i)},\theta)(p(\theta)}{\int_{\theta}\prod_{i=1}^m p(y^{(i)}\lvert x^{(i)},\theta)(p(\theta)d\theta}$$

使用后验分布来预测推断，我们有：

$$p(y\lvert x,S) = \int_{\theta}p(y\lvert x,\theta)p(\theta\lvert S)d\theta$$

在这一点上，我们可以计算条件期望值y。然而计算封闭解的后验值是很难的，因为分母中的积分很难在封闭解中计算。因此，我们用另一种方式来计算，我们找到一个后验值的点估计，在这个点上我们获得后验值的最佳 $\theta$。**最大后验MAP(maximum a posteriori)** 可以用以下方法计算：

$$\theta_{MAP} = \arg\max_{\theta} = \prod_{i=1}^m p(y^{(i)}\lvert x^{(i)},\theta)p(\theta)$$

通常来讲，先验分布有0均值，单位方差。这会使MAP 比ML 更不容易过度拟合。

