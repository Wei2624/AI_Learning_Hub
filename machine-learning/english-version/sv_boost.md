---
layout: single
mathjax: true
toc: true
toc_sticky: true
category: Machine Learning
tags: [notes]
qr: machine_learning_notes.png
title: Ensembling Methods
share: true
permalink: /MachineLearning/sv_boost/
sidebar:
  nav: "MachineLearning"
---

# Introduction

In Decision Tree section, we talked about how decision tree is applied in regression and classification task and how we can grow a decision tree. However, as also mentioned in the section, decision tree has limited power to generalize well. Thus, it has been proposed to use ensembling methods. In a word, multiple trained models perform better than the single model. But why?

Let's have n i.i.d. random variables $X_i$ for $0 \leq i \leq n$ and assume $Var(X_i) = \sigma^2$ for all $X_i$. Then, we have the variance of the mean:

$$Var(\bar{X}) = Var(\frac{1}{n}\sum\limits_i X_i) = \frac{\sigma^2}{n}$$

If we remove the independent assumption, then each random variable is correlated.

$$\begin{align}
Var(\bar{X})&=Var(\frac{1}{n}\sum\limits_i X_i) \\
&= \frac{1}{n^2}\sum\limits_{i,j}Cov(X_i,X_j) \\
&= \frac{n\sigma^2}{n^2} + \frac{n(n-1)p\sigma^2}{n^2} \\
& = p\sigma^2 + \frac{1-p}{n}\sigma^2
\end{align}$$

where p is pearson correlation coefficient $p_{X,Y} = \frac{Cov(X,Y)}{\sigma_x\sigma_y}$. We know that Cov(X,X) = Var(X).

**Math**: The following proof might be helpful for understanding the steps above.

$$\begin{align}
Var(\frac{1}{n}\sum\limits_i X_i) &= \frac{1}{n^2} Var(\sum\limits_i X_i) \\
&= \mathbb{E}[(\sum\limits_i X_i)^2] - (\mathbb{E}[\sum\limits_i X_i])^2 \\
&=\mathbb{E}[\sum\limits_{i,j}X_i X_j] - (\mathbb{E}[\sum\limits_i X_i])^2  \\
&=\sum\limits_{i,j}\mathbb{E}[X_iX_j]- (\mathbb{E}[\sum\limits_i X_i])^2 \\
&= \sum\limits_{i,j}\mathbb{E}[X_iX_j] - \sum\limits_{i,j}\mathbb{E}[X_i] \mathbb{E}[X_j] \\
&= \sum\limits_{i,j} \mathbb{E}[X_iX_j] - \mathbb{E}[X_i] \mathbb{E}[X_j] \\
&= \sum\limits_{i,j} Cov(X_i,X_j)
\end{align}$$

**Back to topic**: Now, if we treat each random variable is the error of a trained model, we can see that the variance can be reduced by:

1, increase the number of random variable (i.e. number of models) n to make the second term smaller

2, reduce the correlation between each random variable to make the first term smaller, and it become more i.i.d.

How do we achieve those? Here, we will introduce **Bagging** and **Boosting**.

# Bootstrap

Bootstrap is basically a re-sampling technique for improving estimators on the data. In this algorithm, we keep sampling from the empirical distribution of the data to estimate statistics of the data.

Let's say we have an trained estimator E for the statistic such as median of the data. The question is how confident our estimator is and how much variance it is. We can use bootstrap to find out. In bootstrap algorithm, we do:

1, Generate bootstrap samples $\mathbb{B}_1,\dots,\mathbb{B}_B$ where $\mathbb{B}_b$ is created by picking n samples from the dataset of size n **with** replacement

2, Evaluate the estimator on each $\mathbb{B}_b$ as:

$$E_b = E(\mathbb{B}_b)$$

3, Estimate the mean and variance of E:

$$\mu_B = \frac{1}{B}\sum\limits_{n=1}^B E_b, \sigma_B^2 = \frac{1}{B}\sum\limits_{b=1}^B (E_b - \mu_B)^2$$

This can tell us how our estimator performs on estimating the median of the data.

# Bagging and Random Forests

Bagging basically uses the idea of bootstrap for regression or Classification. It represents Bootstrap aggregation.

The algorithm is as the following:

For $b=1,\dots,B$,

1, Draw a bootstrap $\mathbb{B}_b$ of size n from training dataset

2, Train a tree classifier or tree regression model $f_b$ on $\mathbb{B}_b$.

To predict, for a new point $x_0$, we compute:

$$f(x_0) = \frac{1}{B} \sum\limits_{b=1}^B f_b(x_0)$$

For regression problem, we can see this is just the average of of prediction of each trained classifier. For classification task, we can view this as a voting mechanism.  


For example, let's say we have an input feature $x\in \mathbb{R}^5$ for a binary classification.  We can use bootstrap strategy to train multiple classifier as:

![Bagging Examples](https://raw.githubusercontent.com/Wei2624/AI_Learning_Hub/master/machine-learning/images/cs229_trees_15.png)

There are two key points that should be emphasized:

1, With bagging,each tree does not need to be perfect. "Ok" is fine.

2, Bagging often improves when the function is non-linear.

## Random Forests

There are drawbacks of Bagging. The bagged tress trained from bootstrap is related because bootstraps are correlated. The bagging will not be able to have the best performance. Thus, random forest is proposed. The modification is small but works. Instead of growing a tree on all dimensions, random forests propose to grow a tree on randomly selected subset of dimensions. In details,

For $b=1,\dots,B$,

1, Draw a bootstrap $\mathbb{B}_b$ of size n from training dataset

2, Train a tree classifier on $\mathbb{B}_b$. For each training, we randomly select a predefined m dimensions of d ($m \approx \sqrt(d)$). For each bootstrap, we have different m dimensions.
