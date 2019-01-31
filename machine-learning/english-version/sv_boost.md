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
