---
layout: single
mathjax: true
toc: true
toc_sticky: true
category: Machine Learning
tags: [notes]
qr: machine_learning_notes.png
title: Factor Analysis
share: true
permalink: /MachineLearning/usv_factor_analysis/
sidebar:
  nav: "MachineLearning"
---

# Introduction

Recall that when we have the data $x^i\in \mathbb{R}^n$ for mixture of Gaussian, we usually assume that the number of samples m is larger than the sample dimension n. Then, EM algorithm can be applied to fit the data. However, EM algorithm will fail if data dimension is larger than the number of sample.

For example, if $n \gg m$, in such case, it might be difficult to model the data with even a single Gaussian. This is because m data points can only span a subspace of feature space$\mathbb{R}^n$. If we model such a dataset using maximum likelihood estimator, we should have:

$$\begin{align}
\mu &= \frac{1}{m}\sum\limits_{i=1}^m x^i \\
\Sigma &= \frac{1}{m} \sum\limits_{i=1}^m (x^i-\mu)((x^i-\mu)^T)
\end{align}$$

Each $(x^i-\mu)((x^i-\mu)^T)$ produces a matrix with rank 1. The rank of the sum of all the matrices is the sum of the rank of each matrix. Thus, the final $\Sigma$ has the most rank m. If $n \gg m$, $\Sigma$ is a singular matrix, and its inverse does not exist. Furthermore, $1/\lvert \Sigma \rvert^{1/2} = 1/0$, which is invalid. This cannot be used to define the density of Gaussian distribution.

Thus, we will talk about how to find the best fit of model given the few amount of data.


# Restrictions of $\Sigma$

If we do not have sufficient data to fit a model, we might want to place some restrictions on $\Sigma$ so that it can be a valid covariance matrix.

The first restriction is to force the covariance matrix to be diagonal. In this setting, we should have our covariance matrix as:

$$\Sigma_{jj} = \frac{1}{m} \sum\limits_{i=1}^m (x_j^i - \mu_j)^2$$

Off-diagonals are just zero.

The second type of restriction is to further force the covariance matrix to the diagonal matrix where all the diagonals are equal. In general, we have $\Sigma = \sigma^2 I$ where $\sigma^2$ is the control parameter.

It can also be found using maximum likelihood as:

$$\sigma^2 = \frac{1}{mn} \sum\limits_{j=1}^n\sum\limits_{i=1}^m (x_j^i - \mu_j)^2$$

If we have a 2D Gaussian and plot it, we should see a contours that are circles.

To see why this helps, if we model a full, unconstrained covariance matrix, it was necessary (not sufficient) that $m\geq n$ in order to make $\Sigma$ non-singular. On the other hand, either of the two restriction above will produce a non-singular matrix $\Sigma$ when $m\geq 2$.

However, both restrictions have the same issue. That is, we cannot model the correlation and dependence between any pair of features in the covariance matrix because they are forced to be zero. So We cannot capture any correlation between any pair of features, which is bad. 

# Marginals and Conditions of Gaussian

Before talking about factor analysis, we want to talk about how to find conditional and marginal distributions of multivariate Gaussian variables. 

Suppose we have a vector-valued random variable:

$$x = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$$

where $x_1\in \mathbb{R}^r, x_2\in$
