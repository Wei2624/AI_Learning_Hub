---
layout: single
mathjax: true
toc: true
toc_sticky: true
category: Machine Learning
tags: [notes]
qr: machine_learning_notes.png
title: Regularization and Model Selection
share: true
permalink: /MachineLearning/sv_regularization_model_selection/
sidebar:
  nav: "MachineLearning"
---


In model selection, if we have k parameters in the model, the quesiton is what k should be?0,1,or 10?Which does one of them give the best bias-varaince tradeoff. In particular, we use a finite set of models $\mathcal{M} = \{M_1,M_2,\dots,M_d\}$ from which we try to select the best. Each model in the set contains either different parameterization of a particular model or different models. 

# 1 Cross Validation

Imagine that given a dataset S and a set of models, it is easy to think to select a model out of the set by:

1 Training each model $M_i$  from S and get the hypothesis $h_i$.

2 Pick the hypothesis with the smallest training error. 

This pipeline does not work simply because the higher order of the polynomial you choose, the better it will fit for the training set. However, the model you select will have a high generalizaton error in a new dataset. That is, it will be high variance.

In this scenario, **hold-out cross validation** will do a better work as:

1 Randomly split S into training set $S_{tr}$ and validation set $S_{cv}$ with 70% and 30% respectively

2 Train each $M_i$ on $S_{tr}$ to get hypothesis $h_i$

3 Select the hypothesis which has the smallest epirical error on the $S_{cv}$, which denotes $\hat{\varepsilon}\_{S_{cv}}(h_i)$

By doing the above, we try to estimate the real generalization error by testing the model on validation set. In step 3, after selecting the best model, we can retrain the model on the entire dataset again to generate the best hypothesis. However, even though that's the case, we still select the model based on 70% dataset. This is bad when data is scarce. 

Thus, we introduce the K-fold corss validation as:

1 Randomly split S into k disjoint subsets of m/k samples each. Denote $S_1,S_2,\dots,S_k$

2 For each model $M_i$, we unselect one of subsets dentoed j, and train the model on the rest of data to get the hypothesis $H_{ij}$. We test the hypothesis on $S_j$ and get $\varepsilon_{S_j}(h_{ij})$. We do this for all j. And lastly, we take average the generalization error over j.

3 We select the model with the smallest averaged generalization error. 

A typical choice for k is 10. This is computationally expensive although it gives the best performance. If the data is scarce, we might set k=m. In this case, we leave one sample at a time. We call it **leave-ont-out cross validation**. 

# 2 Feature Selection

If we have n features and m samples where $n \gg m$ (VC dimension is O(n)), we might have overfitting. In this case, you might want to select some of features which might be the most important. In brute force algorithm, we can have $2^n$ different combinations of feature setting. We can perform model selection over all $2^n$ possible models. This is too expensive to deal with. Thus, we have an option called **forward search** algorithm:

1 We initialize $\mathcal{F} = \emptyset$

2 Repeat: (a)for $i =1,\dots,n$ if $i\notin\mathcal{F}$, let $\mathcal{F}_i = \mathcal{F}\cup\{i\}$ and use some corss validation algorithm to evaluate $\mathcal{F}_i$. (b)Set $\mathcal{F}$ to be the best feature subset from (a)

3 Select the best feature subset from the above. 

You can terminate the loop by setting the number of features you like to have. In contrast, we can also have **backward search** in for feature selection, which is similar wtih the section of **Ablative Analysis**. However, both of them are computationally expensive since it requires $O(n^2)$ in time complexity. 

Instead, we can use **Filter feature selection** heristically. The idea is to give a score to how informative each feature is with respect to labels y. Then we pick the best out of it. 

One intuitive option of the sorce is to compute the correlation between each feature $x_i$ and y. In practice, we set the score to be **mutual information** as:

$$MI(x_i,y) = \sum\limits_{x_i\in\{0,1\}}\sum\limits_{y\in\{0,1\}} p(x_i,y)\log\frac{p(x_i,y)}{p(x_i)p(y)}$$

where we assume each feature and label is binary-valued and the summation is over the domain of the varaibles. Each probability can be calculated empirically from the training dataset. To understand this, we know that:

$$MI(x_i,y) = KL(p(x_i,y)\lvert\lvert p(x_i)p(y))$$

where KL is **Kullback-Leibler divergence**. It simply measures how different the probability distributions from both sides of the two bars are. If $x_i$ and $y$ are independent, then KL is 0. That means there is no relationship between this feature and labels. In contrast, if we have a high score of MI, then such a feature is strongly correlated with labels. 

# 3 Bayesian Statistics and regularization

In the previous section, we talk about the maximum likelihood (ML) algorithm to fit model parameters as:

$$\theta_{ML} = \arg\max\prod_{i=1}^m p(y^{(i)}\lvert x^{(i)},\theta)$$

In this case, we viewed $\theta$ as a unknown parameter. It already exists there and just happens to be unknown. So our job is to find the unknown or estimate it. 

On the other hand, we can have a Bayesian view of this goal. We think the unknown parameter $\theta$ is also random. Thus, we place our prior belief on this parameter. We call it **prior distribution**. Given the prior distribution, we can calculate the posterior with dataset S as :

$$p(\theta\lvert S) = \frac{p(S\lvert\theta)p(\theta)}{p(S)} = \frac{\prod_{i=1}^m p(y^{(i)}\lvert x^{(i)},\theta)(p(\theta)}{\int_{\theta}\prod_{i=1}^m p(y^{(i)}\lvert x^{(i)},\theta)(p(\theta)d\theta}$$

For prediciton inference by using the posterior, we have:

$$p(y\lvert x,S) = \int_{\theta}p(y\lvert x,\theta)p(\theta\lvert S)d\theta$$

At this point, we can calculate the conditional expected value y. However, it is really hard to calculate the posterior in closed form since the intergral in the denominator cannot be solve in closed form. Thus, alternatively, we seek for a point estimate for the posterior at which it will give us one best $\theta$ for the posterior. The **MAP(maximum a posteriori)** can estimate it by:

$$\theta_{MAP} = \arg\max_{\theta} = \prod_{i=1}^m p(y^{(i)}\lvert x^{(i)},\theta)p(\theta)$$

In general, the prior is usually zero mean and unit variance. This will make MAP less susceptiable overfitting than the ML estimate of the parameters. 