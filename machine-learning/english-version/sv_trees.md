---
layout: single
mathjax: true
toc: true
toc_sticky: true
category: Machine Learning
tags: [notes]
qr: machine_learning_notes.png
title: Decision Trees
share: true
permalink: /MachineLearning/sv_trees/
sidebar:
  nav: "MachineLearning"
---



# Introduction

Decision tree is one of the most popular non-linear framework used in the reality. It is known for its robustness to noisy and the capability of learning disjunctive expressions. In reality, decision tree has been widely used in credit risk of loan applicants.

A decision tree maps input $x\in \mathbb{R}^d$ to output y using binary rules. From top-down point of view, each node in the tree has a splitting rule. At the very bottom, each leaf node outputs an value or a class. Note, outputs can repeat. Each splitting rule can be characterized as :

$$h(x) = \mathbb{1}[x_j > t]$$

for some dimension j and $t\in \mathbb{R}$. From the leaf nodes, we can know the predictions. An example of such a decision tree can be visualized below:


![Decision Tree Intuition](https://raw.githubusercontent.com/Wei2624/AI_Learning_Hub/master/machine-learning/images/cs229_trees_1.png)

# Decision Trees Types

Similar to traditional prediction models, decision trees can be grouped as classification trees and regression trees. Classification trees are to classify input, while regression trees are to regress a real value in the domain as output.

## Regression Trees

In this case, you can imagine that we are partitioning the space so that we can make a prediction. As an example, we have a 2 dimensional input space. We can partition each dimension individually and give a regressed value for certain region. You can visualize the idea in the left figure below.

![Regression Trees](https://raw.githubusercontent.com/Wei2624/AI_Learning_Hub/master/machine-learning/images/cs229_trees_2.png)

It turns out to be a tree structure such as the right one in the figure above. To predict, we assign $R_1,R_2,R_3,R_4,R_5$ to their corresponding path. In 3D space, we can see the above regression trees can lead to a stair-like partition in 3D.

![Regression Trees in 3D](https://raw.githubusercontent.com/Wei2624/AI_Learning_Hub/master/machine-learning/images/cs229_trees_3.png)

## Classification Trees

Let's see an example of this type of decision tree. Say we have two features $x_1,x_2$ as input and 3 class labels as output. Formally, $x \in \mathbb{R}^2$ and $y \in \{1,2,3\}$. Figuratively, we can see:

![Classifcation Trees of Data](https://raw.githubusercontent.com/Wei2624/AI_Learning_Hub/master/machine-learning/images/cs229_trees_4.png)

Now, we can start from the first feature. Say we choose 1.7 as the threshold value to split. As a result,we can have:

![Classifcation Trees, First Split](https://raw.githubusercontent.com/Wei2624/AI_Learning_Hub/master/machine-learning/images/cs229_trees_5.png)

The resulted decision tree can be described as:

![Classifcation Trees, First Split, Result](https://raw.githubusercontent.com/Wei2624/AI_Learning_Hub/master/machine-learning/images/cs229_trees_6.png)

We can perform a similar action to the second feature of input. In particular, we select another threshold value along second feature space, which results in:

![Classifcation Trees, Second Split](https://raw.githubusercontent.com/Wei2624/AI_Learning_Hub/master/machine-learning/images/cs229_trees_7.png)

The resulted tree can be visualized as:

![Classifcation Trees, Second Split, Results](https://raw.githubusercontent.com/Wei2624/AI_Learning_Hub/master/machine-learning/images/cs229_trees_8.png)

The steps above show the work flow of constructing a classification decision tree from the input space.

# Learning Decision Tree

In this section, we are talking about how to learn a decision tree for both types. In general, learning trees is using a top-down greedy algorithm. In this algorithm, we start from a single node. Then, we find out the threshold value which can reduce the uncertainty the most. We keep doing this until all rules are found out.

## Learning a Regression Tree

Back to the example:

![Regression Trees](https://raw.githubusercontent.com/Wei2624/AI_Learning_Hub/master/machine-learning/images/cs229_trees_2.png)

In the left figure, we have five regions, two input features and four thresholds. Let's generalize to M regions $R_1,\dots,R_M$. The prediction function can be:

$$f(x) = \sum\limits_{m=1}^M c_m \mathbb{1} \{x\in R_m \}$$

where $R_m$ is the region that x belongs to and $c_m$ is the prediction value.

The goal is to minimize:

$$\sum\limits_i (y_i - f(x_i))^2$$

Let's look at this definition. There are two variables that need to be determined, $c_m, R_m$. The prediction is from $c_m$. So if we are given $R_m$, can we make the prediction of $c_m$ easier? The answer is yes. We can simply average all the samples in the region as $c_m$. Now, the question is how do we find out the regions?

Let's consider to split a reigon R at the splitting value s of dimension j. The intial region is usually the entire dataset. We can define $R^{-}(j,s) = \{ x_i\in\mathbb{R}^d\lvert x_i(j) \leq s \}$ and $R^{+}(j,s) = \{ x_i\in\mathbb{R}^d\lvert x_i(j) \geq s \}$. Then, for each dimension j, we calculate the splitting point s that can achieve the goal the best. We should do this for each existing regirons(leaf node) so far and pick up the best region splitting based on a predefined metric.

**In a word, we always select a region (leafnode), then a feature and then a threshold to form a new split.**

## Learning a classification decision tree.

In regression task, we use a square error to determine the quality of splitting rules. In classification task, we have more options to evaluate the quality.

Overall, there are three common measurements for classification task in growing a tree.

1, Classification error: $1 - \max_k p_k$

2, Gini Index: $1 - \sum_k p_k^2$

3, Entropy: $-\sum_k p_k \ln p_k$

where $p_k$ essentially represents the empirical portion of each class. In this case, k means the class index. For a binary classification, if we plot the value of each evaluation with respect to $p_k$, we can have:

![Evaluation Plot](https://raw.githubusercontent.com/Wei2624/AI_Learning_Hub/master/machine-learning/images/cs229_trees_9.png)

It shows that

1, all evaluations are maximized when $p_k$ is uniform on the K classes in $R_m$.

2, all evaluations are minimized when $p_k=1$ or $p_k = 0$ for some k.

Let's see an example of growing a classification tree. Let's imagine we have a 2D space where some classified points are plotted. Such a plot can be show below.

![First Split Example](https://raw.githubusercontent.com/Wei2624/AI_Learning_Hub/master/machine-learning/images/cs229_trees_10.png)

In this case, the left region $R_1$ is classified as label 1. We can see that it is classified perfectly. Thus, the measurement on this region should be perfect.

For region 2, we need to do more since Gini index is not zero. If we calculate the Gini index, we can find out:

$$G(R_2) = 1 - (\frac{1}{101})^2 - (\frac{50}{101})^2 - (\frac{50}{101})^2 = 0.5089$$

Next, we want to see how break points at different position along different axis can affect the Gini index in this region based on some evaluation function. Such a evaluation function, a.k.a. uncertainty, can be:

$$G(R_m) - (p_{r_m^-}G(R_m^-) + p_{r_m^+}G(R_m^+))$$

where $p_{R_m^+}$ is the fraction of data in $R_m$ split into $R_m^+$ and $G(R_m^+)$ is the Gina index for the new region $R_m^+$. Essentially, we want the Gini index to be zero. So we want to deduct the Gini index resulted from splitting the region furthermore. Thus, we want to plot the reduction amount on the Gini index as the function of different splitting point.

For the above example, we first look at different splitting points by sliding along with horizontal axis.

![Uncertainty Plot](https://raw.githubusercontent.com/Wei2624/AI_Learning_Hub/master/machine-learning/images/cs229_trees_11.png)

You see that there are two clear cuts on both sides because points less than approximately 1.7 belong to class 1 and no point appears after approximately 2.9. We can also experiment another setting by sliding along another axis, namely vertical axis.

![Uncertainty Plot 2](https://raw.githubusercontent.com/Wei2624/AI_Learning_Hub/master/machine-learning/images/cs229_trees_12.png)

As we can see from the graph, we have the largest improvement around 2.7 as a vertical splitting point. Then, we can split our data samples as:

![Uncertainty Plot 2](https://raw.githubusercontent.com/Wei2624/AI_Learning_Hub/master/machine-learning/images/cs229_trees_13.png)

The resulted decision tree should be like:

![Uncertainty Plot 2](https://raw.githubusercontent.com/Wei2624/AI_Learning_Hub/master/machine-learning/images/cs229_trees_14.png)

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
