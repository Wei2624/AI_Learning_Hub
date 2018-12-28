---
layout: single
mathjax: true
toc: true
toc_sticky: true
category: Machine Learning
tags: [notes]
qr: machine_learning_notes.png
title: Variational Inference
share: true
permalink: /MachineLearning/bayes_vi/
sidebar:
  nav: "MachineLearning"
---

# Review of EM algorithm

In [last section](https://wei2624.github.io/MachineLearning/usv_em/), we talked about how EM algorithm works and why it works in general. The idea is to find a proper latent varibale  which can be integrated out to get back to joint distribution. The latent variable comes in to facilitate the math challenges when we are trying to find the optimal parameters for joint distributions. With the latent variable, we can find an point estimate of parameters which result in the highest joint likelihood. 

However, EM does not always work. When EM does not work, VI comes into play to solve the issue. We will see some exapmples where EM does not work and build the transition from EM to VI. 

# Model Setup

In this section, we will see different model setups and how EM fails in which stage. 

## Model V1

In this version of setup, we have the simplest model setup. Formally, we have:

$$y_i \sim \mathcal{N}(x_i^Tw,\alpha), w\sim\mathcal{N}(0,\lambda I)$$

We assume each sample is iid and dimensions are matched. There is no prior distribution on x. Thus, we are in discriminative setting not generative setting where we have the prior model on x. 

We can calculate the posterior on w using Bayes rule:

$$p(w\lvert x,y)=\frac{p(y\lvert w,x)p(w)}{p(y\lvert x)} = \mathcal{N}(\mu,\Sigma)$$

This does not require any iterative algrotihm and is very easy to calculate in the closed form. However, we notice that there are two parameters that we need to pre-define: $\alpha,\lambda$. These parameters can affect the final performance. In general, there are two ways to work on it:

1 We can use cross-validation on these two parameters and select the one witht the best performance. 

2 In Bayesian, we can put a prior distribution on the parameters and learn it!

## Model V2

So we want to pick up a conjugate gamma propr on $\alpha$. Then, we have:

$$y_i \sim \mathcal{N}(x_i^Tw,\alpha), w\sim\mathcal{N}(0,\lambda I), \alpha\sim Gamma(a,b)$$

In this case, $alpha$ becomes another model variable that we want to learn like w. We can do one the following:

1 Try to learn the full posterior on $\alpha$ and w (We cannot assume w and $\alpha$ are independent). 

2 Learn a point estimate of w and $\alpha$ based on MAP inference. 

3 Integrate out some variables to learn the others. 

Let's look at each one:

1 The full posterior of $\alpha$ and w is:

$$ p(w,\alpha\lvert y,x) = \frac{p(y\lvert w, \alpha, x)p(w)p(\alpha)}{\int\int p(y\lvert w, \alpha, x)p(w)p(\alpha) dwd\alpha}$$

We realize the normalizing constant cannot be caluclated in the closed form. We can use other approximation method to approximate it such as Laplace or MCMC sampling. 

2 We can do MAP over w and $\alpha$:

$$w,\alpha = \arg\max_{w,\alpha} \ln  p(y,w,\alpha\lvert x)$$

$$=\arg\max_{w,\alpha} \ln p(y\lvert w,\alpha,x) + \ln p(w) + \ln p(\alpha)$$

We can use coordinate descent algorithm to optimize each parameter. However, this will not tell us about uncertainty. Rather, it will only tell us an point estimate of the model parameters. 

3 We can do marginal likelihood as:

$$p(y\lvert w x) = \int p(y,w, \alpha\lvert x) d\alpha$$

With this marginal likelihood, we can:

(1) try to find posterior inference for $p(w\lvert y,x)\propto p(y\lvert w,x)p(w)$. However, this will not work out since $p(y\lvert w,x)$ is resulted from the integral of $\alpha$, which is not a Gaussian anymore. Rather, it is student-t distribution. This is not conjugate with prior anymore. So it does not work. 

(2) Another option is to maximize $p(y,w\lvert x)$ over w using MAP using gradient method. This is where EM could possibly come in. 

## EM for Model V2

So we want to treat $\alpha$ as the latent variable. We are trying to find a point estimate of w to maximize the marginal distribution $p(y,w\lvert x)=\int p(y,w,\alpha\lvert x)\alpha$. 

Based on the above discussion, we can write EM master equation:

$$\ln p(y,w\lvert x)=\underbrace{\int q(\alpha)\ln\frac{p(y,w,\alpha\lvert x)}{q(\alpha)}d\alpha}_{\mathcal{L}(w)} + \underbrace{\int q(\alpha)\ln\frac{q(\alpha)}{p(\alpha\lvert y,w,x)}d\alpha}_{KL(p\lvert\lvert q)}$$

E-step:

$$\begin{align}
p(\alpha\lvert y,w,x) &\propto \prod_{i=1}^N p(y_i\lvert \alpha.w,x_i)p(\alpha) \\
&\propto \alpha^{\frac{N}{2}}\exp(-\frac{\alpha}{2}\sum\limits_{i=1}^N (y_i-x_i^Tw)^2)\times \alpha^{a-1}\exp(-b\alpha) \\
&=Gamma(a+\frac{N}{2},b+\frac{1}{2}\sum\limits_{i=1}^N (y_i-x_i^Tw)^2)
\end{align}$$ 

Now, we should set $q_t(\alpha) = p(\alpha\lvert y,w_{t-1},x)$, and then calculate the loss function:

$$\begin{align}
\mathcal{L}(w) &= \mathbb{E}_q [\ln p(y,\alpha\lvert w,x)p(w)] + \mathbb{E}_q [\ln q(\alpha)] \\
&= -\frac{\mathbb{E}_{q_t} [\alpha]}{2} \sum\limits_{i=1}^N (y_i -x_i^Tw)^2 - \frac{\lambda}{2}w^Tw + \text{const. w.r.t. w}
\end{align}$$

M-step:

Then, we have to make sure that we can maximize the loss function in a closed form. Otherwise, there is no point of doing this. We can try and find (I will give the math later):

$$w_t = (\lambda I + \mathbb{E}_{q_t} [\alpha]\sum\limits_{i=1}^N x_i x_i^T)^{-1}(\sum\limits_{i=1}^N \mathbb{E}_{q_t}[\alpha]y_i x_i)$$

We can plug $\mathbb{E}_{q_t}[\alpha]$ into the above so that we get the updating form of w. 


Note that there are a few things that are different than we did EM last time.

(1) In the previous case, we introduced a latent variable to reduce the math complexity. However, in this case, the latent variable $\alpha$ has the interpretation, which is related to the observation noise. So this latent varibale has its own meaning. 

(2) In this case, we have w and $\alpha$ to learn. However, we make a compromise by learning a point estimate and a conditional posterior of $\alpha$. It is hard for us to learn point estimates of two variables. Surly, you could have done the reverse. That is, you can learn a point estimate of $\alpha$ and a conditional posterior of w. 


## Model V3

The EM algorithm works in model version 2. Let's see a new version of model, namely version 3. In this setup, we have one more variable to work with. 

$$ y_i \sim \mathcal{N}(x_i^Tw, \alpha^{-1}), w\sim \mathcal{N}(0,\lambda I)$$

$$\alpha\sim Gamma(a,b),\lambda\sim Gamma(e,f)$$

Then, the marginal distribution is:

$$p(y,w\lvert x) = \int\int p(y,w,\alpha,\lambda\lvert x)d\alpha d\lambda$$

## EM for Model V3

Note that we have two latent variables now instead of one in V2. The EM master equation is:

$$\begin{align}
\ln p(y,w\lvert x) &= \int\int q(\alpha,\lambda)\ln\frac{p(y,w,\alpha,\lambda\lvert x)}{q(\alpha,\lambda)}d\alpha d\lambda + \\
& \int\int q(\alpha,\lambda)\ln\frac{q(\alpha,\lambda)}{p(\alpha,\lambda\lvert y,w,x)}
\end{align}$$

Note that $\ln p(y,w\lvert x)$ in this model setup is different from the one in V2, although the form is the same. That is, the underlying distribution is essentially different. 

E-step:

As usual, we have:

$$\begin{align}
p(\alpha,\lambda\lvert y,w,x) &= \frac{p(y\lvert w,\alpha,x)p(\alpha)p(w\lvert\lambda)p(\lambda)}{\int\int p(y\lvert w,\alpha,x)p(\alpha)p(w\lvert\lambda)p(\lambda) d\lambda d\alpha} \\
&= \underbrace{\frac{p(y,\alpha\lvert w,x)}{\int p(y,\alpha\lvert w,x)d\alpha}}_{p(\alpha\lvert y,w,x)}\underbrace{\frac{p(w,\lambda)}{\int p(w,\lambda)d\lambda}}_{p(\lambda\lvert w)}
\end{align}$$

We can calculate the conditional posterior of both parameters:

$$p(\alpha\lvert,y,w,x) = Gamma(a+\frac{N}{2},b+\frac{1}{2}\sum\limits_{i=1}^N (y_i-x_i^Tw)^2)$$

$$p(\lambda\lvert w) = Gamma(e+\frac{d}{2},f+\frac{1}{2}w^Tw)$$

M-step:

We have the same form for the updating as:

$$w_t = (\mathbb{E}_{q_t}[\lambda] I + \mathbb{E}_{q_t} [\alpha]\sum\limits_{i=1}^N x_i x_i^T)^{-1}(\sum\limits_{i=1}^N \mathbb{E}_{q_t}[\alpha]y_i x_i)$$

The only difference is that we have the expectation for $\lambda$ as well. This worked out since we can factorize the full conditional posteriors of the two variables as:

$$q(\alpha,\lambda) = p(\alpha,\lambda\lvert y,w,x) = p(\alpha\lvert y,w,x)p(\lambda\lvert w) = q(\alpha)q(\lambda)$$

This is not always true. In this case, we just make it factorizeable. Eventually, we will have a point estimate and full posterior of variables. 

The question is that what if we cannot factorize or find the posterior of the variables. In this case, we have the problem. If we cannot calculate or factorize the posterior, we cannot move on with the EM algorithm. This is where VI might come in. 

# From EM to VI

In EM, we can learn posteriors of all variables except for the one that is learned as point estimate. So let's make it learn all variables in posteriors. For model V3, we can have:

$$p(y\lvert x) = \int\int\int p(y,w,\alpha,\lambda) d\alpha d\lambda dw$$

Note that different from EM, there is nothing to optimize on the left hand side. We can further write:

$$\begin{align}
\ln p(y\lvert x) &= \int\int\int q(\alpha,\lambda,w)\ln \frac{p(y,w,\alpha,\lambda\lvert x)}{q(\alpha,\lambda,w)}d \alpha d\lambda dw + \\
& \int\int\int q(\alpha,\lambda,w)\ln \frac{q(\alpha,\lambda,w)}{p(\alpha,\lambda,w\lvert y,x)}d \alpha d\lambda dw
\end{align}$$

Again, there is nothing to optimize on the left hand side. In addition, we cannot complete our E-step since we cannot calculate the full posterior:

$$q(\alpha,\lambda, w) = p(\alpha,\lambda, w\lvert y,x)$$

So let's look at the VI equation:

$$\begin{align}
\ln p(y\lvert x) &= \int\int\int q(\alpha,\lambda,w)\ln \frac{p(y,w,\alpha,\lambda\lvert x)}{q(\alpha,\lambda,w)}d \alpha d\lambda dw + \\
& \int\int\int q(\alpha,\lambda,w)\ln \frac{q(\alpha,\lambda,w)}{p(\alpha,\lambda,w\lvert y,x)}d \alpha d\lambda dw
\end{align}$$

In EM, we want to have a proper q distribution to find out an point estimate of w. Both left and right hand side of EM equation are optimized. In VI, there is nothing to optimize on the left hand side. Thus, in VI, we view this equation differently. In particular, we are more interested in q distributions and how to pick them up. 

Let's look at each term in the VI equation individually. 

(1) $\ln p(y\lvert x)$: This is the marginal likelihood of the data given the model. We cannot find this. If we can, we can then find:

$$p(\alpha,\lambda,w\lvert y,x)=\frac{p(y,\alpha,w,\lambda\lvert x)}{p(y\lvert x)}$$

So we cannot explicitly calculate $p(y\lvert x)$. However, given data and model, we know that this value is always constant no matter how we change the model variables. 

(2) $KL(p\lvert\rvert q): Again, we do not know how to calcuate this KL. However, we do know that this is a non-negative number and equals to zero iff p = q. 

(3) Objective function: In this term, we pick up a good q so that we can calculate this term by completing the integral. 

The goal of VI is to pick a good q that can be as close as possible to the posterior distribution $p(w,\alpha,\lambda\lvert y,x)$. There are some other ways to do it. For example, we can use Laplace approximation. However, in this case, we are more interested in inference algorithm. 

To measure the closeness, we can keep using KL divergence as the tool. In other words, we want to minimize the KL value. It is clear that we cannot caluclate the KL directly. However, we know that the left hand side of VI equation is constant, and KL is a non-negative number. Thus, to minimize KL, on the other hand, we can maximize the VI objective function, which is the middle term in the VI equation. This also shows that in VI, we care about the q distribution not the latent variable in EM. 

## VI for Model V3 

Let's step back to the model setup version 3 to see how VI works there. In this case, we need to:

(1) define q distribution family over $\alpha, w, \lambda$. The parameters of q are supposed to be optimized. Note that in EM, we find q such that the marginal likelihood is correct. 

(2) construct variational objective function:

$$\mathcal{L} (\theta) = \mathbb{E}_q [\ln p(y,w,\lambda,\alpha\lvert x)] - \mathbb{E}_q [\ln q(\alpha,\lambda,w)]$$

where $\theta$ is the symbol of all parameters that are to be optimized. In this case, there is no $w,\alpha,\lambda$ by the end since it will be integrated out. 

In EM, we have different story where the objective function $\mathcal{L}$ is the function over model variable. We can do MAP inference or ML inference on it depending if we have the prior of model variable. Also, the parameters of q distribution on the latent variable in EM are always known since we force them to some certain values. On the other hand, in VI, the parameters of q distributions are unknown and instead they are the parameters in the objective function that we are trying to optimize. The key point here is that although EM and VI have the similar form of objective function, they are completely different things in this case. 

Now, we need to define q distributions. In general, we use something called "mean-field" assumption. In this approach, we can have:

$$q(\alpha,w,\lambda) = q(\alpha)q(w)q(\lambda)$$

Then, we can handle them one at a time. Note that if w is a vector, we can also write $q(w)=\prod_j q(w_j)$.

We will talk about how to define the best optimal one for each of them. The algorithm should still work no matter what distributions we define. In this case, we can define them the same as their priors. 

$$q(\alpha) = Gamma(a^{\prime},b^{\prime}), q(\lambda) = Gamma(e^{\prime},f^{\prime}), q(w) = Normal(\mu^{\prime},\Sigma^{\prime})$$

We optimize them to be the best values that can make q to be as close as possible to the full posterior by optimizing the variational objective function. 

The reason that we are doing the factorization is that it is hard to find a good distribution family for multiple parameters that have different support. For example, $\alpha$ is only valid in postive real number while w is valid in the whole real number domain. How do we define a distribution that take both as support? So it is better to consider them one by one. 

Recall that in model setup version v2, we had $\lambda$ and $\alpha$ conditionally independent given w, which happened to let us write $q(\alpha,\lambda) = q(\alpha)q(lambda)$. It is no longer valid now since we use mean-field assumption. That means:

$$q(\alpha)q(w)q(\lambda) \neq p(\alpha,\lambda,w\lvert y,x)$$

It indicates that KL is always larger than zero in VI.

Then, we should calulcate the objective function:

$$\begin{align}
\mathcal{L} &= \mathbb{E}_q [\ln p(y,w,\alpha,\lambda\lvert x)] - \mathbb{E}_q [\ln q(\alpha,\lambda,w)] \\
& = \frac{\mathbb{E}_q [\alpha]}{2}\sum\limits_{i=1}^N \mathbb{E}_q [(y_i-x_i^Tw)^2] + \frac{1}{2}\mathbb{E}_q [\ln \alpha] - \frac{\mathbb{E}_q[\lambda]}{2}\mathbb{E}_q[w^Tw] \\
& + \frac{d}{2}\mathbb{E}_q [\ln\lambda] + (a-1)\mathbb{E}_q[\ln\alpha] - b\mathbb{E}_q[\alpha] + (e-1)\mathbb{E}_q[\ln\lambda] \\
& -f\mathbb{E}_q [\lambda]-\mathbb{E}_q [\ln q(\alpha)] -\mathbb{E}_q [\ln q(\lambda)] - \mathbb{E}_q [\ln q(w)] + \text{const.}
\end{align}$$

You can see how complicated this is. Note that the independent assumption that we made has helped us a lot in this case since each expectation is only w.r.t. the variable of itnerest. So we can move expectation inwards to make calculation easier. 

The next step is to take derivative w.r.t. $a^{\prime},b^{\prime},e^{\prime},f^{\prime}, \mu^{\prime},\Sigma^{\prime}$ and use graident descent to optimize them one at a time. 

# Variational Inference

We will talk about variational inference formally. Let's define the model variables to be $\theta_1,\theta_2,\dots,\theta_m$ and parameters for those variables to be $\psi_1,\psi_2,\dots,\psi_m$.

## Mean-field Assumption

This is an important assumption to make variational inference work. In mean-field assumption, we split $\theta$ into m different groups. Note that each component can be in different space. Each q distribution can be written as $q(\theta_i\lvert\psi_i)$. So all the $\psi_i$ are the parameters that we need to learn from variational objective function. 

In mean-field assumption, we assume that:

$$q(\theta_1,\theta_2,\dots,\theta_m) = \prod_{i=1}^m q(\theta_i\lvert\psi_i)$$

can approximate the posterior:

$$p(\theta_1,\theta_2,\dots,\theta_m\lvert X)$$

## Variational objective function

With this mean-field definition, we can write variational objective function as:

$$\begin{align}
\mathcal{L} &= \int (\prod_{i=1}^m q(\theta_i\lvert\psi_i))\ln p(X, \theta_1,\dots,\theta_m) d\theta_1\dots d\theta_m \\
& - \sum\limits_{i=1}^m \int q(\theta_i\lvert\psi_i)\ln q(\theta_i\lvert\psi_i) d\theta_i
\end{align}$$

We are trying to optimize all the $\psi_i$. There are two ways to make variational inference work out: direct method and optimal method. We will see how direct method works and appareciate the effectiveness of optimal method. Before that, we need to figure out what distribution family we should pick up for each q distribution. 

## Picking $q(\theta_i\lvert\psi_i)$

Essentially, we are facing with two key problems here:

1 How do we choose each $q_i = q(\theta_i\lvert\psi_i)$?

2 How do we get the best $\psi_i$ for each i. 

**Setup**: we have a joint likelihood $p(X,\theta_1,\dots,\theta_m)$ and the mean-field assumption says that $p(\theta_1,\dots,\theta_m) \approx \prod_{i=1}^m q(\theta_i\lvert\psi_i)$

Let's randomly pick $i\in [1,\dots,m]$ and correspondingly $q(\theta_i\lvert\psi_i)$. We expand our variational objective function as:

$$\begin{align}
\mathcal{L} &= \int q(\theta_i\lvert\psi_i) \underbrace{\int (\prod_{j\neq i} q(\theta_j\lvert\psi_j))\ln p(X,\theta_1,\dots,\theta_m)d\theta_{j\neq i}}_{\mathbb{E}_{q_{j\neq i}}[\ln p(X,\theta_1,\dots,\theta_m)]} d\theta_i \\
& -\int q(\theta_i\lvert\psi_i)\ln q(\theta_i\lvert\psi_i)d\theta_i - \sum\limits_{j\neq i} \int q(\theta_j\lvert\psi_j)\ln q(\theta_j\lvert\psi_j)d\theta_j
\end{align}$$

Since we only care about $\theta_i$ and $\psi_i$, we can treat the vartional objective as the function of these two parameters. Then, we can further write:

$$\begin{align}
\mathcal{L} &= \int q(\theta_i\lvert\psi_i)\mathbb{E}_{q_{j\neq i}}[\ln p(X,\theta_1,\dots,\theta_m)] d\theta_i \\
& -\int q(\theta_i\lvert\psi_i) \ln q(\theta_i\lvert\psi_j) d\theta_i + \text{const. w.r.t. } \theta_i \\
&= \int q(\theta_i\lvert\psi_i)\ln\frac{\exp (\mathbb{E}_{q_{j\neq i}}[\ln p(X,\theta_1,\dots,\theta_m)])}{q(\theta_i\lvert\psi_i)} d\theta_i \\
& + \text{const. w.r.t. }\theta_i
\end{align}$$

**Math:** From the first line to second line, we first merge two integrals. Then, we factor off $q(theta_i\lvert\psi_i)$. Last but not least, in order to use the property of ln function, we first exponential and take ln of joint log likelihood. 

Let's look at this variational objective function for a second. We are trying to figure out how we can pick up a proper $q(\theta_i\lvert\psi_i)$ so that the variational objective function can be maximized. For this goal, what we have done so far is taking only $q_i$ and re-write the variational objective function in terms of $q_i$. Thus, given that we only care about $q_i$, the goal turns out to be that how we can maximize the above variational objective function. 

If we look at this form of variational objective function, we can see that this somewhat looks like KL divergenceexcept that the numerator is not a distribution and the order of numerator and denominator is reversed. For the frist issue, if it is not a distribution, we can make it a distribution by normailizing it. For the second issue, we can put a negative multuplier on ln. By doing these two things, we make up a KL divergence. 

In particular, let's define:

$$Z = \int \exp (\mathbb{E}_{q_{j\neq i}}[\ln p(X,\theta_1,\dots,\theta_m)])d\theta_i$$

We can add and substract $\ln Z$ on the variational objective:

$$\begin{align}
\mathcal{L} &= \int q(\theta_i\lvert\psi_i)\ln\frac{\frac{1}{Z}\exp (\mathbb{E}_{q_{j\neq i}}[\ln p(X,\theta_1,\dots,\theta_m)])}{q(\theta_i\lvert\psi_i)} d\theta_i + \ln Z \\
& + \text{const. w.r.t. }\theta_i
\end{align}$$

where:

$$\frac{1}{Z}\exp (\mathbb{E}_{q_{j\neq i}}[\ln p(X,\theta_1,\dots,\theta_m)])$$

is a distribution of $\theta_i$. We can further write the objective as:

$$\int q(\theta_i\lvert\psi_i)\ln\frac{\frac{1}{Z}\exp(\mathbb{E}_{q_{j\neq i}}[\ln p(X,\theta_1,\dots,\theta_m)])}{q(\theta_i\lvert\psi_i)} d\theta_i = $$

$$-KL(q_i\lvert\rvert \frac{1}{Z}\exp (\mathbb{E}_{q_{j\neq i}}[\ln p(X,\theta_1,\dots,\theta_m)]))
\end{align}$$



When can we get the max value of the above variational objective function? The answer is when those two distributions are the same. Thus, for a particular i, we should set:

$$q(\theta_i\lvert\psi_i) = \frac{1}{Z}\exp (\mathbb{E}_{q_{j\neq i}}[\ln p(X,\theta_1,\dots,\theta_m)])$$

This gives us the optimal distribution family we should take no matter what parameters are. Since i is arbitrary, we can say this works for any i. In addition, when we can write out the full distribution of each i, we can also find out what the optimal parameters are for each i. This sovles our second question as well. 

On a high level, the general variational inference algorithm can be described as:

For iteration t:

1 For model variable index $i=1,\dots,m$, we set:

$$q(\theta_i\lvert\psi_i) = \frac{\exp(\mathbb{E}_{q_{j\neq i}}[\ln p(X,\theta_1,\dots,\theta_m)])}{\int \exp(\mathbb{E}_{q_{j\neq i}}[\ln p(X,\theta_1,\dots,\theta_m)])} d\theta_i$$

2 Evaluate the varational objective function using the updating q:

$$\mathcal{L}_t = \mathbb{E}_q [\ln p(X,\theta_1,\dots,\theta_m)] - \sum\limits_{i=1}^m \mathbb{E}_{q_i}[\ln q(\theta_i\lvert\psi_i)]$$

3 If the increase between $\mathcal{L}_t$ and $\mathcal{L}_{t-1}$ is below some threahold value, we terminate the process. 

Next, we look at two different methods to complete our variational inference. 

## Direct Method

In this method, we explicitly define each $q_i$ and optimize objective function. This is a more complciated method than optimal method. If you don't want to see the difference, you can skip tp optimal method. 

**Setup**: 

$$y_i\sim\mathcal{N}(x_i^Tw,\alpha),w\sim\mathcal{N}(0,\lambda I),\alpha\sim Gamma(a,b)$$

The joint likelihood:

$$p(y,w,\alpha\lvert x) = p(\alpha)p(w)\prod_{i=1}^N p(y_i\lvert x_i,w,\alpha)$$

We can approximate the full posterior by picking up:

$$q(\alpha,w) = q(\alpha)q(w) = Gamma(\alpha\lvert a^{\prime},,b^{\prime})\mathcal{N}(w\lvert\mu^{\prime},\Sigma^{\prime})$$

Then, we can write our varitional objective function as:

$$\begin{align}
\mathcal{L}(a^{\prime},b^{\prime},\mu^{\prime},\Sigma^{\prime}) &= \int_0^{\infty}\int_{\mathbb{R}^d} q(w,\alpha)\ln \frac{p(y,w,\alpha\lvert x)}{q(\alpha,w)}dwd\alpha \\
& = \int q(\alpha)\ln p(\alpha)d\alpha + \int q(w)\ln p(w)dw \\
& +\sum\limits_{i=1}^N \int\int q(\alpha)q(w)\ln p(y_i\lvert x_i,w,\alpha)dwd\alpha \\
& -\int q(\alpha)\ln q(\alpha)d\alpha - \int q(w)\ln q(w)dw
\end{align}$$

You can realize how complicated it has been and how complex it will be after we plug everyting in. 

$$\begin{align}
\mathcal{L}(a^{\prime},b^{\prime},\mu^{\prime},\Sigma^{\prime}) &= (a-1)(\psi(a^{\prime}) - \ln b^{\prime}) - b\frac{a^{\prime}}{b^{\prime}} + \text{const.} \\
& -\frac{\lambda}{2}(\mu^{\prime T}\mu^{\prime} + tr(\Sigma^{\prime})) + \text{const.} \\
& +\frac{N}{2}(\psi(a^{\prime})-\ln b^{\prime}) - \sum\limits_{i=1}^N \frac{a^{\prime}}{2b^{\prime}}((y_i - x_i^T\mu^{\prime})^2 \\
& + x_i^T\Sigma^{\prime}x_i) + \text{const.} \\
& +a^{\prime} - \ln b^{\prime} + \ln\Gamma(a^{\prime}) + (1 - a^{\prime})\psi(a^{\prime}) \\
& +\frac{1}{2}\ln\lvert\Sigma^{\prime}\lvert + \text{const.}
\end{align}$$

You can see how complicated it is. The next step is to take derivative of each variables that we are optimizing for. We can then use gradient descent algorithm for each optimization problem. 

However, as said, this is too complicated to handle with. In practice, it is more encouraged to use the optimal method for variational inference. 

## Optimal Method

**Setup**:

$$y_i\sim\mathcal{N}(x_i^Tw,\alpha),w\sim\mathcal{N}(0,\lambda I),\alpha\sim Gamma(a,b)$$

Based on the mean-field assumption, we can factorize the q as:

$$q(\alpha,w) = q(\alpha)q(w) \approx p(\alpha,w\lvert y,x)$$

As indicated in our general variational procedure, we should then pick up the distribution family for each model variable.

For $q(\alpha)$:

$$\begin{align}
q(\alpha) &\propto \exp(\mathbb{E}_{q(w)}[\ln p(y\lvert x,\alpha,w) + \ln p(\alpha) + \ln p(w)]) \\
&\propto \exp(\mathbb{E}_{q(w)}[\ln p(y\lvert x,\alpha,w) + \ln p(\alpha)]) \\
&\propto \exp(\sum\limits_{i=1}^N\mathbb{E}_{q(w)}[\ln p(y_i\lvert x_i,\alpha,w)]) p(\alpha) \\
&\propto (\prod\limits_{i=1}^N \alpha^{\frac{1}{2}}\exp(-\frac{\alpha}{2}\mathbb{E}_{q(w)}[(y_i-x_i^Tw)^2]))\alpha^{a-1}\exp(-b\alpha)
\end{align}$$

Since we haven't calulated $q(w)$, we have to carry this over fot the calculation. However, we can recognize that:

$$\begin{align}
q(\alpha) &= Gamma(a+\frac{N}{2},b+\frac{1}{2}\sum\limits_{i=1}^N\mathbb{E}_{q(w)}[(y_i-x_i^Tw)^2]) \\
& = Gamma(a^{\prime},b^{\prime})
\end{align}$$

For $q(w)$:

$$\begin{align}
q(w) &\propto \exp(\mathbb{E}_{q(\alpha)}[\ln p(y\lvert x,\alpha,w) + \ln p(\alpha) + \ln p(w)]) \\
&\propto \exp(\mathbb{E}_{q(\alpha)}[\ln p(y\lvert x,\alpha,w) + \ln p(w)])  \\
&\propto \exp(\sum\limits_{i=1}^N \mathbb{E}_{q(\alpha)}[\ln p(y_i\lvert x_i,\alpha,w)])p(w) \\
&\propto (\prod_{i=1}^N \exp(\frac{1}{2}\mathbb{E}[\ln\alpha])\exp(-\frac{\mathbb{E}_{q(\alpha)}}{2}(y_i -x_i^Tw)^2))\exp(-\frac{\lambda}{2}w^Tw)
\end{align}$$

So again we just carry over $q(\alpha)$ for now. We realize that:

$$\begin{align}
q(w) &= Normal(\Sigma^{\prime}(\mathbb{E}_{q(\alpha)}[\alpha]\sum\limits_{i=1}^Ny_i x_i),\lambda I+\mathbb{E}_{q(\alpha)}[\alpha]\sum\limits_{i=1}^N x_i x_i^T) \\
&= Normal(\mu^{\prime},\Sigma^{\prime})
\end{align}$$

We now know $q(w)$ is a Gaussian and $q(\alpha)$ is a Gamma. We can plug the respective expectation back to the formula we found and solve for the final expression. 

By looking up online, we can find out that:

$$\mathbb{E}_{q(\alpha)}[\alpha] = \frac{a^{\prime}}{b^{\prime}}$$

$$\mathbb{E}_{q(w)}[(y_i-x_i^Tw)^2] = (y_i - x_i^T\mu^{\prime})^2 + x_i^T\Sigma^{\prime}x_i$$

Put them all together:

**VI for Bayesian Linear regression with unknown noise precision**

1 Initialize $a_0^{\prime},b_0^{\prime},\mu_0^{\prime},\Sigma_0^{\prime}$ ranomly

2 For iteration $t = 1,\dots$:

- Update $q(\alpha)$ by setting:

$$a_t^{\prime} = a + \frac{N}{2}$$

$$b_t^{\prime} = b + \frac{1}{2}\sum\limits_{i=1}^N (y_i - x_i^T\mu_{t-1}^{\prime})^2 + x_i^T\Sigma_{t-1}^{\prime}x_i$$

- Update $q(w)$ by setting:

$$\Sigma_t^{\prime} = \lambda I + \frac{a_t^{\prime}}{b_t^{\prime}}\sum\limits_{i=1}^N x_i x_i^T$$

$$\mu_t^{\prime} = \Sigma_t^{\prime}(\frac{a_t^{\prime}}{b_t^{\prime}}\sum\limits_{i=1}^Ny_i x_i)$$

- Evaluate $\mathcal{L}$ so assess convergence. 

A few comments can be made here:

1 The order of updating each parameter shoudl not matter. Eventually, they will converge to whatever they should end up with. 

2 It is sometimes unavoidable that we have to calculate the variational objective function which we have seen as fairly complicated. However, to check the convergence, we have to do so. 

3 In EM, we get a point estimate of the model variable that we want to learn from data. In VI, we get the full distribution with each model varaible. 