---
layout: single
mathjax: true
toc: true
toc_sticky: true
category: Machine Learning
tags: [notes]
qr: machine_learning_notes.png
title: Support Vector Machine (SVM)
share: true
permalink: /MachineLearning/sv_svm/
sidebar:
  nav: "MachineLearning"
---

# 1 Intuition and Notation

In general, binary classification is of great interests since it is the bottom line for multi-classes classification. The simplest case of binary classification is to classify points in space and can be sovled by using a determinist algorithm called support vector machine. 

![SVM Intuition](/images/svm_intuition.png)

From the figure, we have A, B and C point in the space. A is the safest point since it is far from the **boundary line**, while C is the most dangerous point since it is close to the **hyperplane**. The distance between the boundary line and the point is called **margin**.

We also denote $x$ as feature vector, $y$ as label and $h$ as classifier. Thus, the classifier an be shown as:

$$h_{w,b}(x) = g(w^Tx + b)$$

Note, we have w, b instead $\theta$ here. And the label only takes the value 1 and -1 instead of 0 and 1. The classifier predicts directly as 1 or -1 like **perceptron algorithm** without calculating the probability like what logistic algorithm does. **However, this does not mean SVM cannot output its corresponding probability. Rather, it will tell us how confident the predict is. **

# 2 Functional and Geometric Margins

**Functional margins** with respect to training example:

$$\overset{\wedge}{\gamma^{(i)}} = y^{(i)}(w^Tx^{(i)} + b)$$

We want $(w^Tx^{(i)} + b)$ to be a large positive number if label is positive or large negative number if label is negative. Thus, it means that **functional margin should be positive to be correct. And the larger the margin, the more confident we are.** However, this might not be meaningful when we replace w and b with 2w and 2b without changing anything else. Thus, this leads to the deifnition **geometric margine** coming next. Furthermore, we denote the function margin for the dataset as:

$$\overset{\wedge}{\gamma} = \min_{i=1,\dots,m} \overset{\wedge}{\gamma^{(i)}} $$

where m is the number of training samples. 

**Geometric Margins:** In functional margin, We need to normalize w and b **with respect to the norm of w** since magnitude of w and b should not affect the scale of the margin. A figure for geometric margin can be shown:

![SVM Geometric Margins](/images/svm_gm.png)

It shows a vector w also called **support vector** which is perpendicular to the boundary line, which is always true. To prove this, you just need to take two points on boundary line to get a parallel vector and prove that the dot product is 0. 

Similarily, to find the margin of point A, we take point B as the projected point of A. Formally, $x^{(i)} - \gamma^{(i)} w/\lvert\lvert w \rvert\rvert$. The point is on boundary, meaning that

$$w^T(x^{(i)} - \gamma^{(i)} w/\lvert\lvert w \rvert\rvert) + b = 0$$

Solve:

$$\gamma^{(i)} = (w/\lvert\lvert w \rvert\rvert)^T x^{(i)} + b/\lvert\lvert w \rvert\rvert$$

Thus, **geometric margin** with respect to a training sample is defined as:

$$\gamma^{(i)} = y^{(i)}((w/\lvert\lvert w \rvert\rvert)^T x^{(i)} + b/\lvert\lvert w \rvert\rvert)$$

If $\lvert\lvert w \rvert\rvert = 1$, the functional margin is equal to geometric margin. Similarily, the geometric margin for all samples is:

$$\gamma = \min_{i=1,\dots,m}\gamma^{(i)}$$

# 3 Optimal Margin Classifier

The goal is to maximize the geometric margin.

For now, we assume that data is linearly separable. The optimization problem can be defined as :

$$\max_{\gamma,w,b} \gamma$$

$$ \text{s.t. } y^{(i)}(w^Tx^{(i)} + b) \geq \gamma, i = 1,\dots,m$$

$$\lvert\lvert w \rvert\rvert = 1$$

The nasty point is $\lvert\lvert w \rvert\rvert = 1$ constraint, which makes it is non-convex. If it is convex, we can get the derivative and set to zero. This is another topic. 

We can then transform it to:

$$\max_{\overset{\wedge}{\gamma},w,b} \frac{\overset{\wedge}{\gamma}}{\lvert\lvert w \rvert\rvert}$$

$$ \text{s.t. } y^{(i)}(w^Tx^{(i)} + b) \geq \overset{\wedge}{\gamma}, i = 1,\dots,m$$

Basically, we relate geometric margin with function margine. Instead of geometric margin, we subject to a functional margin. **By doing this, we eliminate $\lvert\lvert w \rvert\rvert = 1$.** However, it is still bad. 

By scaling constraint on w and b, we do not change anything. We use this fact to make $\overset{\wedge}{\gamma} = 1$.And then, the max problem becomes a min problem now. That is,

$$\min_{\gamma,w,b} \frac{1}{2} \lvert\lvert w \rvert\rvert^2$$

$$ \text{s.t. } y^{(i)}(w^Tx^{(i)} + b) \geq 1, i = 1,\dots,m$$

The problem can be solved by using quadratic programming software. We can still go further to simplify this but it requires the knowledge of **Lagrange Duality**

# 4 Lagrange Duality

Let's take a side step on how to solve general **constrained optimizing problem.** In general, we usually use Lagrange Duality to solve this type of question. 

Consider a problem such as :

$$\min_w f(w)$$

$$\text{s.t. } h_i(w) = 0,i = 1,\dots,l$$

Now, we can define **Lagrangian** to be:

$$\mathcal{L}(w,\beta) = f(w) + \sum\limits_{i=1}^l \beta_i h_i(w)$$

where $\beta_i$ is called **Lagrange multiplier.** Now, we can use partial derivative to set to zero and find out w and $\beta$

We can generalize to both inequality and equality constraints. So we can define **primal** problem to be:

$$\min_w f(w)$$

$$\text{s.t. } g_i(w) \leq 0,i = 1,\dots,k$$

$$h_i(w) = 0,i = 1,\dots,l$$

We define **generalized Lagrangian** as:

$$\mathcal{L} = f(w) + \sum\limits_{i=1}^k \alpha_i g_i(w) + \sum\limits_{i=1}^l \beta_i h_i(w)$$

where all $\alpha$ and $\beta$ are Lagrangian multiplier. 

Let's define a quantity for primal problem as :

$$\theta_{\mathcal{P}}(w) = \max_{\alpha,\beta:\alpha_i\geq 0} \mathcal{L}(w,\alpha,\beta)$$

If some constraints are violated, then $\theta_{\mathcal{P}}(w) = \infty$

Thus, we have:

$$\theta_{\mathcal{P}}(w) = \begin{cases} f(w)  \text{, if w satisfy primal constraints} \\ \infty  \text{, otherwise} \\ \end{cases}$$

To match to our primal problem, w define the min problem as:

$$\min_w \theta_{\mathcal{P}}(w) = \min_w \max_{\alpha,\beta:\alpha_i\geq 0} \mathcal{L}(w,\alpha,\beta)$$

This is the same as the primal problem if all constrain are satisfied. We define the value of primal problem to be: $p^{\ast} = \min_w \theta_{\mathcal{P}(w)}$. Then, we define:

$$\theta_{\mathcal{D}}(\alpha,\beta) = \min_w \mathcal{L}(w,\alpha,\beta)$$

to be the dual part. To again match the primal problem, we define the **dual optimization problem** to be:

$$\max_{\beta,\alpha:\alpha_i\geq 0} = \max_{\alpha,\beta:\alpha_i\geq 0} \min_w \mathcal{L}(w,\alpha,\beta)$$

Similarily, the value of dual problem is $d^{\ast} = \max_{\alpha,\beta:\alpha_i\geq 0} \theta_{\mathcal{D}}(\alpha,\beta)$

The primal and dual problem is related by:

$$d^{\ast} = \max_{\alpha,\beta:\alpha_i\geq 0} \theta_{\mathcal{D}}(\alpha,\beta) \leq p^{\ast} = \min_w \theta_{\mathcal{P}(w)}$$

This is always true. The proof can be found online (I will add this into another notes and put a link here). The key is that under certain condition, they are equal. If they are equal, we can focus on dual problem instead of primal problem. 

We assume that f and g are all convex and h are affine(**When f has a Hessian, it is convex iff Hessian is positive semi-definite. All affine are convex. Affine means linear.**) and g are all less than 0 for some w. Wtih these assumptions, there must exist $w^{\ast}$ for primal solution and $\alpha^{\ast},\beta^{\ast}$ for dual solution and $p^{\ast} = d^{\ast}$. And $w^{\ast}$,$\alpha^{\ast}$ and $\beta^{\ast}$ satisfy **Karush-Kuhn-Tucker (KKT) conditions**, which says:

$$\frac{\partial}{\partial w_i}\mathcal{L}(w^{\ast},\alpha^{\ast},\beta_{\ast}) = 0. i = 1,\dots,n$$


$$\frac{\partial}{\partial \beta_i}\mathcal{L}(w^{\ast},\alpha^{\ast},\beta_{\ast}) = 0. i = 1,\dots,l$$

$$\alpha_i^{\ast}g_i(w^{\ast}) = 0,i = 1,\dots,k$$

$$g_i(w^{\ast}) \leq 0,i = 1,\dots,k$$

$$\alpha_i^{\ast} \geq 0,i = 1,\dots,k$$

Third euqaiton is called **KKT dual complementarity condition**. It means if $\alpha_i^{\ast} > 0$, then $g_i(w^{\ast}) = 0$.

# 5 Optimal Margin Classifier

Let's revisit the primal problem:

$$\min_{\gamma,w,b} \frac{1}{2} \lvert\lvert w \rvert\rvert^2$$

$$ \text{s.t. } y^{(i)}(w^Tx^{(i)} + b) \geq 1, i = 1,\dots,m$$

we can re-arrange the constraint to be:

$$g_i(w) = -y^{(i)}(w^Tx^{(i)} + b) + 1 \leq 0$$

where i spans all training samples. From KKT dual complementarity condition, we have $\alpha_i > 0$ only when the functional margin is 1 ($g_i(w) = 0$).

We can vistualize this in the picture below. The three points on the dash line are the ones with the smallest geometric margin which is 1. Thus, those points are the ones with positve $\alpha_i$ and are called **support vector**. 

![SVM Boundary](/images/svm_bound.png)

The Lagranian with only inequality constraint is:

$$\mathcal{L}(w,b,\alpha) = \frac{1}{2}\lvert \lvert w\rvert \rvert^2 - \sum\limits_{i=1}^m \alpha_i [y^{(i)}(w^Tx^{(i)} + b) - 1] \tag{1}$$

To find the dual form of this problem, we first find min of loss with respect to w and b for a fixed $\alpha$. To do that, we have:

$$\triangledown_{w}\mathcal{L}(w,b,\alpha) = w - \sum\limits_{i=1}^m \alpha_i y^{(i)}x^{(i)} = 0\tag{2}$$

$$w = \sum\limits_{i=1}^m \alpha_i y^{(i)}x^{(i)}\tag{3}$$

$$\frac{\partial}{\partial b}\mathcal{L}(w,b,\alpha) = \sum\limits_{i=1}^m \alpha_i y^{(i)} = 0 \tag{4}$$

We take equation (3) back to equation (1) we have:

$$\mathcal{L} = (w,b,\alpha) = \sum\limits_{i=1}^m \alpha_i - \frac{1}{2}\sum\limits_{i,j}^m y^{(i)}y^{(j)}\alpha_i\alpha_j (x^{(i)})^Tx^{(j)} - b\sum\limits_{i=1}^m\alpha_i y^{(i)}$$

$$= \sum\limits_{i=1}^m \alpha_i - \frac{1}{2}\sum\limits_{i,j}^m y^{(i)}y^{(j)}\alpha_i\alpha_j (x^{(i)})^Tx^{(j)}$$

Thus, we have the dual problem as :

$$\max_{\alpha} W(\alpha) = \sum\limits_{i=1}^m \alpha_i - \frac{1}{2}\sum\limits_{i,j}^m y^{(i)}y^{(j)}\alpha_i\alpha_j (x^{(i)})^Tx^{(j)}$$

$$\text{s.t.} \alpha_i \geq 0, i = 1,\dots,m$$

$$\sum\limits_{i=1}^m \alpha_i y^{(i)} = 0$$

which satisfies KKT condition. It means we found out the dual problem to solve instead of primal problem. If we can find $\alpha$ from this dual problem, we can use equation (3) to find $w^{\ast}$. With optimal $w^{\ast}$, we can find $b^{\ast}$:

$$b^{\ast} = -\frac{\max_{i:y^{(i)}=-1}w^{\ast T}x^{(i)} + \min_{i:y^{(i)}=1}w^{\ast T}x^{(i)}}{2}$$

This is easy to verify. The optimal w and b will make the geometric margin of cloest negative and positive sample to be equal. 

The equation (3) says that the optimal w is based on the optimal $\alpha$. To make prediction, we have:

$$w^Tx + b = (\sum\limits_{i=1}^m \alpha_i y^{(i)} x^{(i)})^Tx + b = \sum\limits_{i=1}^m \alpha_i y^{(i)} <x^{(i)},x> + b$$

If it is bigger than zero, we predict one. We also know that $\alpha$ will be all zeros except for the support vectors. That means **we only cares about the inner product between x and support vector**. This makes the prediction faster and brings the **Kernel funciton** into the sight. Keep in mind that so far everything is low dimensional. How about high dimensions and infinite dimension space?


# 6 Kernels

In the example of living area of house, we can use the feature $x.x^2,x^3$ to get cubic function. X is called **input attribute** and $x.x^2,x^3$ is called **features**. We dentoe $\phi (x)$ the feature mapping from attribute to features. 

Thus, we might want to learn inthe new feature space $\phi (x)$. In last section,we only need to calculate inner product $<x,z>$ and now we can replace it with $<\phi(x),\phi(z)>$. 

Formally, given a mapping, we denote **Kernel** to be:

$$K(x,z) = \phi(x)^T\phi(z)$$

We can use Kernel for the replacement instead of mapping itself. The reason is that Kernel is less expensive computationally. So we can learn in high dimensuional space without calculating mapping $\phi$.

An example of how effective it is can be shown in the notes. It should be noted that calculating mapping is exponential time complexity whereas Kernel is linear time. 

In another way, Kernel is a measurement of how close or how far it is between x and z. It indicates the similarity. One of the popular Kernel is called **Gaussian Kernel** defined as: 

$$K(x,z) = \exp(-\frac{\lvert\lvert x-z \rvert\rvert^2}{2\sigma^2})$$

We can use this as learning SVM and it corresponds to infinite dimensional feature mapping $\phi$. It also shows that it is impossible to calculate infinite dimensional mapping but we can use Kernel instead. 

Next, we are insterested in telling if a Kernel is valid or not. 

We define **Kernel Matrix** as $K_{ij} = K(x^{(i)},x^{(j)})$ for m points(i.e. K is m-by-m). Now, if K is valid, it means:

(1)Symmetric: $K_{ij} = K(x^{(i)},x^{(j)}) = \phi(x^{(i)})^T\phi(x^{(j)}) = \phi(x^{(j)})^T\phi(x^{(i)}) = K_{ji}$

(2)Positive semi-definite: $z^TKz \geq 0$ proof is easy. 

**Mercer Theorem: Let $K:\mathbb{R}^n \times \mathbb{R}^n \mapsto \mathbb{R}$ be given. Then for a Kernel to be valid, it is necessary and sufficient that for any $\{x^{(1)},\dots,x^{(m)}\}$, the corresponding kernel matrix is symmetric and postive semi-definite.**

Kernel method is not only used in SVM but also anywhere that inner product is used. So we can replace the inner product with Kernel so that we can work in a higher dimensional space. 

# 7 Regularization and Non-separable Case

Although mapping x to higher dimensional space increases the chance to be separable, it might not always be the case. An outlier could also be the cause that we actually don't want to include. An example of such a case can be shown below. 

![SVM outlier](/images/svm_outlier.png)

To make the algorithm work for non-linear case as well, we add **regularization** to it:

$$\min_{\gamma,w,b} \frac{1}{2}\lvert\lvert w\rvert\rvert^2 + C\sum\limits_{i=1}^m \xi_i$$

$$\text{s.t. } y^{(i)}(w^Tx^{(i)} + b) \geq 1-\xi_i,i=1,\dots,m$$

$$\xi_i \geq 0,i=1,\dots,m$$

It will pay the cost for the functional margin that is less than one. C controls how cost that would be. It says that:

(1) We want w to be small so that margine will be large. 

(2) We want most samples to have functional margin that is larger than 1. 

The Lagrangian is :

$$\mathcal{L}(w,b,\xi,\alpha,r) = \frac{1}{2}w^Tw + C\sum\limits_{i=1}^{m}\xi_i - \sum\limits_{i=1}^m \alpha_i[y^{(i)}(x^{(i)T}w + b) - 1 + \xi_i] - \sum\limits_{i=1}^{m}r_i\xi_i$$

where $\alpha$ and r are Lagrangian multipliers which must be non-negative. Now, we can set derivative on w,b and $\xi$ to zero and find w. Keep in mind that we try to take min in the dual problem so we do not want to give it chance to have anything like $-\infty$. Plugging back will produce the dual problem as:

$$\max_{\alpha} W(\alpha) = \sum\limits_{i=1}^{m}\alpha_i - \frac{1}{2}\sum\limits_{i,j=1}^{m}y^{(i)}y^{(j)}\alpha_i\alpha_j<x^{(i)},x^{(j)}>$$

$$\text{s.t. }0\leq \alpha_i \leq C,i=1,\dots,m$$

$$\sum\limits_{i=1}^{m}\alpha_i y^{(i)} = 0$$

Notice that we have an interval for $\alpha$ becuase it has $\sum\limits_{i=1}^{m}(C-\alpha_i-r_i)\xi_i$. We take derivative with respect to $\xi$ and set to zero and we can eliminate $\xi$.

Also notice that the optimal b is not the same anymore because the margin for both cloest points have changed. In next section, we will find the algrotihm to figure out the solution. 

# 8 The SMO Algorithm

The SMO(sequential minimal optimization) algorithm by John Platt is to solve the dual problem in SVM. 

## 8.1 Coordinate Ascent

In general, the optimization problem

$$\max_{\alpha}W(\alpha_1,\alpha_2,\dots,\alpha_m)$$

can be solved by gradient ascent and Newton's method. In addition, we can also use coordinate ascent:

{% highlight bash %}
for loop until convergence:
  for i in range(1,m):
    alpha(i) = argmin of alpha(i) W(all alpha)
{% endhighlight %}

Basically, we fix all the $\alpha$ except for $\alpha_i$ and then move to next $\alpha$ for updating. **The assumption is that calculating gradient to $\alpha$ is efficient.** An example can be shown below. 

![SVM coordinate](/images/svm_coordinate.png)

Note that the path of the convergence is always parallel to axis because it is updated one variable at a time. 

## 8.2 SMO

We cannot do the same thing in dual problem in SVM because varying only one variable might violate the constraint:

$$\alpha_1 y^{(1)} = -\sum\limits_{i=2}^m \alpha_i y^{(i)}$$

which says once we determine the rest of $\alpha$, we cannot vary the left $\alpha$ anymore. Thus, we have to vary two $\alpha$ at one time and update them. For exmaple, we can have:

$$\alpha_1 y^{(1)} + \alpha_2 y^{(2)} = -\sum\limits_{i=3}^m \alpha_i y^{(i)}$$

We make right side to be constant:

$$\alpha_1 y^{(1)} + \alpha_2 y^{(2)} = \zeta$$

which can be pioctorially shown as:

![SVM coordinate](/images/svm_two_coord.png)

Note that although it is a square where $\alpha$ can lie but with a straight line, we might have a lower bound and upper bound on them. 

We can rewrite the above equation by multiplying $y^{(1)}$ on both sides:

$$\alpha_1 = (\zeta - \alpha_2 y^{(2)})y^{(1)}$$

Then, W will be :

$$W(\alpha_1,\dots,\alpha_m) = W((\zeta-\alpha_2 y^{(2)})y^{(1)},\alpha_2,\dots,\alpha_m)$$

We treat all other $\alpha$ as constants.Thus, after plugging in, W will become quadratic, which can be written as $a\alpha_2^2 + b\alpha_2 + c$ for some a, b and c. 

Last, we define $\alpha_2^{new, unclipped}$ as the current solution to update $\alpha_2$. Thus, with applying constraints, only for this single variable, we can write:

$$\alpha_2^{new} = \begin{cases} H  \text{, if          }\alpha_2^{new, unclipped}>H \\ \alpha_2^{new, unclipped}  \text{, if } L\leq \alpha_2^{new, unclipped} \leq H \\ L  \text{, if          } \alpha_2^{new, unclipped} < L \\ \end{cases}$$
