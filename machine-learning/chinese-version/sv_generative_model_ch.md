---
published: true
layout: single
mathjax: true
toc: true
toc_sticky: true
category: Machine Learning
tags: [notes,chinese]
excerpt: "This post is a translation for one of posts in his machine learning notes."
title: Generative Learning Algorithm in Chinese
share: true
qr: machine_learning_notes.png
author_profile: true
permalink: /MachineLearning/sv_generative_model_ch/
---

# 生成学习算法

## 1 判别模型

判别模型是一种对观测数据进行直接分类的模型，常见的模型有逻辑回归和感知机学习算法等。此模型仅对数据进行分类，并不能具象化或者量化数据本身的分布状态，因此也无法根据分类生成可观测的图像。

定义上，判别模型通过构建条件概率分布p(y&#124;x;θ)预测y，即在特征x出现的情况下标记y出现的概率。此处p可以是逻辑回归模型。

## 2 生成模型

与判别模型不同，生成模型首先了解数据本身分布情况，并进一步根据输入x，给出预测分类y的概率。该模型有着研究数据分布形态的概念，可以根据历史数据生成新的可观测图像。

贝叶斯分类就是一个典型的例子。在这个例子中，我们有一个先验分类，根据这个先验分类，我们可以使用贝叶斯原理计算每个分类的概率，然后取概率最高的概率。同时，我们还可以根据特定的先验生成特征。这就是一个生成过程。

## 3 高斯判别分析

高斯判别分析（GDA）是一个生成模型，其中p(x&#124;y)是多元高斯正态分布。

### 3.1 多元高斯正态分布

在多元正态分布中，一个随机变量是一个R<sub>n</sub>空间中的矢量值，其中n代表维度数。因此，多元高斯的均值向量 μ∈R<sub>n</sub>，协方差矩阵Σ∈R<sub>n x n</sub> ，其中$ \\ Sigma是对称的半正定矩阵。其概率密度函数为：

![image001.png](/images/ML_notes/chinese/Generative_Images/image001.png)

如上所述，μ代表期望值。

随机向量Z（或者说，向量化的随机变量Z）的协方差为：

![image003.png](/images/ML_notes/chinese/Generative_Images/image003.png)


下图显示了几个密度函数，它们的均值均为零，但协方差不同：

![image005.png](/images/ML_notes/chinese/Generative_Images/image005.png)


上图的协方差为（从左到右）：

![image007.png](/images/ML_notes/chinese/Generative_Images/image007.png)




## 4 高斯判别分析和逻辑回归

### 4.1 高斯判别分析

我们再来谈谈二分类问题，我们可以用多元高斯模型对p(x&#124;y)进行建模。 总的来讲，我们有：

![image009.png](/images/ML_notes/chinese/Generative_Images/image009.png)


在这里面，我们想要找出的参数φ，μ<sub>0</sub>，μ<sub>1</sub>，和Σ。 请注意，虽然每个类的均值不同，但它们的协方差相同。

这里有人会问，那为什么它是一个生成模型呢？简而言之，我们首先有一个类，也有这个类的y的先验概率，并且知道这个类的分布类型是伯努利分布。那么生成过程就是（1）从伯努利分布的类中抽样。 （2）基于类标签，我们从相应的分布中抽取x。这便是一个生成过程。

所以，该数据的对数似然函数值为：

![image011.png](/images/ML_notes/chinese/Generative_Images/image011.png)

在上面的等式中，我们插入每个分布，但不指明具体这个分布是哪个类，我们仅将它们抽象为k。我们可以得到：

![image013.png](/images/ML_notes/chinese/Generative_Images/image013.png)


现在，我们需要对每个参数进行取导，然后将它们设为零并找到 argmax（函数值最大时对应的输入值x）。 一些可能对推导有用的公式列举如下：

![image015.png](/images/ML_notes/chinese/Generative_Images/image015.png)（如果A是对称的并且与x相互独立）




![image017.png](/images/ML_notes/chinese/Generative_Images/image017.png)

**证明：**
矩阵A是对称矩阵，所以 *A*= *A*<sup>T</sup>并假设空间维度为n。

![image019.png](/images/ML_notes/chinese/Generative_Images/image019.png)

雅可比公式：

![image021.png](/images/ML_notes/chinese/Generative_Images/image021.png)

**证明：**

![image023.png](/images/ML_notes/chinese/Generative_Images/image023.png)

**证明：**

这个证明有些复杂。你应该事先了解克罗内克函数和Frobenius内部乘积。对于矩阵X，我们可以写成：

![image025.png](/images/ML_notes/chinese/Generative_Images/image025.png)

你可以将H视为Frobenius内积的标识元素。在开始证明之前，让我们准备好去找逆矩阵的导数。也就是说，∂X<sup>-1</sup>/∂X。

![image027.png](/images/ML_notes/chinese/Generative_Images/image027.png)



所以我们可以这么解：

![image029.png](/images/ML_notes/chinese/Generative_Images/image029.png)

接着，让我们回到正题：

![image031.png](/images/ML_notes/chinese/Generative_Images/image031.png)

其中F表示Frobenius内积。

接着，带回到原始公式：

![image033.png](/images/ML_notes/chinese/Generative_Images/image033.png)



现在，我们已经有足够的准备去找到每个参数的梯度了。


对ϕ取导并设为0：

![image035.png](/images/ML_notes/chinese/Generative_Images/image035.png)


对 μk取导并设为0：

![image037.png](/images/ML_notes/chinese/Generative_Images/image037.png)


对 Σ 取导并设为0:

![image039.png](/images/ML_notes/chinese/Generative_Images/image039.png)


结果如图所示：

![image041.png](/images/ML_notes/chinese/Generative_Images/image041.png)


请注意，由于有着同样的协方差，因此上图两个轮廓的形状是相同的，然而均值不同。在边界线这条线上（自左上到右下的直线），每个类的概率为50%。

### 4.2 高斯判别分析（GDA）和逻辑回归

高斯判别分析又是如何与逻辑回归相关联的呢？我们可以发现如果上述p(x&#124;y) 是具有共同协方差的多元高斯，我们就可以计算p(x&#124;y)并证明它是遵循逻辑函数的。要证明这一点，我们可以：

![image043.png](/images/ML_notes/chinese/Generative_Images/image043.png)


由于高斯属于指数族，我们最终可以将分母中的比率转换为exp（θ<sup>T</sup>x），其中 θ 是φ，μ<sub>0</sub>，μ<sub>1</sub>，Σ的函数。

同样的，如果p(x&#124;y) 是具有不同 λ 的泊松分布，则p(x&#124;y) 也遵循逻辑函数。 这意味着GDA模型本身有一个强假设，即每个类的数据都可以用具有共享协方差的高斯模型建模。但是，如果这个假设是正确的话，GDA将可以更好并且更快地训练模型。

另一方面，如果不能做出假设，逻辑回归就不那么敏感了。因此，你可以直接使用逻辑回归，而无需接触高斯假设或Possion假设。

## 5 朴素贝叶斯

在高斯判别分析中，随机变量应使用具有连续值特征的数据。 而朴素贝叶斯则用于学习离散值随机变量，如文本分类。在文本分类中，模型基于文本中的单词将文本标记为二进制类，单词被向量化并用于模型训练。一个单词向量就像一本字典一样，其长度是字典中单词储存的数量，其二进度值则代表着是否为某个词。 一个单词在单词向量中由1表示“是”，而单词向量中的其他位置则是0。

然而，这可能并不起作用。 比方说，如果我们有50,000个单词并尝试将其建模为多项式，则参数的维数为250,000-1,250,000-1，这太大了。 因此，为了解决这个问题，我们做出了

**朴素贝叶斯假设**：

基于给定分类下，每个词彼此间条件独立。

于是，我们有：![image045.png](/images/ML_notes/chinese/Generative_Images/image045.png)


我们对第一步应用**概率论中的链式法则**，对第二步应用朴素贝叶斯假设。

找到对数似然函数值的最大值：

![image047.png](/images/ML_notes/chinese/Generative_Images/image047.png)


其中 ϕ<sub>j&#124;y=1</sub> = P (x<sub>j</sub>=1&#124;y=1)，ϕ <sub>j&#124;y=1</sub> = P(x<sub>j</sub>=1&#124;y=1), ϕ<sub>j&#124;y=0</sub> = P(x<sub>j</sub>=1&#124;y=0) 并且 ϕ<sub>y</sub>= p(y=1)。 这些是我们需要训练的参数。

我们可以对其求导:

![image049.png](/images/ML_notes/chinese/Generative_Images/image049.png)


为了预测新样本，我们可以使用**贝叶斯法则**来计算P（y = 1 &#124; x）并比较哪个更高。

![image051.png](/images/ML_notes/chinese/Generative_Images/image051.png)


**延伸**: 在这种情况下，因为y是二进制值（0，1），我们将P（x<sub>i</sub> &#124; y）建模为伯努利分布。 也就是说，它可以是“有那个词”或“没有那个词”。 伯努利将类标签作为输入并对其概率进行建模，前提是它必须是二进制的。 如果是处理非二进制值X<sub>i</sub>，我们可以将其建模为多项式分布，多项式分布可以对多个类进行参数化。

**总结**: 朴素贝叶斯适用于离散空间，高斯判别分析适用于连续空间。我们任何时候都能将其离散化。

## 6 拉普拉斯平滑处理

上面的示例通常是好的，不过当新邮件中出现过去训练样本中不存在的单词时，该模型将会预测失败。 在这种情况下，它会因为模型从未看到过这个词而导致两个类的φ变为零，以至于无法进行预测。

这时我们则需要另一个解决方案，其名为**拉普拉斯平滑**，它将每个参数设置为：

![image053.png](/images/ML_notes/chinese/Generative_Images/image053.png)


其中k是类的数量。在实际操作中，拉普拉斯平滑并没有太大的区别，因为我们的模型中通常包含了所有的单词。不过有个Plan B总是极好的~
