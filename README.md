# AI Learning Hub

[![LICENSE](https://img.shields.io/badge/license-MIT-lightgrey.svg)](https://raw.githubusercontent.com/Wei2624/AI_Learning_Hub/master/LICENSE)

<a href="url"><img src="https://raw.githubusercontent.com/Wei2624/AI_Learning_Hub/master/AI.jpg" align="left" height="450" width="900" ></a>

<br><br>
<br><br>

Photo Credit: [Liam Kay](https://www.thirdsector.co.uk/author/4626/Liam-Kay)


AI Learning Hub is an open-sourced machine learning handbook. We contribute to this repo by summarizing interesting blog, course and/or notes of machine learning, deep learning, computer vision, robotics and/or statistics. We also intend to provide each post with Chinese version. 

We do this because we love AI and sharing. Excellent materials are the step stone for learning AI. We think everyone is deserved a chance to study AI with excellent materials. We welcome anyone to join us to make it better! 

And you own whatever you write here! 

## View Contents

We provide with two ways to view and learn the blogs. 

### View author's homepage (Highly Recommended!)

The best way to view the contents of any blog is to view the homepage of the author of that blog that especially interests you. The information of author's homepage of each blog is listed in this README and will be updated as any changes happen. 

We highly recommend this way to view the contents of any blog. 

### Use Ruby to view locally (Not Recommended)

1. Install Ruby environment. Instructions can be found [here](https://jekyllrb.com/docs/installation/).

2. Run

```
gem install jekyll bundler
```

3. Run

```
git clone https://github.com/Wei2624/AI_Learning_Hub.git
cd AI_Learning_Hub
bundle install
bundle exec jekyll build

```

4. In `_site` directory, you can find `.html` file. Then, you are able to view them locally. 

## Join us

You are very welcome to join us to improve this repo more! 

### Write Blog

The easiest way to contribute is to [fork](https://help.github.com/articles/fork-a-repo/) this project and write your own contents. Remember that you own whatever you write. 

To unify the style of each blog, you should use `markdown` as the syntax with `mathjax` as a plugin for math. Of course, you can insert `html` code whenever you want. An example of header of a blog can be as below:

```
---
layout: single
mathjax: true
title: Regularization and Model Selection
share: true
permalink: /MachineLearning/sv_regularization_model_selection/
---
```

For `layout`, you better either choose `single` where comments are enabled or `archive` where comments are disabled. For more layout options, you can view [here](https://mmistakes.github.io/minimal-mistakes/docs/layouts/). 

`permalink` is a slef-defined relative url path. If you want to host up your blog, you can append `permalink` to your `site-url`. 

**You better follow this procedure so that people can run `ruby` command to generate local page for view.**


### Host Blog

You can put up your own blog. The easiest way to do this is to use [submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules) from git. 

Essentially, you have your own repo. Then you can run `git submodule` command to add this repo as a subdirectory to your original repo. This repo will just become one of the folders in your repo. You can access whatever you write here. 


## Distribution of contents

**Distribution of contents without author's permission is strictly prohibited.**

Please respect the authorship of each blog there. If you want to distribute them, you can ask the author for permission. Every author here has all the rights to their written blog and is fully responsible for their written blogs. 


# Blog Information

| Module | Blog Title | Lang | Author | Contact |
|:--------:|:------------:|:------:|:--------:|:---------:|
|ML|[Generative Algorithm](https://wei2624.github.io/MachineLearning/sv_generative_model/)|EN|[Wei Zhang](https://wei2624.github.io/)|weiuw2624@gmail.com|
|ML|[Generative Algorithm](https://air-yan.github.io/machine%20learning/Generative-Learning-Algorithm/)|CH|[Zishi Yan](https://air-yan.github.io/)|WeChat:air-sowhat|
|ML|[Discriminative Algorithm](https://wei2624.github.io/MachineLearning/sv_discriminative_model/)|EN|[Wei Zhang](https://wei2624.github.io/)|weiuw2624@gmail.com|
|ML|[Support Vector Machine](https://wei2624.github.io/MachineLearning/sv_svm/)|EN|[Wei Zhang](https://wei2624.github.io/)|weiuw2624@gmail.com|
|ML|[Bias-Varaince and Error Analysis](https://wei2624.github.io/MachineLearning/sv_bias_variance_tradeoff/)|EN|[Wei Zhang](https://wei2624.github.io/)|weiuw2624@gmail.com|
|ML|[Learning Theory ](https://wei2624.github.io/MachineLearning/sv_learning_theory/)|EN|[Wei Zhang](https://wei2624.github.io/)|weiuw2624@gmail.com|
|ML|[Regularization and Model Selection](https://wei2624.github.io/MachineLearning/sv_regularization_model_selection/)|EN|[Wei Zhang](https://wei2624.github.io/)|weiuw2624@gmail.com|
|ML|[Online Learning and Perceptron Algorithm](https://wei2624.github.io/MachineLearning/sv_online_learning_perceptron/)|EN|[Wei Zhang](https://wei2624.github.io/)|weiuw2624@gmail.com|
|ML|[K-Means](https://wei2624.github.io/MachineLearning/usv_kmeans/)|EN|[Wei Zhang](https://wei2624.github.io/)|weiuw2624@gmail.com|
|ML|[EM Algorithm](https://wei2624.github.io/MachineLearning/usv_em/)|EN|[Wei Zhang](https://wei2624.github.io/)|weiuw2624@gmail.com|
|ML|[Variational Inference](https://wei2624.github.io/MachineLearning/bayes_vi/)|EN|[Wei Zhang](https://wei2624.github.io/)|weiuw2624@gmail.com|
|DL|[Nerual Networks ](https://wei2624.github.io/MachineLearning/dl_neural_network/)|EN|[Wei Zhang](https://wei2624.github.io/)|weiuw2624@gmail.com|
|DL|[Backpropagation](https://wei2624.github.io/MachineLearning/dl_propagtion/)|EN|[Wei Zhang](https://wei2624.github.io/)|weiuw2624@gmail.com|

