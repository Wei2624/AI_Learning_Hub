---
layout: single
mathjax: true
toc: true
toc_sticky: true
category: Machine Learning
tags: [notes]
qr: machine_learning_notes.png
title: Reinforcement Learning
share: true
permalink: /MachineLearning/rl/
sidebar:
  nav: "MachineLearning"
---


# 1 Introduction

In supervised learning scheme, we have labels for each data sample, which sorts of represents the "right answer" for each sample. In reinforcement learning (RL), we have no such labels at all because it might not be appropriate to define the "right answer" in some scenario. For example, it is hard to label the correct movement for game Go to achieve a higher score given a current state. Unlike unsupervised learning either where no metric evalution is given for a new prediction, RL has the reward function to evluate the proposed action. The goal is to maximize the reward. 


# 2 Markov Decision Processes (MDP)

MDP has been the foundation of RL. MDP typically contains a typle $(S, A, \{P_{sa}\}, \gamma, R$ where: 

- S is the set of states.

- A is the set of actions that an agent can take.

- $P_{sa}$ is the transition probability vector of taking action $a\in A$ at the state $s \in S$. 

- $\gamma$ is the discount factor

- $R: S\times A \rightarrow \mathbb{R}$ is the reward function. 

A typical MDP starts from an intial state $s_0$ and $a_o \in A$. Then we trainsit to $s_1 \sim P_{s_0 a_0}$. This process will continue as:

$$s_0 \rightarrow{a_0} s_1 \rightarrow{a_1} s_2 \rightarrow{a_2} \dots$$ 

The **reward** for such a sequence can be defined as:

$$R(s_0,a_0) + \gamma R(s_1,a_1) + \gamma^2 R(s_2, a_2) + \dots$$

Or without the loss of generality:

$$R(s_0) + \gamma R(s_1) + \gamma^2 R(s_2) + \dots$$

Our goal is to maxmize:

$$\mathbb{E}[R(s_0) + \gamma R(s_1) + \gamma^2 R(s_2) + \dots]$$

One key note here is to notice the discount factor which is compounded over time. This simply means that to maximize the total reward, we want to get the largest reward as soon as possible and postpone negative rewards as long as possible. 

To make an action at a given state, we also have the **policy** $\pi: S\rightarrow A$ mapping from the states to the actions, namely $a=\pi(s)$. In addition, we also define the **value function** with the fixed policy as:

$$V^{\pi}(s) = R(s) + \gamma \sum\limits_{s^{\prime}\in S} P_{s\pi(s)}(s^{\prime})V^{\pi}(s^{\prime})$$

which is also called **Bellman equation**. The first term is the immediate reward of a state s. The second term is the expected sum of discount rewards for starting in state $s^{\prime}$. Basically, it is $\mathbb{E}_{s^{\prime}\sim P_{s\pi(s)}}[V^{\pi}(s^{\prime})]$. Bellman equations enable us to solve a finite-state MDP problem ($\| S \| < \infty$). We can write dowm $\| S \|$ equations with one for each state with $\| S \|$ variables. 

The **optimal value function** is defined as:

$$V^{\ast}(s) = \max_{\pi}V^{\pi}(s)$$

We can also write it in Bellman's form:

$$V^{\ast}(s) = R(s) + \max_{a\in A} \gamma \sum\limits_{s^{\prime}\in S} P_{sa}(s^{\prime})V^{\ast}(s^{\prime})$$

Similarily, we can have:

$$\pi^{\ast}(s) = \arg\max_{a\in A} \sum\limits_{s^{\prime}\in S} P_{sa}(s^{\prime})V^{\ast}(s^{\prime})$$

Then, we can conlcude that:

$$V^{\ast}(s) = V^{\pi^{\ast}}(s) \geq V^{\pi}(s)$$

One thing to notice is that $\pi^{\ast}$ is optimal for all the states regardless of what current state it is. 



