---
layout: post
title: Stochastic Gradient Descent, SGD with momentum, AdaGrad, RMSProp, and Adam algorithms
date: 2024-06-18
description: A discussion on different stochastic gradient algorithms and why they are efficient
tags: optimization algorithms
categories: algorithms
pseudocode: true
related_publications: true
---

As we know, the Adam algorithm is a highly efficient algorithm for solving unconstrained optimization problems, widely used in machine learning and deep learning. How does this algorithm work? What's the logic behind it? Why is it so efficient? In this post, I would like to discuss the Adam algorithm and the core idea behind it. 

Since Adam is based on Stochastic Gradient Descent (SGD), it is necessary to first discuss some SGD algorithms. To delve deep into SGD algorithms, we'd better start with Gradient Descent (GD) algorithm, which is the fundamental algorithm in solving unconstrained optimization problems.

### GD Algorithm

Suppose we have a general unconstrained optimization problem:

$$
\min_{\theta} f(\theta)
$$

Our goal is to find the optimal $\theta$ that minimizes the objective function. Unlike simple quadratic functions, most nonlinear functions are difficult to minimize directly by using basic mathematical techniques. GD is an effective and brilliant algorithm for solving such problems. The core idea of this algorithm is to iteratively adjust the variables incrementally. Many iterations are typically required to approach the optimal solution. In each iteration, we use a specific rule to update $\theta$:

$$
\theta_{t+1} = \theta_t - \alpha \nabla f(\theta_t)
$$

Here, $t$ represents the iteration index, $\theta$ is the variable being optimized, $\alpha$ is the step size determining how large a step should be taken in the opposite direction of the gradient, and $\nabla f$ is the gradient of the objective function. So why does this rule work? Acctually, this rule guarantees that the objective function decreases at each iteration. 

Suppose our goal if to find a rule that can improve the variable incrementally. At each iteration, we want to update it by:

$$
\theta_{t+1} = \theta_t + \alpha p_t
$$

where $p_t$ is a direction we pick at iteration t. At previous iteration, the objective value is $f(\theta_t)$, while at this iteration the objective value is $f(\theta_{t+1})$. 

$$
f(\theta_{t+1}) = f(\theta_t + \alpha p_t)
$$
 
The objective value decreases if $$ f(\theta_{t+1}) < f(\theta_t) $$, which implies $ f(\theta_t + \alpha p_t) < f(\theta_t) $ for sufficiently small $\alpha > 0$. Let us define $\phi : \mathbb{R}^+ \rightarrow \mathbb{R} $ as $\phi(\alpha) = f(\theta_t+\alpha p_t)$. We can have $p_t$ as a descent direction if $\phi'(0) < 0$.

$$
\phi'(0) = p_t^T \nabla f(\theta_t)
$$

So if we choose $p_t = - \nabla f(\theta_t)$, then $p_t^T \nabla f(\theta_t) = - \nabla^2f(\theta_t) < 0$ is a steepest descent direction.


<br />
### SGD Algorithm
SGD algorithm is like a variant of GD algorithm, which targeted at the stochastic objective function or large scale machine learning problem. The only difference is the update rule at each iteration. In GD, the update rule at each iteration is:

$$
\theta_{t+1} = \theta_t - \alpha \nabla f(\theta_t)
$$

While in SGD, the update rule is:

$$
\theta_{t+1} = \theta_t - \alpha \nabla f_{i_t}(\theta)
$$

$\nabla f_{i_t}(\theta_t)$ represents stochastic gradient instead of the full gradient. Let's illustrate this firstly. Let's use machine learning loss function as an example. Suppose we are trying to minimize the loss function of machine learning: 

$$\min_{\theta} \frac{1}{n} \sum_{i=1}^{n} f_i(\theta)$$

n could be super large. So in this case, the full gradient of objective function at $\theta$ is: $\frac{1}{n} \sum_{i=1}^{n} \nabla f_i(\theta)$. The computational cost is large when n is large since we need to calculate the full gradient in every iteration. 

<br />
### GD with momentum

To the best of my knowledge, {% cite rumelhart1986general%} is the first to propose adding momentum term in gradient descent algorithm and have proved that it increased the convergence rate dramatically. The update rule at each iteration is like this:

$$
m_t = -\alpha \nabla f(\theta_t) + \beta m_{t-1}
$$
 
$$
\theta_{t+1} = \theta_t + m_t
$$

Here, $\alpha$ is the step size, or called learning rate. We will use these two terms interchangeably $\beta$ is the momentum parameter. $m_t$ is the modification on $\theta$ at $t$ iteration. The motification of $\theta$ at current iteration depends on both the current gradient and the $\theta$ change at previous iterations. Intuitively, the rationale for the use of the momentum term is that the steepest descent is particularly slow when there is a long and narrow valley in the error function surface. In this situation, the direction of the gradient is almost perpendicular to the long axis of the valley. The system thus oscillates back and forth in the direction of the short axis, and only moves very slowly along the long axis of the valley. The momentum term helps average out the oscillation along the short axis while at the same time adds up contributions along the long axis {% cite rumelhart1986general%}.

To better understand the intuition behind the momentum term, Rauf Bhat provides a clear and simple illustration in his article on [gradient descent with momentum](https://towardsdatascience.com/gradient-descent-with-momentum-59420f626c8f). 

In this update rule, $\alpha$ and $\beta$ are hyperparameters. When $\alpha = 1-\beta$, $m_t$ represents the exponential moving average. Intuitively, we assign a larger weight to the gradient from a more recent iteration. This approach makes sense because gradients are typically more similar when their corresponding points are closer together.


<br />
### SGD with momentum



<br />
### AdaGrad
AdaGrad {% cite duchi2011adaptive%} is also a very important algorithm based on SGD. One of the drawbacks of traditional SGD is its reliance on a global step size. Setting this step size is crucial: if it is too large, the algorithm may oscillate across the function and fail to achieve an acceptable loss, whereas if it is too small, it may take an excessively long time to converge to the optimal solution. AdaGrad addresses this issue by using an adaptive step size, allowing the algorithm to adjust the step size dynamically based on the gradients, thereby improving convergence and performance.

The update rule for AdaGrad is:

$$
g_t = \nabla_{\theta} f(\theta_{t-1})
$$

$$
G_t = G_{t-1} + g_t \odot g_t
$$

$$
\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot g_t
$$

This update rule is a little bit abstract to understand, so I will change it a little bit. For parameter $\theta^{(1)}$, the first parameter in all parameter vector $\theta$, the update rule is:

$$
g_t^{(1)} = \nabla_{\theta^{(1)}} f(\theta_{t-1})
$$

$$
G_t^{(1)} = G_{t-1}^{(1)} + g_t^{(1)} \odot g_t^{(1)}
$$

$$
\theta_t^{(1)} = \theta_{t-1}^{(1)} - \frac{\eta}{\sqrt{G_t^{(1)} + \epsilon}} \odot g_t^{(1)}
$$

If the parameter $\theta^{(1)}$ is frequently updated, indicating it is far from its optimal value, then $G_t^{(1)}$ becomes larger, leading to a smaller learning rate. Conversely, if the parameter $\theta^{(1)}$ is infrequently updated, suggesting it is close to its optimal value, then $G_t^{(1)}$ remains smaller, resulting in a larger learning rate. The ultimate goal is to reduce the number of iterations needed to reach convergence. [An article discuss AdaGrad](https://medium.com/konvergen/an-introduction-to-adagrad-f130ae871827) vividly, and I use it as a reference. 



<br />
### RMSProp
 RMSProp also utilizes adaptive gradients and is essentially an extension of AdaGrad. One of the drawbacks of AdaGrad is that the step size consistently decays as the number of iterations increases. Consequently, the algorithm tends to converge slowly in the later iterations, making the final stages of optimization very slow. RMSProp avoids this problem by using the update rule:

$$
g_t^{(1)} = \nabla_{\theta^{(1)}} f(\theta_{t-1})
$$

$$
G_t^{(1)} = \beta G_{t-1}^{(1)} + (1-\beta) g_t^{(1)} \odot g_t^{(1)}
$$

$$
\theta_t^{(1)} = \theta_{t-1}^{(1)} - \frac{\eta}{\sqrt{G_t^{(1)} + \epsilon}} \odot g_t^{(1)}
$$

The primary difference between AdaGrad and RMSProp is that RMSProp uses an exponential moving average of the squared gradients instead of the equally weighted average employed by AdaGrad. This approach places more emphasis on recent gradient values, rather than equally distributing importance among all gradients from the beginning of training.




<br />
### Adam

Finally, it comes to the Adam algorithm. The update rule is:

$$
t \leftarrow t + 1 \text{ (7a)}
$$

$$
g_t = \nabla_\theta f_t(\theta_{t-1}) \text{ (7b)}
$$

$$
m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t \text{ (7c)}
$$

$$
v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2 \text{ (7d)}
$$

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t} \text{ (7e)}
$$

$$
\hat{v}_t = \frac{v_t}{1 - \beta_2^t} \text{ (7f)}
$$

$$
\theta_t = \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \text{ (7g)}
$$

