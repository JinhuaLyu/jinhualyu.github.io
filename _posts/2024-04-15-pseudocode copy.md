---
layout: post
title: Stochastic Gradient Descent, SGD with momentum, AdaGrad, RMSProp, and Adam algorithms
date: 2024-06-18
description: A discussion on different stochastic gradient algorithms and why are they efficient
tags: optimization algorithms
categories: sample-posts
pseudocode: true
---

In this post, I would like to talk about different kinds of stochastic gradient descent algorithms and explain why they are efficient.

## Gradient Descent 

Gradient Descent is the basic algorithm to solve unconstrained optimization problems. Suppose we have a general unconstrained optimization problem:

$$
\min f(\theta)
$$

Our goal is to find the optimal $\theta$ that can minimize the objective function. Unlike some simple quadratic functions, most nonlinear functions are difficult to find the minimal directly by using simple math tricks. Gradient Descent is a very useful algorithm in solving this kind of problems. The logic here is to improve the variables a little bit in every iteration. We need many iterations to reach the optimal solution. We use this rule to update $\theta$ at each iteration.

$$
\theta_{t+1} = \theta_t + \alpha g_t
$$

```pseudocode
% Gradient Descent Algorithm
\begin{algorithm}
\caption{Gradient Descent}
\begin{algorithmic}
\WHILE{$\theta_t$ not converge}
    \IF{$$p < r$$}
        \STATE $$q = $$ \CALL{Partition}{$$A, p, r$$}
        \STATE \CALL{Quicksort}{$$A, p, q - 1$$}
        \STATE \CALL{Quicksort}{$$A, q + 1, r$$}
    \ENDIF
\ENDWHILE
\end{algorithmic}
\end{algorithm}
```


This is an example post with some pseudo code rendered by [pseudocode](https://github.com/SaswatPadhi/pseudocode.js). The example presented here is the same as the one in the [pseudocode.js](https://saswat.padhi.me/pseudocode.js/) documentation, with only one simple but important change: everytime you would use `$`, you should use `$$` instead. Also, note that the `pseudocode` key in the front matter is set to `true` to enable the rendering of pseudo code. As an example, using this code:

````markdown
```pseudocode
% This quicksort algorithm is extracted from Chapter 7, Introduction to Algorithms (3rd edition)
\begin{algorithm}
\caption{Quicksort}
\begin{algorithmic}
\PROCEDURE{Quicksort}{$$A, p, r$$}
    \IF{$$p < r$$}
        \STATE $$q = $$ \CALL{Partition}{$$A, p, r$$}
        \STATE \CALL{Quicksort}{$$A, p, q - 1$$}
        \STATE \CALL{Quicksort}{$$A, q + 1, r$$}
    \ENDIF
\ENDPROCEDURE
\PROCEDURE{Partition}{$$A, p, r$$}
    \STATE $$x = A[r]$$
    \STATE $$i = p - 1$$
    \FOR{$$j = p$$ \TO $$r - 1$$}
        \IF{$$A[j] < x$$}
            \STATE $$i = i + 1$$
            \STATE exchange
            $$A[i]$$ with $$A[j]$$
        \ENDIF
        \STATE exchange $$A[i]$$ with $$A[r]$$
    \ENDFOR
\ENDPROCEDURE
\end{algorithmic}
\end{algorithm}
```
````

Generates:

```pseudocode
% This quicksort algorithm is extracted from Chapter 7, Introduction to Algorithms (3rd edition)
\begin{algorithm}
\caption{Quicksort}
\begin{algorithmic}
\PROCEDURE{Quicksort}{$$A, p, r$$}
    \IF{$$p < r$$}
        \STATE $$q = $$ \CALL{Partition}{$$A, p, r$$}
        \STATE \CALL{Quicksort}{$$A, p, q - 1$$}
        \STATE \CALL{Quicksort}{$$A, q + 1, r$$}
    \ENDIF
\ENDPROCEDURE
\PROCEDURE{Partition}{$$A, p, r$$}
    \STATE $$x = A[r]$$
    \STATE $$i = p - 1$$
    \FOR{$$j = p$$ \TO $$r - 1$$}
        \IF{$$A[j] < x$$}
            \STATE $$i = i + 1$$
            \STATE exchange
            $$A[i]$$ with $$A[j]$$
        \ENDIF
        \STATE exchange $$A[i]$$ with $$A[r]$$
    \ENDFOR
\ENDPROCEDURE
\end{algorithmic}
\end{algorithm}
```
