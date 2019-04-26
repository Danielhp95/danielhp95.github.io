---
layout: post
cover: assets/images/shiva.jpg
title: "A Review: Policy Gradient Algorithms"
date: 2018-10-9 23:20:29
tags: [RL, Theory-Dive]
---

Policy gradient theorem
-----------------------

Let’s assume an stochastic environment $$\xi$$ from which to sample states
and rewards. Consider a stochastic control policy
$$\pi_{\theta}(a | s)$$[^1] parameterized by a parameter vector $$\theta$$,
that is, a distribution over the action set $$\mathcal{A}$$ conditioned on
a state $$s \in \mathcal{S}$$. $$\theta$$ is a D-dimensional real valued
vector, $$\theta \in \mathbb{R}^{D}$$, where $$D$$ is the number of
parameters (dimensions) and $$D << |\mathcal{S}|$$. The agent acting under
policy $$\pi_{\theta}$$ is to maximize the (possibly discounted)[^2] sum
of rewards obtained inside environment $$\xi$$, over a time horizon $$H$$
(possibly infinitely long). 
Recall that in this [introductory blog to Reinforcement Learning]({{ site.baseurl }}/technical-introduction-to-reinforcement-learning)
we mathematically boiled down Reinforcement Learning as a quest 
to find a divine policy, rightful maximizer of the expected reward obtained in an environment:

$$\pi^{*} = \underset{\pi}{\text{max}}\;  \mathbb{E}_{s_0 \sim \rho_0, s \sim \xi, a \sim \pi(\cdot | s_t)}[\sum_{t=0}^{T} \gamma r_t] \tag{0}$$

When we explicitly parameterize a policy $$\pi_{\theta}$$ by a parameter vector $$\theta$$
we can rewrite the equation above as:
 
$$\label{equation:expected-reward-theta}
    \underset{\theta}{\text{max}} = \mathbb{E}_{s_{t} \sim \xi, a_t \sim \pi_{\theta}}[\sum^{H}_{t=0} r(s_t, a_t) | \pi_{\theta}] \tag{1}$$

There are strong motivations for describing the optimization problem on
the parameter space of a policy instead of the already discussed
approaches:

1.  It offers a more direct way of approaching the reinforcement
    learning problem. Instead of computing the value functions $$V$$ or
    $$Q$$ and from those deriving a policy function, we are calculating
    the policy function directly.

2.  Using stochastic policies smoothes the optimization problem. With a
    deterministic policy, changing which action to execute in a given
    state can have a dramatic effect on potential future rewards[^3]. If
    we assume a stochastic policy, shifting a distribution over actions
    slightly will only slightly modify the potential future rewards.
    Furthermore, Many problems, such as partially observable
    environments or adversarial settings have stochastic optimal
    policies [(Degris 2012)](https://arxiv.org/abs/1205.4839), [(Lanctot 2017)](https://arxiv.org/pdf/1711.00832.pdf). 

3.  Often $$\pi$$ can be simpler than $$V$$ or $$Q$$.

4.  If we learn $$Q$$ in a large or continuous actions space, it can be
    tricky to compute $$\underset{\text{a}}{argmax}\; Q(s,a)$$.

For an episode of length $$H$$ let $$\tau$$ be the trajectory followed by an
agent in an episode. This trajectory $$\tau$$ is a sequence of
state-action tuples $$\tau = (s_0, a_0, \dots, s_H, a_H)$$. We overload
the notation of the reward function $$\mathcal{R}$$ thus:
$$\mathcal{R}(\tau) = \sum_{t=0}^{H}r(s_t, a_t)$$, indicating the total
reward obtained by following trajectory $$\tau$$. From here, the utility
of a policy parameterized by $$\theta$$ is defined as:

$$\label{equation:utility}
U(\theta) = \mathbb{E}_{s_t \sim \xi, a_t \sim \pi_{\theta}}[\sum_{t=0}^{H}r(s_t, a_t) | \pi_{\theta}] = \sum_{\tau}P(\tau ; \theta){R}(\tau) \tag{2}$$

Where $$P(\tau ; \theta)$$ denotes the probability of trajectory $$\tau$$
happening when taking actions sampled from a parameterized policy
$$\pi_{\theta}$$. More informally, how likely is this sequence of
state-action pairs to happen as a result of an agent following a policy
$$\pi_{\theta}$$. Linking equations $$\ref{equation:expected-reward-theta}$$
and $$\ref{equation:utility}$$, the optimization problem becomes:

$$\label{equation:utility-optimization}
\underset{\theta}{\text{max}}\; U(\theta) = \underset{\theta}{\text{max}}\; \sum_{\tau}P(\tau ; \theta)\mathcal{R}(\tau) \tag{3}$$

Policy gradient methods attempt to solve this maximization problem by
iteratively updating the policy parameter vector $$\theta$$ in a direction
of improvement w.r.t to the policy utility $$U(\theta)$$. This direction
of improvement is dictated by the gradient of the utility
$$\nabla_{\theta}U(\theta)$$. The update is usually done via the well
known gradient descent algorithm. This idea of iteratively improving on
a parameterized policy was introduced by [(Williams 1992)](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf) under the
name of *policy gradient theorem*. In essence, the gradient of the
utility function aims to increase the probability of sampling
trajectories with higher reward, and reduce the probability of sampling
trajectories with lower rewards.

$$\label{equation:utility-gradient}
\nabla_{\theta} U(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}} [\nabla_{\theta} \log \pi_{\theta}(\tau) R(\tau)] \\
\tag{4}
$$

Equation $$\ref{equation:utility-gradient}$$ presents the gradient of the
policy utility function. The next section shows the derivation from
equation $$\ref{equation:utility}$$ to equation $$\ref{equation:utility-gradient}$$.


Policy gradient theorem derivation
-----------------------

[Let's jam](https://www.youtube.com/watch?v=n2rVnRwW0h8)

Having defined the utility of a function given a parameter vector $$\theta \in \mathcal{R}^D$$. 
We could solve equation $\ref{equation:utility}$ by following the Utility's gradient wr.t to $\theta$.
Hence, the goal is to find the expression $$\nabla_{\theta} U(\theta)$$ that will
allow us to update our policy parameter vector $$\theta$$ in a direction
that improves the estimated value of the utility of the policy
$$\pi_{\theta}$$. Taking the gradient w.r.t $$\theta$$ gives:


$$
\begin{aligned}
\nabla_{\theta} U(\theta) & = \nabla_{\theta} \sum_{\tau}P(\tau ; \theta) R(\tau) \\
& =  \sum_{\tau} \nabla_{\theta} P(\tau ; \theta) R(\tau) \quad &\text{(Move gradient operator inside sum)} \\
& =  \sum_{\tau} \nabla_{\theta} \frac{P(\tau; \theta)}{P(\tau ; \theta)} P(\tau ; \theta) R(\tau) \quad & (\text{Multiply by} \frac{P(\tau; \theta)}{P(\tau ; \theta)}  )  \\
& =  \sum_{\tau} P(\tau; \theta) \frac{\nabla_{\theta} P(\tau ; \theta)}{P(\tau ; \theta)} R(\tau) \quad & (\text{Rearrange}) \\
& =  \sum_{\tau} P(\tau; \theta) \nabla_{\theta} \log P(\tau ; \theta) R(\tau) \quad & (\text{Note:} \frac{\nabla_{\theta}P(\tau; \theta)}{P(\tau; \theta)} = \nabla_{\theta} \log P(\tau; \theta) ) \\
\nabla_{\theta} U(\theta) & = \mathbb{E}_{\tau \sim \pi_{\theta}} [\nabla_{\theta} \log P(\tau ; \theta) R(\tau)] & (\mathbb{E}[f(x)] = \sum_{x} xf(x))\\
\end{aligned}
$$

This derived equation is *very* similar, but not quite like, equation $\ref{equation:utility-gradient}$:

$$\label{equation:expectance-gradient-partial-derivation} \tag{5}
\nabla_{\theta} U(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}} [\nabla_{\theta} \log P(\tau ; \theta) R(\tau)] ]
$$

This leaves us with an expectation for the term
$$\nabla_{\theta} \log P(\tau ; \theta) R(\tau)$$. Note that as of now we
have not discussed how to calculate $$P(\tau ; \theta)$$, the probability of a trajectory
under a policy $$\pi_{\theta}$$. Let's define just that:

$$P (\tau; \theta) = \Pi_{t=0}^{H} \underbrace{P (s_{t+1} | s_t, a_t)}_\textrm{dynamics models} \underbrace{\pi_{\theta} (a_t | s_t)}_\textrm{policy}$$

Because we **assume** that we are working in a Markovian environment;
The probability of a trajectory happening is nothing more than
the concatenated multiplication of (1) the probability of each action that was taken
(2) The transition probability from each state in the trajectory and its empirical successor.

From here we can calculate the term
$$\nabla_{\theta} \log P(\tau ; \theta)$$ present in
equation $\ref{equation:expectance-gradient-partial-derivation}$:

$$
\begin{aligned}
\nabla_{\theta} \log P(\tau ; \theta) & = \nabla_{\theta} \log [\Pi_{t=0}^{H} P(s_{t+1} | s_t, a_t) \pi_{\theta}(a_t | s_t)] \\
 & = \nabla_{\theta} [(\sum_{t=0}^{H} \log P(s_{t+1} | s_t, a_t)) + (\sum_{t=0}^{H} \log \pi_{\theta}(a_t | s_t))] \\ 
 \nabla_{\theta} \log P(\tau ; \theta)& = \sum_{t=0}^{H} \underbrace{\nabla_{\theta} \log \pi_{\theta}(a_t | s_t)}_\textrm{no dynamics required!}
\end{aligned}$$

Plugging this last derived result of into
equation $\ref{equation:expectance-gradient-partial-derivation}$ we obtain the following
equation for the gradient of the utility function w.r.t to parameter vector $$\theta$$:

$$
\tag{4}
\nabla_{\theta} U(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}} [\nabla_{\theta} \log \pi_{\theta}(\tau) R(\tau)]
$$

[(Sutton 1999)](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf) offers a different approach to this derivation by
calculating the gradient for the state value function on an initial
state $$s_0$$, calculating $$\nabla_{\theta} V_{\pi_{\theta}}(s_0)$$.

A key advantage of the policy gradient theorem, as inspected
by [(Sutton 1999)](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)
is that equation $$\ref{equation:utility-gradient}$$ does not contain any term of the
form $$\nabla_{\theta}P(\tau ; \theta)$$. This means that we don’t need to
model the effect of policy changes on the transition probability
function $$\mathcal{P}$$. Policy gradient methods therefore classify as model-free
methods.

How to actually compute this gradient
-----------------------

It's all well and good when you derive this formula for the first time.
Everybody pats each other's backs and leave the office with a 
satisfied face. But somebody stays behind and asks the real question:
*How do you actually compute the gradient in equation $\ref{equation:utility-gradient}$?*

We can use [Monte Carlo methods](https://arxiv.org/abs/0905.1629) to generate an empirical estimation of
the expectation in equation $$\ref{equation:utility-gradient}$$. This is done
by sampling $$m$$ trajectories under the policy $$\pi_{\theta}$$. This works
even if the reward function $$R$$ is unkown and/or discontinuous, and on
both discrete and continuous state spaces. The equation for the
empirical approximation of the utility gradient is the following:

$$\label{equation:expectance-gradient-vanilla}
\begin{aligned}
\nabla_{\theta}U(\theta) \approx \hat{g} &= \frac{1}{m} \sum_{i = 0}^{m} \nabla_{\theta} \log \pi_{\theta}(\tau^{(i)}) R(\tau^{(i)}) \\
\nabla_{\theta}U(\theta) \approx \hat{g} &= \frac{1}{m} \sum_{i = 0}^{m} \sum_{t=0}^{H-1} \nabla_{\theta} \log \pi_{\theta}(a_t^{(i)} | s_t^{(i)}) (\sum_{k=0}^{H}R(s_t^{(i)}, a_t^{(i)}))
\end{aligned}
\tag{6}$$

The estimate $$\hat{g}$$ is an unbiased estimate and it works in theory.
However it requries an impractical amount of samples, otherwise the
approximation is very noisy (has high variance). There are various
techniques that can be introduced to reduce variance.

It's all about that Baseline
---------

Intuitively, we want to reduce the probability of sampling trajectories
that are worse than average, and increase the probability of
trajectories that are better than average. [(Williams 1992)](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf), 
in the same paper that introduces the policy gradient theorem, explores the
idea of introducing a baseline $$b \in \mathbb{R}$$ as a method of
variance reduction. These authors also prove that introducing a baseline
keeps the estimate unbiased. It is imporant to note that this estimate
is not biased as long as the baseline at time $$t$$ does not 
depend on action $$a_t$$. Introducing a baseline in equation 
$$\ref{equation:utility-gradient}$$ yields the equation:

$$\label{equation:utility-gradient-baseline}
\nabla_{\theta} U(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}} [\nabla_{\theta} \log \pi_{\theta}(\tau) (R(\tau) - b)] \\
\tag{7}$$

The most basic type of baseline is the global average reward, which
keeps track of the average reward across all episodes. We can also add
time dependency to the baseline, such as keeping a running average
reward. [(Greensmith 2004)](http://jmlr.csail.mit.edu/papers/volume5/greensmith04a/greensmith04a.pdf) 
derives the optimal constant value baseline.


Adding a baseline does not bias the gradient estimate
---------

Take a equation $\ref{equation:utility-gradient-baseline}$, 
representing the gradient of the utility of a parameterized 
policy $\pi_{\theta}$ using a *constant* and real valued baseline $b \in \mathbb{R}$:

$$
\begin{aligned}
    \nabla_{\theta} U(\theta) &= \mathbb{E}_{\tau \sim \pi_{\theta}} [\nabla_{\theta} \log P(\tau ; \theta)(R(\tau) - b)] \\
&= \mathbb{E}_{\tau \sim \pi_{\theta}} [\nabla_{\theta} \log P(\tau ; \theta)R(\tau)] - \mathbb{E}_{\tau \sim \pi_{\theta}} [\nabla_{\theta} \log P(\tau ; \theta)b] \quad &\text{(Linearity of expectation)}
\end{aligned}
$$

In order to prove that substracting the baseline $b$ leaves the estimation unbiased, we need to show that the right hand side of the substraction evaluates to $0$. Which is indeed the case:

$$
\begin{aligned}
& \mathbb{E}_{\tau \sim \pi_{\theta}} [\nabla_{\theta} \log P(\tau ; \theta)b] \\
&= \sum_{\tau} P(\tau ; \theta) \nabla_{\theta} \log P(\tau ; \theta)b \quad &(\mathbb{E}[f(x)] = \sum_{x} xf(x))\\
&= \sum_{\tau} P(\tau ; \theta) \frac{\nabla_{\theta} P(\tau ; \theta)}{P(\tau ; \theta)}b \quad & (\text{Note: } \nabla_{\theta} \log P(\tau; \theta) = \frac{\nabla_{\theta}P(\tau; \theta)}{P(\tau; \theta)} )\\
&= \sum_{\tau} 1 \times \nabla_{\theta} P(\tau ; \theta)b \quad & (\frac{P(\tau ; \theta)}{P(\tau ; \theta)} = 1)\\
&= b \sum_{\tau} \nabla_{\theta} P(\tau ; \theta) \quad &\text{(Move baseline outside of summation)}\\
&= b \nabla_{\theta} (\sum_{\tau} P(\tau ; \theta)) \quad &\text{(Move gradient operator outside of summation)}\\
&= b \times 0 = 0 \quad & \text{(Apply gradient operator)}
\end{aligned}
$$

Thus, adding a baseline leaves the gradient of the policy utility unbiased!

Improvements over constat Baselines
---------

Furthermore, it is not optimal to scale the probability of taking an
action by the whole sum of rewards. A better alternative is, for a given
episode, to weigh an action $$a_t$$ by the reward obatined from time $$t$$
onwards, otherwise we would be ignoring the Markov property underlying
the environment’s Markov Decission Process by adding history dependency.
Removing the terms which don’t depend on the current action $$a_t$$
reduces variance without introducing bias. This changes
equation $$\ref{equation:utility-gradient-baseline}$$ to:

$$\label{equation:utility-gradient-baseline-temporal}
\begin{aligned}
    \nabla_{\theta} U(\theta) & = \mathbb{E}_{s_t \sim E, u_t \sim \pi_{\theta}} [\nabla_{\theta} \sum_{t=0}^{H-1}\log \pi_{\theta}(a_t \mid s_t) (\sum_{k=t}^{H-1}R(s_k, a_k) - b)] \\
\end{aligned}
\tag{8}$$

A powerful idea is to make the baseline state-dependent
$$b(s_t)$$ [(Baxter2001)](https://arxiv.org/pdf/1106.0665.pdf). For each state $$s_t$$, this baseline should
indicate what is the expected reward we will obtain by following policy
$$\pi_{\theta}$$ starting on state $$s_t$$. By comparing the empirically
obtained reward with the estimated reward given by the baseline
$$b(s_t)$$, we will know if we have obtained more or less reward than
expected. Note how this baseline is the exact definition of the state
value function $$V_{\pi_{\theta}}$$, as shown in
equation $$\ref{equation:baseline-state-dependent}$$. This type of baseline
allows us to increase the log probability of taking an action
proportionally to how much its returns are better than the expected
return under the current policy.

$$\label{equation:baseline-state-dependent}
b(s_t) = \mathbb{E}[r_t + r_{t+1} + r_{t+2} + \cdots + r_{H-1} \mid \pi_{\theta}] = V_{\pi_{\theta}}(s_t)
\tag{9}$$

The term $$\sum_{k=t}^{H-1}R(s_k, a_k)$$ can be regarded as an estimate of
$$Q_{\pi_{\theta}}(s_t, a_t)$$ for a single roll out. This term has high
variance because it is sample based, where the amount of variance
depends on the stochasticity of the environment. A way to reduce
variance is to include a discount factor $$\gamma$$, rendering the
equation: $$\sum_{k=t}^{H-1} \gamma^k  R(s_k, a_k)$$. However, this
introduces a slight bias. Even with this addition, the estimation
remains sample based, which means that it is not generalizable to unseen
state-action pairs. This issue can be solved by using function
approximators to approximate the function $$Q_{\pi_{\theta}}$$. We can
define another real valued parameter vector $$\phi \in R^F$$, where $$F$$ is
the dimensionality of the parameter vector. From here, we can use $$\phi$$
to parameterize the function approximator $$Q^{\phi}_{\pi_{\theta}}$$.
This function will be able to generalize for unseen state-action pairs.

$$
\begin{equation}
\begin{aligned}
Q^{\phi}_{\pi_{\theta}}(s,a) & = \mathbb{E}[r_0 + r_1 + r_2 + \cdots + r_{H-1} \mid s_0 = s, a_0 = a] \quad & (\infty\text{-step look ahead}) \\
                      & = \mathbb{E}[r_0 + V^{\phi}_{\pi_{\theta}}(s_1) \mid s_0 = s, a_0 = a] \quad & (1\text{-step look ahead}) \\
                      & = \mathbb{E}[r_0 + + r_1 + V^{\phi}_{\pi_{\theta}}(s_2) \mid s_0 = s, a_0 = a] \quad & (2\text{-step look ahead})
\end{aligned}
\end{equation}
\label{equation:q-n-step-lookahead}
\tag{10}
$$

Deciding on how many steps into the future to use for the state-value
function $$Q_{\pi_{\theta}}^{\phi}(s, a)$$ entails a variance-bias
tradeoff. The more actual sampled rewards $$r_t$$ used in our state-value
function estimation, the more variance is introduced, whilst reducing
the variance from the function approximator.

Notice how we use the parameter vector $$\phi$$ to approximate the state
value function $$V_{\pi_{\theta}}$$. This approach can be viewed as an
actor-critic architecture where the policy $$\pi_{\theta}$$ is the actor
and the baseline $$b_t$$ is the critic.

Advantage function and Generalized Advantage Function (GAE)
-----------------------------------------------------------
[Nice post about TD error vs Advantage vs Bellman error](http://boris-belousov.net/2017/08/10/td-advantage-bellman/)

Finally, let's introduce the The advantage function.
The advantage function is defined as $$A_{t}(s_t, a_t) \in \mathbb{R}$$
and it denotes how much better or worse the result of taking a specific action
at a specific state is, compared to the policy’s default behaviour. 
This is captured by the expression:

$$A_{t}(s_t, a_t) = Q_{\pi}(s_t, a_t) - V_{\pi}(s_t)$$

Using the advantage function inside of the policy gradient estimation
yields almost the lowest variance, although this equation is not known
in practice, and must be estimated. This can be done, as mentioned
before, by approximating the function $$V_{\pi_{\theta}}$$.

 [(Schulman 2015)](https://arxiv.org/abs/1506.02438) introduces a very smart idea, which generalizes
the $$n$$-step lookahead of equation $\ref{equation:q-n-step-lookahead}$.
Instead of deciding on a single value for the number of lookahead steps,
it is possible to take into account *all* of them simultaneously. Let’s
define $$A^n_{\pi}$$ as the $$n$$-step lookahead advantage function.
[(Schulman 2015)](https://arxiv.org/abs/1506.02438) introduces the generalized advantage estimation
(GAE), parameterized by the discount factor $$\gamma \in [0,1]$$ and a
special parameter $$\lambda \in [0,1]$$, which is used to manually tune
yet another variance-bias tradeoff.

$$A^{GAE(\gamma, \lambda)}_{t} = (1 - \lambda) (A^1_{t} + \lambda A^2_{t} + \lambda^2 A^3_{t} + \dots)$$

By choosing low values for $\lambda$, we are biasing the estimation of the
advantage function towards low values of $$n$$ for all $$n$$-step lookahead
cases, reducing variance and increasing bias. If we use a higher value
for $$\lambda$$, we increase the weight of the higher $$n$$ values of the
$$n$$-step lookahead cases. The GAE can be analytically written as:

$$A_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V$$

Where $$\delta_{t}^V = r_t + \gamma V_{\pi}(s_{t+1}) - V_{\pi}(s_t)$$ is
the TD residual for a given policy $$\pi$$ as introduced
in [(Sutton 1999)](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf).
 There are two notable cases of this formula,
obtained by setting $$\lambda = 0$$ and $$\lambda = 1$$:

$$
\begin{aligned}
   GAE(\gamma, 0):  A_t &= \delta_{t+l}^V &= r_t + \gamma V(s_{t+1}) - V(s_t)\\
   GAE(\gamma, 1):  A_t &= \sum_{l=0}^{\infty} \gamma^l \delta_{t+l}^V &= \sum_{l=0}^{\infty} \gamma^l r_{t+l} - V(s_t)
\end{aligned}
$$

The first one is the TD error. The second one is the sample based estimation of $Q_{\pi_{\theta}}(s_t, a_t)$

Policy gradient equation summary
--------------------------------

In summary, policy gradient methods maximize the expected total reward
by repeatedly estimating the policy’s utility gradient
$$g = \nabla_{\theta} \mathbb{E}[\sum_{t=0}^{\infty}r_t]$$. There are many
expresessions for the policy gradient that have the form:

$$g = \mathbb{E}[\sum_{t=0}^{\infty} \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t) \Psi]$$

The variable $$\Psi$$ can be one of the following:

1.  $$\sum_{t=0}^{\infty} r_t$$: total trajectory reward.

2.  $$\sum_{t'=t}^{\infty} r_{t'}$$: reward following action $$a_t$$.

3.  $$\sum_{t'=t}^{\infty} r_{t'} - b(s_t)$$: baseline version.

4.  $$Q_{\pi}(s_t, a_t)$$: state-action value function.

5.  $$A_{\pi}(s_t, a_t)$$: Advantage function.

6.  $$\delta_t^V = r_t + V_{\pi}(s_{t+1}) - V_{\pi}(s_t)$$: TD residual.

7.  $$A_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V$$:
    GAE

Out of the 7 different possibilities, state of the art algorithms use
GAE($$\gamma$$,$$\lambda$$), as it has been proved empirically and loosely
theoretically that it introduces an "acceptable" amount of bias, whilst
being one of the methods that most reduces variance.

[^1]: Some researchers prefer the notation $$\pi(\cdot, \theta)$$,
    $$\pi(\cdot \mid \theta)$$ or $$\pi(\cdot; \theta)$$. These notations
    are equivalent.

[^2]: [(Williams 1992)](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf),
      [(Sutton 1999)](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf) present proofs of this same
    derivation using a discount factor, which makes policy gradient
    methods work for environments with infinite time horizons.

[^3]: An example of this concept are *greedy* or $$\epsilon$$-*greedy*
    policies derived thus:
    $$\pi(s) = \underset{a \in \mathcal{A}}{\text{argmax}} Q(s,a)$$.
