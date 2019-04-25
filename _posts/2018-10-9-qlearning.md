---
layout: post
cover: assets/images/shiva.jpg
title: "A note on: Q learning"
date: 2018-9-4 14:48:15
tags: []
---

The Q-learning algorithm was first introduced by (Watkins1989), and
is arguably one of the most famous, most studied and most widely
implemented methods in the entire field. Given an MDP, Q-learning aims
to calculate the corresponding optimal action value function $$Q^*$$,
following the principle of optimality and the proof of existence of an
optimal deterministic policy in an MDP as described in
Section \[section:markov-decision-processes\]. It is model free,
learning via interaction with the environment, and it is an off-policy
algorithm. The latter is because, even though we are learning the
optimal action value function $$Q^*$$, we can choose any *behavioural*
policy to gather experience from the environment. Researchers
like (Tijsma2017) benchmarked the efficiency of using various
exploratory policies in grid world stochastic maze environments.
Unfortunately, it did not seem that any of the benchmarked exploratory
policies consistently beat any of the others.

Q-learning has been proven to converge to the optimal solution for an
MDP under the following assumptions:

1.  The $$Q^{\pi^*}$$ function is represented in tabular form, with each
    state-action pair represented discretely (Watkins1992). (That is, the underlying MDP is finite)

2.  Each state-action pair is visited an infinite number of
    times (Watkins1989).

3.  The sequence of updates of Q-values has to be monotonically
    increasing
    $$Q(s_i, a_i) \leq Q(s_{i+1}, a_{i+1})$$. (Thrun1993).

4.  The learning rate $$\alpha$$ must decay over time, and such decay must
    be slow enough so that the agent can learn the optimal Q values.
    Expressed formally: $$\sum_{t} \alpha_t = \infty$$ and
    $$\sum_{t} {(\alpha_{t})}^{2} < \infty$$. (Watkins1989)

Initialize $$Q$$ table
$$\forall s \in \mathcal{S} \wedge \forall a \in \mathcal{A}$$,
$$Q(s,a) = 0$$ Sample $$s_0 \sim \rho_0$$ $$s = s_0$$

Q-learning features its own share of imperfections. If there is a
function approximator[^1] in place, (Thrun1993) shows that if the
approximation error is greater than a threshold which depends on the
discount factor $$\gamma$$ and episode length, then a systematic
overestimation effect occurs, negating convergance. This is mainly due
to the joint effort of function approximation methods and the $$argmax$$
operator used in step 7 of the algorithm. On top of
this, (Kaisers2010) introduces the concept of *Policy bias*, which
states that state-action pairs that are favoured by the policy are
chosen more often, biasing the updates. Ideally all state-action pairs
are updated on every step. However, because agent’s actions modify the
environment, this is generally not possible in absence of an environment
model.

Frequency Adjusted Q-learning (FAQL) proposes scaling the update rule of
Q-learning inversely proportional to the likelihood of choosing the
action taken at that step (Kaisers2010). (Abdallah2016)
introduces Repeated Update Q-learning (RUQL), a more promising
Q-learning spin off that proposes running the update equation *multiple
times*, where the number of times is inversely proportional to the
probability of the action selected given the policy being followed.

Time for References!
------------

<div id="refs" class="references">

<div id="ref-Barto2003">

Joseph Watkins. (1989). *Learning from Delayed Rewards* (His PhD thesis!)

</div>

<div id="ref-Watkins1992">
Watkins, Christopher J.C.H. Dayan, Peter. 1992. *Technical Note: Q-Learning*
</div>

<div id="ref-Thrun1993">
Thrun, Sebastian Schwartz, Anton. 1993. *Issues in Using Function Approximation for Reinforcement Learning*
</div>

<div id="ref-Tijsma2017">

Tijsma, Arryon D Drugan, Madalina M Wiering, Marco A. 2017. *Comparing exploration strategies for Q-learning in random stochastic mazes*. 

</div>

<div>
Kaisers, Michael Tuyls, Karl. 2010. *Frequency Adjusted Multi-agent Q-learning*
</div>

<div>
Abdallah, Sherief. 2016. *Addressing Environment Non-Stationarity by Repeating Q-learning Updates*
</div>

</div>

[^1]: With neural networks being the most famous function approximators
    in reinforcement learning at the time of writing.


