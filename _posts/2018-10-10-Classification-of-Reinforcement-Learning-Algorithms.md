---
layout: post
cover: 'assets/images/shiva.jpg'
title: Classification of RL algorithms
date: 2017-07-21 04:00:00
tags: [RL]
author: Daniii
---

Every RL algorithm attempts to learn an optimal policy $$\pi^*$$ for a
given environment $$\xi$$. So far, there is not a single algorithm which
is used in every single environment to find an optimal policy. The
choice of algorithm depends on many factors, such as the nature of the
environment, the availability of the underlaying mechanics of the
environment, access to already existing policies and more practical
constraints such as the amount of computational power available. More
importantly, RL algorithms are not, against common belief, black boxes.
They can be modularized and composed together to overcome the weaknesses
of some approaches with the strenghts of others. For this, it is
important to know what type of algorithms exist. Most RL algorithms can
be divided into the following categories, note that not all of them are
mutually exclusive:

On-policy and off-policy algorithms
-----------------------------------

If any RL algorithm can be regarded as a learning function mapping
state-action-reward sequences, also known as paths or trajectories, to
policies (Laurent2011); Esentially all that RL algorithms do is
applying a learning function over paths sampled from the environment
to compute a policy. The key and only difference between on-policy and off-policy
algorithms is the following:

-   On-policy algorithms use the same policy that they are learning to
    sample actions in the environment. A policy $$\pi$$ is both being
    improved overtime[^1] and also used to sample actions
    $$a \sim \pi(s)$$ inside of the environment.

-   Off-policy algorithms use a behavioural policy $$\mu$$ to sample
    actions $$a \sim \mu(s)$$ and paths inside of the environment, and use
    this information to improve a target policy $$\pi$$. 

The learning that takes place in off-policy algorithms
can be regarded as learning from somebody else’s experience,
whilst on policy algorithms focus on learning from an agent’s own experience.

On-policy algorithms dedicate all computational power on learning and
using a single policy $$\pi$$. These methods can therefore focus on
spending the computational resources on applying a learning function for
the sole benefit of improving this policy. With off-policy algorithms it
is possible to dedicate computational time to modifying both the
behavioural policy $$\mu$$ and the policy $$\pi$$ being learnt. The
motivation behind this being that the paths sampled from the environment
using $$\pi$$ won’t necessarily yield the best paths to learn from.
However, by spending some of the computational resources to using, or
even changing, the behavioural policy $$\mu$$, it is possible to generate
more “imformative” trajectories with which we can improve $$\pi$$ through
a learning function.

By freeing computational time from directly improving the target policy
$$\pi$$, it is possible to tackle many other tasks. (Sutton2010) use
sensorimotor interaction with an environment to learn a multitude of
pseudoreward that are used in conjunction with the environment’s reward
signals. (Jaderberg2016) takes this idea further by using an
off-policy algorithms to learn auxiliary extra tasks: immediate reward
prediction[^2] and a separate policy that maximizes the change in the
state representation[^3].

A method to allow algorithms to perform off-policy updates to their
policies is to introduce the notion of an *experience
replay* (Lin1993), which was made famous after the success
of (Mnih2013). An experience replay is a list of experiences, where
each experience is a 5 element tuple
$$<s_t, a_t, r_t, s_{t+1}, a_{t+1}>$$. As an agent acts in an environment
the experience replay is filled. At the time of updating the policy an
experience (or batch of experiences) is sampled uniformly from the
experience replay. This is not the case with algorithms such as
Q-learning, where updates to a value function happen always using the
latest experiences. Because these sampled experiences may have been
generated using a previous policy, experience replay allows for policy
updates to happen in an off-policy fashion.

The idea of the experience replay buffer has been the focus of much
recent research (Schaul2015; Hessel2017). A basic improvement is
to use a *prioritized* experience replay. The difference being that
experiences are not sampled uniformly from the replay buffer.

**Famous on-policy algorithms:** Sarsa (Sutton1998),
[$$Q(\sigma)$$](https://arxiv.org/abs/1703.01327), [Monte Carlo Tree search](http://mcts.ai/pubs/mcts-survey-master.pdf),
[REINFORCE](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf), 
[A3C](https://arxiv.org/pdf/1602.01783).

**Famous off-policy algorithms:** Q-learning, [Deep Q-Network](https://arxiv.org/abs/1312.5602),
[Deterministic Policy Gradient](http://proceedings.mlr.press/v32/silver14.pdf),
[Deep Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971),
[Importance Weighted Actor-Learner Architecture](https://arxiv.org/abs/1802.01561).

Value based
-----------

Value based, also known as critic only methods, rely on deriving a
policy $$\pi$$ from a state value function $$V(s)$$ or a state action value
function $$Q(s,a)$$. These include most if not all of the traditional RL
algorithms. There are various methods for extracting a policy from a
value function. The simplest form of deriving a policy from a value
function is to create a policy that acts greedily with respect to a value function:

$$\label{equation:policy-extraction}
\begin{aligned}
    \forall s \in \mathcal{S}: \quad \pi(s) &= \underset{a}{\text{argmax}} \sum_{s' \in Succ(s)} P(s' \mid s, a) V(s') \quad & (\text{Deriving policy from } V(s)) \\
    \forall s \in \mathcal{S}: \quad \pi(s) &= \underset{a}{\text{argmax}} \;  Q_{\pi}(s,a) \quad & (\text{Deriving policy from } Q(s,a))
\end{aligned}$$

Some algorithms such as Q-learning or SARSA bootstrap a state action value
function $$Q(s,a)$$ towards the value functions of an optimal policy,
$$Q_{\pi^*}(s,a)$$. Temporal Difference (TD) algorithms bootstrap a state
value function $$V(s)$$ towards the value function of an optimal policy,
$$V_{\pi^*}(s)$$. These algorithms have been proved to converge to the
optimal value functions ([under some assumptions]({{ site.baseurl }}/qlearning)), so an optimal policy $$\pi^*$$ can be derived
once the bootstraping process converges.

Other methods, like Value iteration or Policy iteration go through an
iterative loop of *policy evaluation* and *policy improvement*. The
policy evaluation step computes a value function $$V_{\pi}(s)$$ or
$$Q_{\pi}(s,a)$$ for a given a policy $$\pi$$, which is randomized on
initialization. The policy improvement step extracts a new policy $$\pi'$$
from the pre-computed value functions $$V_{\pi}(s)$$ or $$Q_{\pi}(s,a)$$
using equation \[equation:policy-extraction\]. The next iteration of the
loop is computed using $$\pi'$$. This two step process is proved to
converge to both the optimal value function and optimal policy.

**Famous value based algorithms:** Value iteration, Policy iteration,
SARSA, TD(0), TD(1), TD($$\lambda$$) (Sutton1998), Q-learning, DQN
and it’s many variants: Dueling DQN, Distributional DQN, Prioritized DQN
and Double DQN. These can be found in (Hessel2017).

Policy based
------------

Both Policy based and actor-critic methods are covered [in this post]({{ site.baseurl }}/policy-gradient-algorithms-a-review). 
These algorithms do not
extract a policy from a calculated value function. Instead, they
represent a policy $$\pi_{\theta}$$ through a parameter vector
$$\theta \in \mathbb{R}^D$$. By defining both the utility of a policy’s
parameters $$U(\theta)$$ and its gradient w.r.t the policy’s parameters
$$\nabla_{\theta}U(\theta)$$, it is possible to iteratively update the
policy parameters in a direction of utility improvement.

**Famous policy based algorithms:** 
[REINFORCE](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf),
[PPO](https://arxiv.org/abs/1707.06347), [TRPO](https://arxiv.org/abs/1502.05477).

Actor-critic
------------

Policy based only (actor only) and value based only (critic only)
algorithms feature some crippling weaknesses that make them unsuitable
for complex environments. (Konda2000) states some of their key
downsides:

-   **Critic only algorithms**: The goal of any reinforcement learning
    problem is to find a policy. Spending all computational resources on
    calculating a value function (the critic) is an indirect way of
    approaching the problem. Furthermore, most value based algorithms
    only converge under strict assumptions.

-   **Actor only algorithms**: These algorithms rely on estimating the
    gradient of a policy’s performance (the actor) w.r.t the policy’s
    parameters. Gradient estimators tend to have large variance, leading
    to unstable parameter updates and, potentially, lost of convergence
    properties. More philosophically, because new gradients are
    calculated independently from previous gradients, there is no
    “learning” in the sense of understanding and consolidation of
    experience.

Actor-critic methods combine the strong points of both policy based and
value based algorithms, and overcome some of their individual
weaknesses. The critic has an approximation architecture to compute a
value function. This value function is used to update the actor’s policy
parameters in a direction of improvement. This dynamic leads to faster
convergence than actor-only methods because of a significant decrease in
variance in the estimation of the gradient of the policy’s performance.
It also entails better convergence properties than critic-only methods.

 (Konda2000) make the key observation that in actor-critic
algorithms, the actor parameterization and the critic parameterization
should *not* be independent. The choice of critic parameters should be
directly prescribed by the choice of the actor parameters. That is why
all real world applications that implement an actor-critic algorithm use
a single parameterized model (e.g. a neural network) to represent both
the policy (actor) and the value function approximation (critic). This
is the most straight forward way of sharing parameters between actor and
critic.

**Famous actor critic algorithms:** [A3C](https://arxiv.org/pdf/1602.01783),
[PPO](https://arxiv.org/abs/1707.06347), [TRPO](https://arxiv.org/abs/1502.05477),
[ACKTR](https://arxiv.org/pdf/1708.05144).

Model based and model free approaches
-------------------------------------

In RL literature the *model* or the dynamics of an environment is
considered to be the transition function $$\mathcal{P}$$ and reward
function $$\mathcal{R}$$. Model free algorithms aim to approximate an
optimal policy $$\pi^*$$ without explicitly using either $$\mathcal{P}$$ or
$$\mathcal{R}$$ in their calculations.

Model based algorithms are either given a prior model that they can use
for planning (Browne2012), or they learn a
representation via their own interaction with the environment
(Sutton1991, Deisenroth2011). Note that an advantage
of learning a model specifically tailored for an agent, is that you can
choose a representation of the environment that is relevant to the
agent’s decision making process, which can allow you to learn only the 
elements of the environment that influence the agent's learning (Pathak2017). Another advantage of
having a model is that it allows for forward planning, which is the main
method of learning for search-based artificial intelligence.


Time for References!
------------

<div id="refs" class="references">
  
<div id="ref-Laurent2011">
Laurent, Guillaume J.  Matignon, Laëtitia Fort-Piat, N. Le. 2011. *The world of independent learners is not markovian*.
</div>

<div id="ref-Sutton2010">
Sutton, Richard, S Modayil, Joseph Delp, Michael. 2010. *Horde: A Scalable Real-time Architecture for Learning Knowledge from Unsupervised Sensorimotor Interaction Categories and Subject Descriptors*.
</div>

<div id="ref-Jaderberg2016">
Jaderberg, Max Mnih, Volodymyr Czarnecki, Wojciech Marian. 2016. *Reinforcement Learning with Unsupervised Auxiliary Tasks*.
</div>

<div id="ref-Lin1993">
Lin, Long-ji. 1993. *Reinforcement Learning for Robots Using Neural Networks*.
</div>

<div id="ref-Mnih2013">
Mnih, Volodymyr Kavukcuoglu, Koray Silver, David. 2013. *Playing Atari with Deep Reinforcement Learning*.
</div>

<div id="ref-Schaul2015">
Schaul, Tom Quan, John Antonoglou, Ioannis Silver, David. 2015. *Prioritized Experienced Replay*.
</div>

<div id="ref-Hessel2017">
Hessel, Matteo Modayil, Joseph van Hasselt, Hado. 2017. *Rainbow: Combining Improvements in Deep Reinforcement Learning*.
</div>

<div id="ref-Sutton1998">
Richard Sutton, Barto. 1998. *Reinforcement learning: An introduction*.
</div>

<div id="ref-Konda2000">
Konda Vijay, R Tsitsiklis, John N. 2000. *Actor-Critic algorithms* 
</div>

<div id="ref-Sutton1991">
Richard Sutton. 1991. *Dyna, an integrated architecture for learning, planning, and reacting*
</div>

<div id="ref-Browne2012">
Browne Cameron B, Powley Edward, Whitehouse Daniel, Lucas Simon M, Cowling Peter, Rohlfshagen Philipp, Tavener Stephen, Perez Diego, Samothrakis Spyridon, Colton Simon. 2012. *A Survey of Monte Carlo Tree Search Methods*.
</div>

<div id="ref-Deisenroth2011">
Deisenroth Marc, Peter Rasmussen, Carl Edward. 2011. *PILCO: A Model-Based and Data-Efficient Approach to Policy Search*
</div>

<div id="ref-Pathak2017">
Pathak Deepak, Agrawal Pulkit. 2017. *Curiosity-Driven Exploration by Self-Supervised Prediction*
</div>

</div>

[^1]: The notion of improvement overtime is expressed as a monotonic
    increase in the expected reward of an episode
    $$\mathbb{E}_{a \sim \pi_0}[\sum_{t=0}^{\infty}r_t] < \mathbb{E}_{a \sim \pi_1}[\sum_{t=0}^{\infty}r_t]$$

[^2]: This is different from value function estimation because the value
    that the off-policy algorithm is trying to predict is expected
    immediate reward, instead of expected future cummulative reward.

[^3]: Given a matrix of pixels as input, the authors define “pixel
    control” as a separate policy that tries to maximally change the
    pixels in the following state. The reasoning behind this approach is
    that big changes in pixel values may correspond to important events
    inside of the environment.
