---
layout: post
cover: assets/images/shiva.jpg
title: "A Historical Introduction: Self-Play in Reinforcement Learning"
date: 2019-4-26 12:47:21
tags: [RL, Self-Play]
author: Daniii
---

Introduction 
------------

In the classical single agent reinforcement learning (RL) scenarios
described by [(Sutton 1999)](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf), [where a stationary environment is
modelled by a Markov Decision Process (MDP)]({{ site.baseurl }}/technical-introduction-to-reinforcement-learning), a solution concept can be
defined. MDPs are solved by computing a policy which yields the highest
possible episodic reward. However, it is not clear how to define a
pragmatic solution concept when training a single policy in a
multi-agent system, for an agent's optimal strategy is dependant on
behaviours of the other agents that inhabit the environment. An initial
solution is to compute the expected reward obtained by a given policy
defined over the *entire* set of all possible other policies in the
environment. Discouragingly, this policy set may not only be
computationally intractable to process, it may even be infinite, if
stochastic policies are allowed.

To approximate this solution, an existing family of multi-agent RL
(MARL) methods train and benchmark a policy against a set of preexisting
fixed agents, using as a success metric the relative performance against
these agents. These methods rest on two assumptions. Firstly, the
availability of benchmarking policies at training and testing time.
Secondly, these existing policies dominate, in a game theoretical sense,
most of the policy space. Thus it would not be necessary to compute the
expectation over the entire policy space, using as a proxy an
expectation over the existing policies.

Within the field of RL there are multiple methods for computing these
benchmarking policies which must be available before training commences.
To name a few, these preexisting policies can be computed using
supervised learning on datasets of expert human moves to bias learning a
policy towards expert human
play [(Silver 2016)](http://clm.utexas.edu/compjclub/wp-content/uploads/2016/10/silver2016.pdf)
[(Tesauro 1992)](https://papers.nips.cc/paper/465-practical-issues-in-temporal-difference-learning.pdf); they can be
tree-search based algorithms using hand-crafted evaluation functions or
Monte Carlo based approaches if an environment model is
present [(Browne 2012)](http://mcts.ai/pubs/mcts-survey-master.pdf). Some methods are as creative as
deriving a strong policy by using off-policy methods on video
replays [(Aytar 2018)](https://arxiv.org/abs/1805.11592)[(Malysheva 2018)](http://ala2018.it.nuigalway.ie/papers/ALA_2018_paper_32.pdf).

What about the cases in which we don't have access to these learning
resources? Such as when developing a new game for which no prior expert
information is known, and for which any hand-crafted evaluation
functions yields a fruitless policy. A priori methods such as optimistic
policy initialization are still permitted [(Machado 2014)](https://webdocs.cs.ualberta.ca/~machado/publications/optimistic.pdf).
 Yet, under such constraints, there is little room to compute a set of good
benchmarking policies, let alone a set of dominating policies.

Authors such as [(Samuel 1959)](https://researcher.watson.ibm.com/researcher/files/us-beygel/samuel-checkers.pdf)
 began experimenting on self-play (SP).
SP is a training scheme which arises in the context of multi-agent
training. A SP training scheme trains a learning agent *purely* by
simulating plays with itself, or with policies which have been generated
during training. These generated policies can dynamically build a set of
benchmarking policies during training. Such set can potentially be
curated to remove dominated or redundant policies.

Historically, SP lacks a formal definition, and notation is often not
shared among researchers. This has led to isolated, and sometimes
conflicting, conceptions of what constitutes SP as a training scheme in
MARL. Wouldn't it be *nice* to have a formally-grounded framework with
rigorous and unified notations to allow for the creation of more nuanced 
and efficient contributions to Self-Play research? A shared language to 
express incremental efforts on existing and future work? (Foreshadowing for a future post ;) ) 

Self-Play throughout the ages
-----------------------------

The notion of SP has been present in the game playing AI community for
over half a century. [(Samuel 1959)](https://researcher.watson.ibm.com/researcher/files/us-beygel/samuel-checkers.pdf) 
discusses the notion of learning
a state-value function to evaluate board positions in the game of
checkers, to later inform a 1-ply tree search algorithm to traverse more
effectively the game's search space. This learning process takes place as the
opponent uses the same state-value function, both playing agents
updating simultaneously the shared state-value function. Such training
fashion was named self-play. The TD-Gammon algorithm [(TD-Gammon)](http://www.bkgm.com/articles/tesauro/tdl.html)
featured SP to learn a policy using TD($$\lambda$$) [(Sutton 1998)](https://researcher.watson.ibm.com/researcher/files/us-beygel/samuel-checkers.pdf) 
to reach expert level backgammon play. This approach surpassed previous
work by the same author, which derived a backgammon playing policy by
performing supervised learning on expert datasets [(Tesauro 1990)](http://www.bkgm.com/articles/tesauro/NeurogammonANeuralNetworkBackgammonProgram.pdf).
More recently, AlphaGo [(Silver 2016)](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf) 
used a combination of supervised learning on expert moves and SP to beat the world champion Go
player. This algorithm was later refined [(Silver 2017)](https://deepmind.com/blog/alphago-zero-learning-scratch/), removing
the need for expert human moves. A policy was learnt purely by using an
elaborate mix of supervised learning on moves generated by SP and MCTS,
as presented in [(Anthony 2017)](https://arxiv.org/abs/1705.08439). These works echo the sentiment that
superhuman AI needs not be limited or biased by preexisting human
knowledge.

In the game of Othello, [(VanDerRee 2013)](https://www.researchgate.net/publication/236645828_Reinforcement_Learning_in_the_Game_of_Othello_Learning_Against_a_Fixed_Opponent_and_Learning_from_Self-Play) experimented with training
single agent RL algorithms using two different training schemes: SP and
training versus a fixed opponent. Their results show that, depending on
the RL algorithm used, learning by SP yields a higher quality policy
than learning against a fixed opponent. Concretely, TD($$\lambda$$) learnt
best from self-play, but Q-learning performed better when learning
against a fixed opponent. Similarly, [(Firoiu 2017)](https://arxiv.org/abs/1702.06230) found that
DQN [(Mnih 2013)](https://arxiv.org/abs/1312.5602), a deep variant of Q-learning, did not perform well
when trained against other policies which were themselves being updated
simultaneously, but otherwise performed well when training against fixed
opponents. The environments used for their experiments differ too much
to draw parallel conclusions from their results, one of them being a
board game and the other a fast-paced fighting video game.

It is often assumed that a training scheme can be defined as SP if, and
only if, all agents in an environment follow the same policy,
corresponding to the latest version of the policy being trained. Meaning
that, when the learning agent's policy is updated, every single agent in
the environment mirrors this policy update. [(Bansal 2017)](https://arxiv.org/abs/1710.03748) relaxes
this assumption by allowing some agents to follow the policies of
"past-selves". Instead of replicating the same policy over all agents,
the policy of all of the non-training agents can *also* come from a set
of *fixed* "historical" policies. This set is built as training
progresses, by taking *checkpoints*[^1] of the policy being trained. At
the beginning of a training episode, policies are uniformly sampled from
this "historical" policy set and define the behaviour of some of the
environment's agents. The authors claim that such version of SP aims at
training a policy which is able to defeat random older versions of
itself, ensuring continual learning.

From this scenario, consider the following: each *combination* of fixed
policies sampled as opponents from the "historical" dataset can be
considered as a separate MDP. This is because by leaving a single agent
learning in a stationary environment, the fixed agents' influence on the
environment is stationary [(Laurent 2011)](https://www.researchgate.net/publication/220301660_The_world_of_Independent_learners_is_not_Markovian). This is of genuine
importance, given that most RL algorithms' convergence properties
heavily rely on the assumption of a stationary
environment [(Asai 2001)](https://arxiv.org/abs/1810.05587). Self-play algorithms can leverage the
assumption that they are using SP, so they can provide the learning
agent with a label denoting which combination of agent behaviours
inhabits the environment, a powerful assumption in transfer
learning [(Sutton 2007)](http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Publications_files/tracking.pdf) 
and multi-task learning [(Taylor 2009)](https://www.cs.utexas.edu/~ai-lab/pubs/JMLR09-taylor.pdf). In
fact, there already are multitask meta-RL algorithms which assume
knowledge of a distribution over MDPs which the agent is being trained
on, such as RL$$^2$$ [(Duan2016)](https://arxiv.org/abs/1611.02779). Note that a SP algorithm featuring a
growing set of "historical" policies will introduce a non-stationary
distribution over the policies that will inhabit the environment during
training. It ensues that the distribution over the set of MDPs, that the
training agent will encounter, becomes non-stationary.

Similar ideas have also been independently discovered in other fields.
Some methods in computational game theory directly tackle the idea of
computing a strong policy[^2] by iteratively constructing a set of
monotonically stronger policies. In turns, iterated best responses
better challenge the current policy to compute better responses to
those, thus generating a stronger policy. Alternating fictitious
play [(Berger 2007)](https://www.researchgate.net/publication/4975577_Brown's_original_fictitious_play) iteratively computes a best response over set of
policies that the learning agent expects the opponent agents to
use. [(Lanctot 2017)](https://www.researchgate.net/publication/331477508_A_Unified_Game-Theoretic_Approach_to_Multiagent_Reinforcement_Learning) devised a unifying game theoretical framework to
capture this iterative best-response computation over a set of
potential, or previously encountered, opponent agents. 

In psychology, [(Treutwein1995)](https://www.researchgate.net/publication/14605368_Adaptive_Psychophysical_Procedures) 
introduces the *adaptive staircase 
procedure*, where a learning agent is presented with a set of
increasingly difficult tasks. After multiple successful trials at a
task, the agent is promoted to harder tasks, otherwise it is demoted to
easier ones. Such procedure was shown to prevent catastrophic
forgetting [^3] on trials outside its current level of difficulty,
linking their results with [(Bansal 2017)](https://arxiv.org/abs/1710.03748) and SP. This was
empirically demonstrated in the deep RL architecture
UNREAL [(Beattie2016)](https://deepmind.com/blog/reinforcement-learning-unsupervised-auxiliary-tasks/) for virtual visual acuity
tests [(Leibo2018)](https://arxiv.org/abs/1801.08116).

Unfortunately, the numerous empirical successes which motivate SP as a
promising training scheme suffer from lack of formal proofs of
convergence or even rate thereof [(Tesauro 1992)](https://papers.nips.cc/paper/465-practical-issues-in-temporal-difference-learning.pdf). One can only hope that this will change in the future. 

Actually, one can do much more than that. One can research, experiment 
and try out new and old things to further our shared understanding of 
the limitations and strengths of Self-Play. Reach me on my [(Github)]() 
if you are interested in collaborating on precisely anything of what you've just read.

[^1]: For deep RL, this is equivalent to freezing the weights of the
    neural networks used as part of the algorithm.

[^2]: The notion of a policy in RL is roughly equivalent to that of a
    strategy in game theory, the term policy is used for consistency.

[^3]: In a multi-agent reinforcement learning context, catastrophic
    forgetting refers to the event of a policy dropping in performance
    against policies for which it used to perform favourably better.
