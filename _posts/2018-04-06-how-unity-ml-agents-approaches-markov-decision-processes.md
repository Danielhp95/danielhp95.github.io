---
layout: post
title: How Unity ML-agents approaches Markov Decision Processes (MDPs)
tags: [Unity, Reinforcement Learning, Markov Decision Processes]
---

*This article was written for Unity ML-agents v0.3*

At the end of 2017 released their own [module for reinforcement learning](https://blogs.unity3d.com/2017/09/19/introducing-unity-machine-learning-agents/). They call it Unity ML-agents, but really it's just about reinforcement agents. If you know anything about reinforcement learning, you know that problems (the environments) are modelled as [Markov Decision Processes](https://en.wikipedia.org/wiki/Markov_decision_process). Leaving the Markov property aside, the idea of an MDP is simple. There is an environment with a set of states and possible actions that transition one state to another. At each state, the agent takes one of these actions which modify the environment and in turn recieves an immediate reward linked to the state and action. The goal of the agent is to maximize the cummulative reward that it recieves as it interacts with the environment. However, Unity doesn't present us with an exact concept of an MDP, at least not too clearly. This ultimately works in their favour, as this allows them to add a few levels of abstraction to the basic MDP. To name a few:  MDPs are inherently single agent (another one?). They also provide a nice enough interface to switch the controll (the brain) of an agent to the Player, the built in Unity code or to external processes. This is precisely how they allow to debug an AI using user testing, to train it with external processes, and to embed it inside the game. Pretty neat.

On the left hand side of the diagram below we have the classical graphical representation of an MDP. On the right hand side, we have the framework that Unity presents us with. Even though these two concepts are really similar, the transition from one to the other wasn't 100% clear to me at first. This article investigates exactly how one relates to one another. Disclaimer: this is a high-level  discussion and the level of technical depth won't be consistent.

 
(FIX)
![MDP]({{ site.baseurl }}/assets/img/posts/MDP-and-Unity-ml-agents.png)


For this we need to understand what are the exact components of an MDP. We not only need to know the relation between those components, but also how they are _not_ related. We need to define their boundaries. On a world where encapsulation is a thing, knowing the realm of things is important.

## Elements of an MDP:
### Environment
$$P(s_{t+1} | s_t, a_t)$$
**: Transition probability function**. It represents the probability of transitioning to state $$s_{t+t}$$ if at state $$s_t$$ the agent took action $$a_t$$. Typically a state $$s$$ is represented as a vector $$ s \in \mathbb{R}^D $$.

$$R(s_t, a_t, s_{t+1})$$
**: Reward function**. Represents the immediate reward obained by action $$a_t$$ in state $$s_t$$ changing the environment state to $$s_{t+1}$$. A reward $$r$$ is always a real valued $$r \in \mathbb{R}$$

### Agent
$$ \pi(a_t | s_t) $$ **: Policy**. The function that fully defines the agent's behaviour. A mapping from states to actions (or distributions over actions in the stochastic case, but that distiction is not important today). 

![RL-loop-equations]({{ site.baseurl }}/assets/img/posts/rl-loop-equations.png)


Given the elements of an MDP and the diagram above, we can sketch out a bit of pseudocode for an episode in an Markov decision Process.

## Reinforcement learning loop pseudocode algorithm

Repeat for t = 0 until termination
* Initialize environment and agents
* check for termination
* Agent receives state_t and reward_t
* Agent decides on action_t+1 using state_t
* Agent executes action_t+1, modifying the environment
done

## Elements of Unity ML-agents
In order to differentiate between the three elements in Unity ML-agents, I'll use distinct colours to represent weather a function / property belongs to an <span style="color:red">academy</span>, <span style="color:green">brain</span> or <span style="color:blue">agent</span>.
* <span style="color:blue">**Agent**</span>: each Agent can have a unique set of states and observations, take unique actions within the environment, and receive unique rewards for events within the environment. An agent’s actions are decided by the brain it is linked to.
* <span style="color:green">**Brain**</span>: Each Brain defines a specific state and action space, and is responsible for deciding which actions each of its linked agents will take. As a technical note, the `Brain` class is an intermediary class between the <span style="color:red">Academy</span> and the <span style="color:blue">agents</span>. The `Brain` delegates the heavy lifting to the `CoreBrains`. These `CoreBrains` can be:
    * `External`, which gather *state*  and *reward* information from Unity <span style="color:blue">agents</span>, send it to an external Python process which decides on an action, and then communicates that decision to all agents.
    * `Internal`, *state* and *reward* information is gathered from agents and given to a trained RL model as input, which spits out an action that is given to the agents to execute. The key is that the model is embedded inside the Unity application. 
    * `Player`, no information is gathered from the agents linked to the brain. The player decides on an input by herself.
* <span style="color:red">**Academy**</span>: The Academy object within a scene also contains as children all Brains within the environment. Each environment contains a single Academy which defines the scope of the environment, in terms of:
    * Engine Configuration – The speed and rendering quality of the game engine in both training and inference modes.
    * Frameskip – How many engine steps to skip between each agent making a new decision.
    * Global episode length – How long the episode will last. When reached, all agents are set to done.


Now that we have briefly described both the formal elements that define and MDP and the 3 hierarchical tools provided by Unity ML-agents, we are going to make the connection. Let's jam. We will start by diving into the code, don't be afraid.

First, we will open the <span style="color:red">`Academy.cs`</span> file. Notice how <span style="color:red">`FixedUpdate()`</span> almost exclusively handles a call to <span style="color:red">`RunMdp()`</span> method. [The FixedUpdate() method](https://docs.unity3d.com/ScriptReference/MonoBehaviour.FixedUpdate.html) is called every fixed framerate, which means that hopefully the underlying MDP will move a timestep at a fixed framerate.

{% highlight c# %}

// Snippet from Academy.cs
void FixedUpdate() {
   if (acceptingSteps) // Makes sure the Academy has finished initialization
   {
       RunMdp();
   }
}
{% endhighlight %}

<span style="color:red">`RunMdp()`</span> Contains the initialization, running, termination and reset operations for the whole MDP. So it shoudn't be surprising to find a really long and complicated function.

{% highlight c# %}

// (Simplified!) snippet from Academy.cs
void RunMDP() {
    // Initialization code
    if (skippingFrames) { // Is it time to act or use same action
        Step()
        DecideAction()
    }
    AcademyStep()
    foreach (Brain b in brains) {
        brain.Step()
    }
}
{% endhighlight %}

The hyperparameter `framesToSkip` is the number of frames that the Academy will wait per call to <span style="color:red">`Step()`</span> and <span style="color:red">`DecideAction()`</span>. However, the user defined function <span style="color:red">`AcademyStep()`</span> will be called on each timestep. The `Brain` function <span style="color:green">`Step()`</span> which results in a call to agent's function <span style="color:blue">`AgentStep()`</span> will be called each timestep as well. This means that an action is carried out every timestep, but the decision on what action to take happens every `framesToSkip`. For future work, this constraint of taking an action per timestep is what differentiates a Markov Decision Process from a Semi-Markov Decision Process (SMDP).

The <span style="color:red">`Step()`</span> function is responsible for reseting any agent that has finished an episode and initializing the reward for that timestep to 0. Most importantly, it calls the <span style="color:red">`SendState()`</span> function. Its functionality is analogous to the bottom arrow on the MDP diagram, the one that carries $$ r_t $$ and $$ s_t $$ from the environment over to the agent. The exact implementation of <span style="color:green">`SendState()`</span> (yes, in green, as the academy forwards the call to a `Brain` function with the same name) depends on the type of `CoreBrain` being used (`External`, `Internal`, `Player`). But the basic idea is that the function <span style="color:blue">`CollectState()`</span> is called, which collects the state from the environment, a list of `floats`. Remember how we defined a state $$ s $$ as $$ s \in \mathbb{R}^D $$? It is important to iterate the fact that the agent does not receive an state $$ s_t $$ per timestep $$ t $$. The agent *collects* it in the function <span style="color:blue">`CollectState()`</span>. This becomes handy when you have different agents in the scene, each one pillaging different things from the environment. It also allows the agent to pick and choose whichever elements from the environment that it wants, and leave out other parts. The reward is collected by <span style="color:green">`CollectRewards()`</span> which merely stores the value inside <span style="color:blue">`agent.reward`</span> computed in the previous iteration. Now we have a state-reward pair $$ (s_t, r_t) $$ and we are ready to use the policy $$ \pi $$ to compute the next action.




## Notes
* I like the fact that for a Player Core Brain, the agent's function CollectState() is completely ignored
* The reward function and the reward transition function lives inside <span style="color:blue">`AgentStep()`</span>.
* The transition function $$ P(s_{t+1} \mid s_t, a_t) $$ lives partially in <span style="color:red">`AcademyStep()`</span> and <span style="color:blue">`AgentStep()`</span>. This isn't ideal.
* The reward function $$ R(s_t, a_t, s_{t+1}) $$ should live inside <span style="color:blue">`AgentStep()`</span>.
* The action is computed, but not executed. It is up to the user to implemnt AgentStep in such a way that the action is executed in the environment.

## Old stuff
Jekyll supports the use of [Markdown](http://daringfireball.net/projects/markdown/syntax) with inline HTML tags which makes it easier to quickly write posts with Jekyll, without having to worry too much about text formatting. A sample of the formatting follows.

Tables have also been extended from Markdown:

First Header  | Second Header
------------- | -------------
Content Cell  | Content Cell
Content Cell  | Content Cell

Here's an example of an image, which is included using Markdown:

![Image of a glass on a book]({{ site.baseurl }}/assets/img/pexels/book-glass.jpeg)

Highlighting for code in Jekyll is done using Base16 or Rouge. This theme makes use of Rouge by default.

{% highlight js %}
// count to ten
for (var i = 1; i <= 10; i++) {
    console.log(i);
}

// count to twenty
var j = 0;
while (j < 20) {
    j++;
    console.log(j);
}
{% endhighlight %}

Type on Strap uses KaTeX to display maths. Equations such as $$S_n = a \times \frac{1-r^n}{1-r}$$ can be displayed inline.

Alternatively, they can be shown on a new line:

$$ f(x) = \int \frac{2x^2+4x+6}{x-2} $$
