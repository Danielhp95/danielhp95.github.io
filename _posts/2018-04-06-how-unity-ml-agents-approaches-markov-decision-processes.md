---
layout: post
title: How Unity ML-agents approaches Markov Decision Processes (MDPs)
tags: [Unity, Reinforcement Learning, Markov Decision Processes]
---


<p style="text-align: center; font-style: italic;">This article was written for Unity ML-agents v0.3</p>

At the end of 2017, Unity released their own [module for reinforcement learning](https://blogs.unity3d.com/2017/09/19/introducing-unity-machine-learning-agents/). They call it Unity ML-agents, but really it's just about reinforcement learning agents. If you know anything about reinforcement learning, you know that the environments where agents act are modelled as [Markov Decision Processes](https://en.wikipedia.org/wiki/Markov_decision_process). Leaving the Markov property aside, the idea of an MDP is simple. There is an environment with a set of states and possible actions that transition one state to another. At every timestep $$t$$, the agent takes one of these actions which modify the environment and in turn recieves an immediate reward linked to the state and action. The goal of the agent is to maximize the cummulative reward that it recieves as it interacts with the environment.

However, Unity doesn't present us with an exact concept of an MDP, at least not too clearly. This ultimately works in their favour, as it allows them to add a few levels of abstraction to the basic MDP. To name a few:  MDPs are inherently single agent (another one?). They also provide a nice enough interface to switch the controll (the brain) of an agent to the Player, the built in Unity code or to external processes. This is precisely how they allow to debug an AI using user testing, to train it with external processes, and to embed it inside the game. Pretty neat.

On the left hand side of the diagram below we have the classical graphical representation of an MDP. On the right hand side, we have the framework that Unity presents us with. Even though these two concepts are really similar, the transition from one to the other wasn't 100% clear to me at first. This article investigates exactly how one relates to one another. Disclaimer: this is a high-level  discussion and the level of technical depth won't be consistent.

 
![mdp-and-unity]({{ site.baseurl }}/assets/img/posts/MDP-and-Unity-ml-agents.png)


For this we need to understand what are the exact components of an MDP. We not only need to know the relation between those components, but also how they are _not_ related. We need to define their boundaries. On a world where encapsulation is a thing, knowing the realm of things is important.

## Elements of an MDP:
### Environment
$$P(s_{t+1} | s_t, a_t)$$
**: Transition probability function**. It represents the probability of transitioning to state $$s_{t+t}$$ if at state $$s_t$$ the agent took action $$a_t$$. Typically a state $$s$$ is represented as a vector $$ s \in \mathbb{R}^D $$.

$$R(s_t, a_t, s_{t+1})$$
**: Reward function**. Represents the immediate reward obained by carrying out action $$a_t$$ at state $$s_t$$, changing the environment state to $$s_{t+1}$$. A reward $$r$$ is always a real valued $$r \in \mathbb{R}$$

### Agent
$$ \pi(a_t | s_t) $$ **: Policy**. The function that fully defines the agent's behaviour. A mapping from states to actions (or distributions over actions in the stochastic case, but that distiction is not important today). 

<figure>
    <img src="{{ site.baseurl }}/assets/img/posts/rl-loop-equations.png" alt="my alt text"/>
    <figcaption >Reinforcement learning loop for a Markov Decision Process.</figcaption>
</figure>


Given the elements of an MDP and the diagram above, we can sketch out a bit of pseudocode for an episode in an Markov decision Process.

## Reinforcement learning loop pseudocode algorithm

1. Initialize environment and agents.<br/><br/>
Repeat 2-5 until termination. t= 0.
2. check for termination.
3. Agent receives state $$s_t$$ and reward $$r_t$$
4. Agent decides on action $$a_{t+1} = \pi(s_t)$$.
5. Agent executes action $$a_{t+1}$$, modifying the environment.

$$_t$$

## Elements of Unity ML-agents


In order to differentiate between the three elements in Unity ML-agents, I'll use distinct colours to represent weather a function / property belongs to an <span class="redtag">academy</span>, <span style="color:green">brain</span> or <span style="color:blue">agent</span>.
* <span style="color:blue">**Agent**</span>: each Agent can have a unique set of states and observations, take unique actions within the environment, and receive unique rewards for events within the environment. An agent’s actions are decided by the brain it is linked to.
* <span style="color:green">**Brain**</span>: Each Brain defines a specific state and action space, and is responsible for deciding which actions each of its linked agents will take. As a technical note, the `Brain` class is an intermediary class between the <span style="color:red">Academy</span> and the <span style="color:blue">agents</span>. The `Brain` delegates the heavy lifting to the `CoreBrains`. These `CoreBrains` can be:
    * `External`, which gather *state*  and *reward* information from Unity <span style="color:blue">agents</span>, send it to an external Python process which decides on an action, and then communicates that decision to all agents.
    * `Internal`, *state* and *reward* information is gathered from agents and given to a trained RL model as input, which spits out an action that is given to the agents to execute. The key is that the model is embedded inside the Unity application. 
    * `Player`, no information is gathered from the agents linked to the brain. The player decides on an input by herself.
* <span style="color:red">**Academy**</span>: The Academy object within a scene also contains as children all Brains within the environment. Each environment contains a single Academy which defines the scope of the environment, in terms of:
    * Engine Configuration – The speed and rendering quality of the game engine in both training and inference modes.
    * Frameskip – How many engine steps to skip between each agent making a new decision.
    * Global episode length – How long the episode will last. When reached, all agents are set to done.


## Dwelving into the code


Now that we have briefly described both the formal mathematical elements that define an MDP and the 3 hierarchical tools provided by Unity ML-agents, let's study how they are related.  This section will cover the main loop of Unity ML-agents, which is embedded in the function <span style="color:red">`RunMdp()`</span>. Remember the color convention for <span style="color:red">academy</span>, <span style="color:green">brain</span> and <span style="color:blue">agent</span> function and variables.

[Let's jam](https://www.youtube.com/watch?v=n2rVnRwW0h8)

First, we will open the <span style="color:red">`Academy.cs`</span> file. Notice how <span style="color:red">`FixedUpdate()`</span> almost exclusively handles a call to <span style="color:red">`RunMdp()`</span> method. At runtime, Unity's [FixedUpdate() method](https://docs.unity3d.com/ScriptReference/MonoBehaviour.FixedUpdate.html) is called every fixed framerate, which means that hopefully the underlying MDP will go through its loop on a fixed framerate. That is, it will broadcast it's current state and reward for the agent's previous action, the agent will decide on an action to take, and it will be processed by the environment at a fixed rate.

{% highlight c# %}

// The FixxedUpdate() method, snippet from Academy.cs
void FixedUpdate() {
   if (acceptingSteps) // Makes sure the Academy has finished initialization
   {
       RunMdp();
   }
}
{% endhighlight %}

<span style="color:red">`RunMdp()`</span> Contains the initialization, running, termination and reset operations for the whole MDP. So it shoudn't be surprising to find a really long and complicated function. I am taking the liberty to present you with a simplified version.

{% highlight c# %}

// (Simplified!) snippet from Academy.cs
void RunMDP() {
    // Initialization code
    if (skippingFrames) { // Is it time to act or use same action
        Step()
        DecideAction()
    }
    AcademyStep()
    brain.Step()
}
{% endhighlight %}
The <span style="color:red">`skippingFrames`</span> boolean present inside the `if` statement depends on the <span style="color:red">academy</span> hyperparameter <span style="color:red">`framesToSkip`</span>, which represents the number of frames that the <span style="color:red">academy</span> will wait per call to <span style="color:red">`Step()`</span> and <span style="color:red">`DecideAction()`</span>. However, the user defined function <span style="color:red">`AcademyStep()`</span> and the  <span style="color:green">brain</span> function <span style="color:green">`Step()`</span> will be called on *every* timestep. <span style="color:green">`Step()`</span> invokes agent's <span style="color:blue">`AgentStep()`</span> function, which carries out the last computed action. This means that an action computed inside <span style="color:red">`DecideAction()`</span> will be carried out for <span style="color:red">`skippingFrames`</span> consecutive timesteps, but the decision on what action to take happens every <span style="color:red">`framesToSkip`</span> timesteps. The name <span style="color:red">`framesToSkip`</span> is not great, perhaps a better alternative would be <span style="color:red">`framesBeforeDecisionPoint`</span>.

As a side note, this constraint of taking an action per timestep is what differentiates a Markov Decision Process from a Semi-Markov Decision Process (SMDP).

The <span style="color:red">`Step()`</span> function is responsible for various things. On the side, it resets any agent that has finished an episode and sets the reward for that timestep to `0`. Most importantly, it calls the <span style="color:red">`SendState()`</span> function.

<figure class="image-right">
    <img src="{{ site.baseurl }}/assets/img/posts/s-r.png" alt="my alt text"/>
    <figcaption >Reward and state going from environment to agent.</figcaption>
</figure>
<figure class="image-right">
    {% highlight c# %}

    // Sinnpet from Brain.cs
    public void SendState()
    {
        coreBrain.SendState();
    }

    {% endhighlight %}
    <figcaption >Reward and state going from environment to agent.</figcaption>
</figure>

The functionality of <span class="redtag">`SendState()`</span> is analogous to the bottom arrow on the MDP diagram, the one that carries $$ r_t $$ and $$ s_t $$ from the environment over to the agent. The exact implementation of <span style="color:green">`SendState()`</span> (yes, in green, as the academy forwards the call to a <span style="color:green">brain</span> function with the same name) depends on the type of `CoreBrain` being used (`External`, `Internal` or `Player`). But the basic idea is that the <span style="color:blue">agent</span> function <span style="color:blue">`CollectState()`</span> is invoked, which collects the state from the environment. In `C#` terms, the state is represented as a `List<float>`. Remember how we defined a state $$ s $$ as $$ s \in \mathbb{R}^D $$? It is important to iterate the fact that the agent does not receive an state $$ s_t $$ per timestep $$ t $$. The agent *collects* it in the function <span style="color:blue">`CollectState()`</span>, it is not handed over by the environment in the same way it would in a classical MDP scenario. This may seem cumbersome, but it becomes handy when you have different agents in the scene, each one recording different values from the environment. Essentially allowing for different MDPs to be created from a single environment. It also allows the agent to pick and choose whichever elements from the environment that it wants, and leave out other parts it considers irrelevant. The function <span style="color:blue">`CollectState()`</span> is defined by the user.

The reward is then collected by the <span style="color:green">brain</span> function <span style="color:green">`CollectRewards()`</span> which merely retreives the value inside the <span style="color:blue">agent</span> variable <span style="color:blue">`agent.reward`</span>, computed in the previous iteration. Recall that <span style="color:blue">`agent.reward`</span> is set to `0` inside <span style="color:red">`Step()`</span>. Now we have a state-reward pair $$ (s_t, r_t) $$ and we are ready to use the policy $$ \pi $$ to compute the next action!

<figure class="image-left">
    <img class="resizedImage" src="{{ site.baseurl }}/assets/img/posts/agent-box.png" alt="my alt text"/>
</figure>
We now move to the left hand side box of the MDP diagram, the one containing the <span style="color:blue">agent's</span> policy. We want to sample the next action to be taken: $$ a_{t+1} \sim \pi(s_t) $$. This process begins with the <span style="color:red">`academy`</span> invoking the <span style="color:red">`DecideAction()`</span> function. This call triggers a further call to the <span style="color:green">brain</span>'s <span style="color:green">`DecideAction()`</span> function whose implementation, again, depends on the type of `CoreBrain`. For `External`, the action $$a_t$$ is retrieved from an external process to which Unity has given the current state $$s_t$$, reward $$r_t$$ and other meaningful environment information. For `Internal`, Unity takes the current state $$s_t$$ and inputs it to a trained TensorFlow model. After running a forward pass through the model, the action $$a_t$$ is sampled from the output layer of the model (Assumming the model is a neural network). Finally, for a `Player` `CoreBrain`, the action is taken directly from user input, simples!.

By now we have essentially sampled an action from the agent's policy, $$ a_{t+1} \sim \pi(s_t) $$. The sampled action is sent from the <span style="color:green">brain</span> to the <span style="color:blue">agent</span> by calling <span style="color:green">`SendAction()`</span>, which invokes <span style="color:blue">`UpdateAction()`</span> storing the action inside the <span style="color:blue">`agent.StoredAction`</span> property. This property will serve as input to <span style="color:blue">`AgentStep()`</span>, but let's not get ahead of ourselves.

<figure class="image-right">
    <img src="{{ site.baseurl }}/assets/img/posts/environment-box.png" alt="my alt text"/>
    <figcaption ></figcaption>
</figure>
We now arrive at the right hand side box of the classical MDP diagram, the one that samples an state $$ s_{t+1} $$ from the transition probability function $$P$$ and calculates a reward $$r_{t+1}$$ using $$R$$. The last piece of the puzzle.

It isn't until now that the user (you) has some agency over the architechture of the MDP. It is now the turn of the user defined <span style="color:red">`AcademyStep()`</span> function. What is it's purpose? Well, the <span style="color:red">`AcademyStep()`</span> function should be use as part of the transition probability function $$ P(s_{t+1} \mid s_t, a_t) $$. It should encapsulate the part of the transition $$ s_t \to s_{t+1} $$ which is independent of the action $$a_t$$, whatever $$a_t$$ may be. The part of the transition that depends on $$a_t$$ should be encapsulated inside <span style="color:blue">`AgentStep()`</span>.


 Here lies a conceptual difference between the classical MDP definition and Unity's take on it. On the classical MDP definition, the agent sends an action $$ a_t $$ to the environment, and the environment carries $$ a_t$$ out, calculating the reward associated to it. In Unity, this task corresponds to the <span style="color:blue">agent</span>. Specifically, it corresponds to the user defined function  <span style="color:blue">`AgentStep()`</span>.

{% highlight c# %}

public virtual void AgentStep(float[] action)
{
    // Logic for:
    //       - Transition Function P
    //       - Reward Function R
}

{% endhighlight %}


Here's a similar diagram analogous to the classical MDP one presented above, but with color coded function calls instead of mathematical operations. If you've understood the contents of this article, hopefully you can use this final diagram for your future work with the **fun**tastic Unity ML-agents module!

![unity-rl-loop]({{ site.baseurl}}/assets/img/posts/unity-rl-loop.png)

## Bonus: Some questions and some answers.

Here are some questions that came through my head as I was doing the research for this article.
#### *Why is the reward stored in a class field and the state is retreived from a function?*

If we the <span style="color:green">`Brain`</span> sent a decision (an action) to the <span style="color:blue">`Agent`</span> at every timestep, then it could make sense to either keep both reward and state as a class field, or both as a function. It isn't a good coding practice to keep two concepts at the same conceptual level at different levels of accessibility (fix). However, notice that the brain function <span style="color:green">`Step()`</span> is called regardless of whether there is a new action to be carried out. This means that an action needs to be taken at every timestep. Therefore it makes sense to keep the reward as a field, which we add to after performing an action on every frame: <span style="color:blue">agent.reward</span> += $$\mathbb{R}(s_t, a_t, s_{t+1}) $$. (specify that the state only needs to be collected on demand, thus keeping in in a function makes more sense that changing the value of a field on evey frame?)


## Notes
* I like the fact that for a Player Core Brain, the agent's function CollectState() is completely ignored
* The reward function and the reward transition function lives inside <span style="color:blue">`AgentStep()`</span>.
* The transition function $$ P(s_{t+1} \mid s_t, a_t) $$ lives partially in <span style="color:red">`AcademyStep()`</span> and <span style="color:blue">`AgentStep()`</span>. This isn't ideal.
* The reward function $$ R(s_t, a_t, s_{t+1}) $$ should live inside <span style="color:blue">`AgentStep()`</span>.
* The action is computed, but not executed. It is up to the user to implemnt AgentStep in such a way that the action is executed in the environment.
