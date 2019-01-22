


irl  from  demo

Algorithms for inverse reinforcement learning

https://www.researchgate.net/profile/Changxi_You/publication/330400231_Advanced_Planning_for_Autonomous_Vehicles_Using_Reinforcement_Learning_and_Deep_Inverse_Reinforcement_Learning/links/5c3e13b0299bf12be3c9f540/Advanced-Planning-for-Autonomous-Vehicles-Using-Reinforcement-Learning-and-Deep-Inverse-Reinforcement-Learning.pdf
Advanced Planning for Autonomous Vehicles Using Reinforcement
Learning and Deep Inverse Reinforcement Learning
Changxi You1
Jianbo Lu2 Dimitar Filev3 Panagiotis Tsiotras4
January 8, 2019
Abstract
Autonomous vehicles promise to improve traffic safety while, at the same time, increase fuel efficiency and reduce
congestion. They represent the main trend in future intelligent transportation systems. This paper concentrates on the
planning problem of autonomous vehicles in traffic. We model the interaction between the autonomous vehicle and
the environment as a stochastic Markov decision process (MDP) and consider the driving style of an expert driver as
the target to be learned. The road geometry is taken into consideration in the MDP model in order to incorporate more
diverse driving styles. The desired, expert-like driving behavior of the autonomous vehicle is obtained as follows: First,
we design the reward function of the corresponding MDP and determine the optimal driving strategy for the autonomous
vehicle using reinforcement learning techniques. Second, we collect a number of demonstrations from an expert driver
and learn the optimal driving strategy based on data using inverse reinforcement learning. The unknown reward function
of the expert driver is approximated using a deep neural-network (DNN). We clarify and validate the application of the
maximum entropy principle (MEP) to learn the DNN reward function, and provide the necessary derivations for using
the maximum entropy principle to learn a parameterized feature (reward) function. Simulated results demonstrate the
desired driving behaviors of an autonomous vehicle using both the reinforcement learning and inverse reinforcement
learning techniques.



VIREL:
A Variational Inference Framework for
Reinforcement Learning
Matthew Fellows† Anuj Mahajan† Tim G. J. Rudner Shimon Whiteson
University of Oxford University of Oxford University of Oxford University of Oxford
Abstract
Applying probabilistic models to reinforcement
learning (RL) has become an exciting direction
of research owing to powerful optimisation tools
such as variational inference becoming applicable
to RL. However, due to their formulation, existing inference frameworks and their algorithms
pose significant challenges for learning optimal
policies, for example, the absence of mode capturing behaviour in pseudo-likelihood methods and
difficulties in optimisation of learning objective
in maximum entropy RL based approaches. We
propose VIREL, a novel, theoretically grounded
probabilistic inference framework for RL that
utilises the action-value function in a parametrised
form to capture future dynamics of the underlying
Markov decision process. Owing to its generality,
our framework lends itself to current advances
in variational inference. Applying the variational
expectation-maximisation algorithm to our framework, we show that the actor-critic algorithm can
be reduced to expectation-maximisation. We derive a family of methods from our framework,
including state-of-the-art methods based on soft
value functions. We evaluate two actor-critic algorithms derived from this family, which perform
on par with soft actor critic, demonstrating that
our framework of

1
We evaluate our simple algorithm
against an existing state of the art actor-critic algorithm,
soft actor-critic (SAC), demonstrating similar performance
across a variety of OpenAI gym domains (Brockman et al.,
2016). In the complex, high dimensional humanoid domain,
we outperform SAC.

3.4 Relationship to Soft Actor-Critic and Soft
Q-Learning
We now show that SAC (Haarnoja et al., 2018; Abdolmaleki
et al., 2018) and the related soft Q-learning, (Haarnoja
et al., 2017) algorithms purportedly derived from MERLIN (Levine, 2018) can be shown to arise from a model
that is closer to VIREL. Consider now the MERLIN lower
bound (Levine, 2018),
















Recall Traces: Backtracking Models for Efficient Reinforcement Learning
Anirudh Goyal, Philemon Brakel, William Fedus, Timothy Lillicrap, Sergey Levine, Hugo Larochelle, Yoshua Bengio
(Submitted on 2 Apr 2018)
In many environments only a tiny subset of all states yield high reward. In these cases, few of the interactions with the environment provide a relevant learning signal. Hence, we may want to preferentially train on those high-reward states and the probable trajectories leading to them. To this end, we advocate for the use of a backtracking model that predicts the preceding states that terminate at a given high-reward state. We can train a model which, starting from a high value state (or one that is estimated to have high value), predicts and sample for which the (state, action)-tuples may have led to that high value state. These traces of (state, action) pairs, which we refer to as Recall Traces, sampled from this backtracking model starting from a high value state, are informative as they terminate in good states, and hence we can use these traces to improve a policy. We provide a variational interpretation for this idea and a practical algorithm in which the backtracking model samples from an approximate posterior distribution over trajectories which lead to large rewards. Our method improves the sample efficiency of both on- and off-policy RL algorithms across several environments and tasks.



refed by infobot
