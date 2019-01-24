


irl  from  demo


Visual novelty, curiosity, and intrinsic reward in machine learning and the brain
Andrew Jaegle, Vahid Mehrpour, Nicole Rust
(Submitted on 8 Jan 2019)
A strong preference for novelty emerges in infancy and is prevalent across the animal kingdom. When incorporated into reinforcement-based machine learning algorithms, visual novelty can act as an intrinsic reward signal that vastly increases the efficiency of exploration and expedites learning, particularly in situations where external rewards are difficult to obtain. Here we review parallels between recent developments in novelty-driven machine learning algorithms and our understanding of how visual novelty is computed and signaled in the primate brain. We propose that in the visual system, novelty representations are not configured with the principal goal of detecting novel objects, but rather with the broader goal of flexibly generalizing novelty information across different states in the service of driving novelty-based learning.





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



VIREL:  还有 serge leven 几篇相关；
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




Amplifying the Imitation Effect for Reinforcement Learning of
UCAV’s Mission Execution
Gyeong Taek Lee 1 Chang Ouk Kim 1
Abstract
This paper proposes a new reinforcement learning
(RL) algorithm that enhances exploration by amplifying the imitation effect (AIE). This algorithm
consists of self-imitation learning and random network distillation algorithms. We argue that these
two algorithms complement each other and that
combining these two algorithms can amplify the
imitation effect for exploration. In addition, by
adding an intrinsic penalty reward to the state that
the RL agent frequently visits and using replay
memory for learning the feature state when using an exploration bonus, the proposed approach
leads to deep exploration and deviates from the
current converged policy. We verified the exploration performance of the algorithm through experiments in a two-dimensional grid environment.
In addition, we applied the algorithm to a simulated environment of unmanned combat aerial
vehicle (UCAV) mission execution, and the empirical results show that AIE is very effective for
finding the UCAV’s shortest flight path to avoid
an enemy’s missiles

4. AIE
4.1. Combining SIL and RND
In this section, we explain why combining RND and SIL can amplify the imitation effect and lead to deep exploration. The SIL updates only when the past R is greater than the current Vθ and imitates past decisions. Intuitively, if we combine SIL and RND, we find that the (R − Vθ) value is larger than the SIL because of the exploration bonus. In the process of optimizing the actor-critic network to maximize Rt = Σ∞k=tγk−t(it + et)k, where it is intrin- sic reward and et is extrinsic reward, the increase in it by the predictor network causes R to increase. That is, the learning progresses by weighting the good decisions of the
past. This type of learning thoroughly reviews the learn- ing history.If the policy starts to converge as the learning progresses, the it will be lower for the state that was fre- quently visited. One might think that learning can be slower as (Rt − Vθ) > (Rt+k − Vθ), where k > 0 for the same state and it decreases. However, the SIL exploits past good decisions and leads to deep exploration. By adding an ex- ploration bonus, the agent can further explore novel states. Consequently, the exploration bonus is likely to continue to occur. In addition, using the prioritized experience replay (Schaul et al., 2015), the sampling probability is determined by the (R − Vθ); thus, there is a high probability that the SIL will exploit the previous transition even if it decreases. In other words, the two algorithms are complementary to each other, and the SIL is immune to the phenomenon in which the prediction error (it) no longer occurs.

4.2. Intrinsic Penalty Reward
Adding an exploration bonus to a novel state that the agent visits is clearly an effective exploration method. However, when the policy and predictor networks converge, there is no longer an exploration bonus for the novel state. In other words, the exploration bonus method provides a reward when the agent itself performs an unexpected action, not when the agent is induced to take the unexpected action. Therefore, an exploration method that entices the agent to take unexpected behavior is necessary. We propose a method to provide an intrinsic penalty reward for an action when it frequently visits the same state rather than reward- ing it when the agent makes an unexpected action. The intrinsic penalty reward allows the agent to escape from the converged local policy and helps to experience diverse policies. Specifically, we provide a penalty by transform- ing the current intrinsic reward into λlog(it), where λ is a penalty weight parameter, if the current intrinsic reward is less than the quantile α of the past N intrinsic rewards. This reward mechanism prevents the agent from staying in the same policy. In addition, adding a penalty to the intrinsic reward indirectly amplifies the imitation effect. Since the (Rt − Vθ ) becomes smaller due to the penalty, the probabil- ity of sampling in replay memory is relatively smaller than that of non-penalty transition. SIL updates are more likely to exploit non-penalty transitions. Even if (Rt − Vθ ) < 0 due to a penalty, it does not affect SIL because it is not updated because of the objective of SIL in equation 4. In other words, the intrinsic penalty reward allows the policy network to deviate from the constantly visited state of the agent and indirectly amplifies the imitation effect for the SIL.
4.3. Catastrophic Forgetting in RND
The predictor network in RND mainly learns about the state that the agent recently visited, which is similar to the
catastrophic forgetting of continual task learning that forgets learned knowledge of previous tasks. If the prediction error increases for a state that the agent has visited before, the agent may recognize the previous state as a novel state. Consequently, an agent cannot effectively explore. The method to mitigate this phenomenon is simple but effective. We store the output of the target network and state feature as the memory of the predictor network, just like using a replay memory to reduce the correlation between samples(Mnih et al., 2013), and train the predictor network in a batch mode. Using the predictor memory reduces the prediction error of states that the agent previously visited, which is why the agent is more likely to explore novel states. Even if the agent returns to a past policy, the prediction error of the state visited by the policy is low, intrinsic penalty is given to the state, and the probability of escaping from the state is high.

end




SIL

2
prior
apex--
relatation:
Distributed Importance Sampling A complementary family of techniques for speeding up train- ing is based on variance reduction by means of importance sampling (cf. Hastings, 1970). This has been shown to be useful in the context of neural networks (Hinton, 2007). Sampling non-uniformly from a dataset and weighting updates according to the sampling probability in order to counteract the bias thereby introduced can increase the speed of convergence by reducing the variance of the gradients. One way of doing this is to select samples with probability proportional to the L2 norm of the corresponding gradients. In supervised learning, this approach has been successfully extended to the distributed setting (Alain et al., 2015). An alternative is to rank samples according to their latest known loss value and make the sampling probability a function of the rank rather than of the loss itself (Loshchilov & Hutter, 2015).
Prioritized Experience Replay Experience replay (Lin, 1992) has long been used in reinforce- ment learning to improve data efficiency. It is particularly useful when training neural network function approximators with stochastic gradient descent algorithms, as in Neural Fitted Q-Iteration (Riedmiller, 2005) and Deep Q-Learning (Mnih et al., 2015). Experience replay may also help to prevent overfitting by allowing the agent to learn from data generated by previous versions of the policy. Prioritized experience replay (Schaul et al., 2016) extends classic prioritized sweeping ideas (Moore & Atkeson, 1993) to work with deep neural network function approximators. The approach is strongly related to the importance sampling techniques discussed in the previous section, but us- ing a more general class of biased sampling procedures that focus learning on the most ‘surprising’ experiences. Biased sampling can be particularly helpful in reinforcement learning, since the reward signal may be sparse and the data distribution depends on the agent’s policy. As a result, prioritized experience replay is used in many agents, such as Prioritized Dueling DQN (Wang et al., 2016), UNREAL (Jaderberg et al., 2017), DQfD (Hester et al., 2017), and Rainbow (Hessel et al., 2017). In an ablation study conducted to investigate the relative importance of several algorithmic ingredi- ents (Hessel et al., 2017), prioritization was found to be the most important ingredient contributing to the agent’s performance.

SIL 2
Learning from imperfect demonstrations A few stud- ies have attempted to learn from imperfect demonstrations, such as DQfD (Hester et al., 2018), Q-filter (Nair et al., 2017), and normalized actor-critic (Xu et al., 2018). Our self-imitation learning has a similar flavor in that the agent learns from imperfect demonstrations. However, we treat the agent’s own experiences as demonstrations without us- ing expert demonstrations. Although a similar idea has been discussed for program synthesis (Liang et al., 2016; Abo- lafia et al., 2018), this prior work used classification loss without justification. On the other hand, we propose a new objective, provide a theoretical justification, and systemati- cally investigate how it drives exploration in RL.

与4.1. Entropy-Regularized Reinforcement Learning 
The goal of entropy-regularized RL is to learn a stochastic policy which maximizes the entropy of the policy as well as the γ-discounted sum of rewards (Haarnoja et al., 2017; Ziebart et al., 2008): 
SIL. 这个和sac 一样？
已读
infobot -goal --recall trace---SIL 高reward；也可以是好奇心的rnd的高reward?。rnd-图的-meaning events ；rnd 高reward是否可以作为infobot的goal？



----------------------------

EPISODIC CURIOSITY THROUGH REACHABILITY

step

计算量太大吧？ every step！
用divertity is all you need  改进是不是会很好！！。