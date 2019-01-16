





Imagination-Augmented Agents







https://sites.google.com/view/unsupervised-via-meta codes;





LEARNING A PRIOR OVER INTENT
VIA META-INVERSE REINFORCEMENT LEARNING





Automata Guided Reinforcement Learning With Demonstrations






Few-Shot Goal Inference for Visuomotor Learning and Planning




UNSUPERVISED CONTROL THROUGH NON-PARAMETRIC DISCRIMINATIVE REWARDS





OFF-POLICY DEEP REINFORCEMENT LEARNING WITHOUT EXPLORATION





maml--On First-Order Meta-Learning Algorithms





-------------------------
https://bair.berkeley.edu/blog/2018/11/30/visual-rl/   link to polo paper;

Model-free reinforcement learning systems typically only learn from the supervision provided from the reward function, whereas model-based RL agents utilize the rich information available in the pixels they observe. Now, how do we actually use these predictions?

Planning to Perform Human-Specified Tasks
If we have a predictive model of the world, then we can use it to plan to achieve goals. That is, if we understand the consequences of our actions, then we can use that understanding to choose actions that lead to the desired outcome. We use a sampling-based procedure to plan. In particular, we sample many different candidate action sequences, then select the top plans—the actions that are most likely to lead to the desired outcome—and refine our plan iteratively, by resampling from a distribution of actions fitted to the top candidate action sequences. Once we come up with a plan that we like, we then execute the first step of our plan in the real world, observe the next image, and then replan in case something unexpected happened.
goal  her

Experiments   action-conditioned video prediction model
We train a single action-conditioned video prediction model on all of this data, including two camera viewpoints, and use the iterative planning procedure described previously to plan and execute on user-specified tasks.
Since we set out to achieve generality, we evaluate the same predictive model on a wide range of tasks involving objects that the robot has never seen before and goals the robot has not encountered previously.




paper： Visual Foresight: Model-Based Deep Reinforcement Learning for Vision-Based Robotic Control
abs：
a predictive model learns to directly predict the future from raw sensory readings, such as camera images. At test time, we explore three distinct goal specification methods: designated pixels, where a user specifies desired object manipulation tasks by selecting particular pixels in an image and corresponding goal positions, goal images, where the desired goal state is specified with an image, and image classifiers, which define spaces of goal states
We demonstrate that visual MPC can generalize to never-before-seen objects—both rigid and deformable—and solve a range of user-defined object manipulation tasks using the same model.
1
learning be- haviors that generalize to new tasks and objects remains an open problem. The key to generalization is diversity.
Learning skills in diverse environments, such as the real world, presents a number of significant challenges: external reward feedback is extremely sparse or non-existent, and the agent has only indirect access to the state of the world through its senses, which, in the case of a robot, might correspond to cameras and joint encoders.
Prediction is often considered a fundamental component of intelligence [5]. Through prediction, it is possible to learn useful concepts about the world even from a raw stream of sensory observations, such as images from a camera. If we predict raw sensory observations directly, we do not need to assume availability of low-dimensional state information or an extrinsic reward signal.












---------------------
RND：

Figure 1: RND exploration bonus over the course of the first episode where the agent picks up the torch (19-21). To do so the agent passes 17 rooms and collects gems, keys, a sword, an amulet, and opens two doors. Many of the spikes in the exploration bonus correspond to meaningful events: losing a life (2,8,10,21), narrowly escaping an enemy (3,5,6,11,12,13,14,15), passing a difficult obstacle (7,9,18), or picking up an object (20,21). The large spike at the end corresponds to a novel experience of interacting with the torch, while the smaller spikes correspond to relatively rare events that the agent has nevertheless experienced multiple times. See here for videos.

blog
unfamiliar states it’s hard to guess the output, and hence the reward is high

What Do Curious Agents Do?
We tested our agent across 50+ different environments and observed a range of competence levels from seemingly random actions to deliberately interacting with the environment. To our surprise, in some environments the agent achieved the game’s objective even though the game’s objective was not communicated to it through an extrinsic reward.


Bowling - The agent learned to play the game better than agents trained to maximize the (clipped) extrinsic reward directly. We think this is because the agent gets attracted to the difficult-to-predict flashing of the scoreboard occurring after the strikes.

Mario - The intrinsic reward is particularly well-aligned with the game’s objective of advancing through the levels. The agent is rewarded for finding new areas because the details of a newly found area are impossible to predict. As a result the agent discovers 11 levels, finds secret rooms, and even defeats bosses.


EXPLORATION BY RANDOM NETWORK DISTILLATION
Yuri Burda∗ Harrison Edwards∗ Amos Storkey Oleg Klimov
OpenAI OpenAI
Univ. of Edinburgh OpenAI
ABSTRACT
We introduce an exploration bonus for deep reinforcement learning methods that is easy to implement and adds minimal overhead to the computation performed. The bonus is the error of a neural network predicting features of the observations given by a fixed randomly initialized neural network. We also introduce a method to flexibly combine intrinsic and extrinsic rewards. We find that the random network distillation (RND) bonus combined with this increased flexibility enables significant progress on several hard exploration Atari games. In particular we establish state of the art performance on Montezuma’s Revenge, a game famously difficult for deep reinforcement learning methods. To the best of our knowledge, this is the first method that achieves better than average human performance on this game without using demonstrations or having access to the underlying state of the game, and occasionally completes the first level.
1 INTRODUCTION


2.2.1 SOURCES OF PREDICTION ERRORS
In general, prediction errors can be attributed to a number of factors:
1. Amount of training data. Prediction error is high where few similar examples were seen by the predictor (epistemic uncertainty).
2. Stochasticity. Prediction error is high because the target function is stochastic (aleatoric un- certainty). Stochastic transitions are a source of such error for forward dynamics prediction.
3. Model misspecification. Prediction error is high because necessary information is missing, or the model class is too limited to fit the complexity of the target function.
4. Learning dynamics. Prediction error is high because the optimization process fails to find a predictor in the model class that best approximates the target function.



2.2.2 RELATION TO UNCERTAINTY QUANTIFICATION
If we specialize the regression targets yi to be zero, then the optimization problem arg minθ E(xi,yi)∼D∥fθ(xi) + fθ∗ (xi)∥2 is equivalent to distilling a randomly drawn function from the prior. Seen from this perspective, each coordinate of the output of the predictor and target net- works would correspond to a member of an ensemble (with parameter sharing amongst the ensemble), and the MSE would be an estimate of the predictive variance of the ensemble (assuming the ensemble is unbiased). In other words the distillation error could be seen as a quantification of uncertainty in predicting the constant zero function.

2.3 COMBINING INTRINSIC AND EXTRINSIC RETURNS
In preliminary experiments that used only intrinsic rewards, treating the problem as non-episodic resulted in better exploration. In that setting the return is not truncated at “game over”. We argue that this is a natural way to do exploration in simulated environments, since the agent’s intrinsic return should be related to all the novel states that it could find in the future, regardless of whether they all occur in one episode or are spread over several. It is also argued in (Burda et al., 2018) that using episodic intrinsic rewards can leak information about the task to the agent.


内部reward  episode不截断


3
 Most experiments are run for 30K rollouts of length 128 per environment with 128 parallel environments, for a total of 1.97 billion frames of experience.
 Contrary to expectations in Figure 4 recurrent policies performed worse than non-recurrent counterparts with γE = 0.99. However in Figure 6 the RNN policy with
γE = 0.999 outperformed the CNN counterpart at each scale1. Comparison of Figures 7 and 9 shows that across multiple games the RNN policy outperforms the CNN more frequently than the other way around.

4
Other methods of exploration include adversarial self-play (Sukhbaatar et al., 2018), maximizing empowerment (Gregor et al., 2017), parameter noise (Plappert et al., 2017; Fortunato et al., 2017), identifying diverse policies (Eysenbach et al., 2018; Achiam et al., 2018), and using ensembles of value functions (Osband et al., 2018; 2016; Chen et al., 2017).
Random features. Features of randomly initialized neural networks have been extensively studied in the context of supervised learning (Rahimi & Recht, 2008; Saxe et al., 2011; Jarrett et al., 2009; Yang et al., 2015). More recently they have been used in the context of exploration (Osband et al., 2018; Burda et al., 2018). The work Osband et al. (2018) provides motivation for random network distillation as discussed in Section 2.2.

5
However global exploration that involves coordinated decisions over long time horizons is beyond the reach of our method.


rnd. 5 DISCUSSION 
 To solve the first level of Montezuma’s Revenge, the agent must enter a room locked behind two doors. There are four keys and six doors spread throughout the level. Any of the four keys can open any of the six doors, but are consumed in the process. To open the final two doors the agent must therefore forego opening two of the doors that are easier to find and that would immediately reward it for opening them. 
To incentivize this behavior the agent should receive enough intrinsic reward for saving the keys to balance the loss of extrinsic reward from using them early on. From our analysis of the RND agent’s behavior, it does not get a large enough incentive to try this strategy, and only stumbles upon it rarely. 
Solving this and similar problems that require high level exploration is an important direction for future work.

We find that the RND exploration bonus is sufficient to deal with local exploration, i.e. exploring the consequences of short-term decisions, like whether to interact with a particular object, or avoid it. However global exploration that involves coordinated decisions over long time horizons is beyond the reach of our method.






---------------------------------------------------------------

Unsupervised Meta-Learning for Reinforcement Learning

Abstract
Meta-learning is a powerful tool that builds on multi-task learning to learn how to quickly adapt a model to new tasks. In the context of reinforcement learning, meta-learning algorithms can acquire reinforcement learning procedures to solve new problems more efficiently by meta-learning prior tasks. The performance of meta-learning algorithms critically depends on the tasks available for meta-training: in the same way that supervised learning algorithms generalize best to test points drawn from the same distribution as the training points, meta-learning methods generalize best to tasks from the same distribution as the meta-training tasks. In effect, meta-reinforcement learning offloads the design burden from algorithm design to task design. If we can automate the process of task design as well, we can devise a meta-learning algorithm that is truly automated. In this work, we take a step in this direction, proposing a family of unsupervised meta-learning algorithms for reinforcement learning. We describe a general recipe for unsuper- vised meta-reinforcement learning, and describe an effective instantiation of this approach based on a recently proposed unsupervised exploration technique and model-agnostic meta-learning. We also discuss practical and conceptual consid- erations for developing unsupervised meta-learning methods. Our experimental results demonstrate that unsupervised meta-reinforcement learning effectively ac- quires accelerated reinforcement learning procedures without the need for manual task design, significantly exceeds the performance of learning from scratch, and even matches performance of meta-learning methods that use hand-specified task distributions.

1 Introduction
Reusing past experience for faster learning of new tasks is a key challenge for machine learning. Meta-learning methods propose to achieve this by using past experience to explicitly optimize for rapid adaptation [23, 32, 30, 9, 6, 14, 37, 1]. In the context of reinforcement learning, meta- reinforcement learning algorithms can learn to solve new reinforcement learning tasks more quickly through experience on past tasks [6, 14]. Typical meta-reinforcement learning algorithms assume the ability to sample from a pre-specified task distribution, and these algorithms learn to solve new tasks drawn from this distribution very quickly. However, specifying a task distribution is tedious and requires a significant amount of supervision [10, 6] that may be difficult to provide for large real-world problem settings. The performance of meta-learning algorithms critically depends on the meta-training task distribution, and meta-learning algorithms generalize best to new tasks which
are drawn from the same distribution as the meta-training tasks [8]. In effect, meta-reinforcement learning offloads some of the design burden from algorithm design to designing a sufficiently broad and relevant distribution of meta-training tasks. While this greatly helps in acquiring representations for fast adaptation to the specified task distribution, a natural question is whether we can do away with the need for manually designing a large family of tasks, and develop meta-reinforcement learning algorithms that learn only from unsupervised environment interaction. In this paper, we take an initial step toward the formalization and design of such methods.

Our goal is to automate the meta-training process by removing the need for hand-designed meta- training tasks. To that end, we introduce unsupervised meta-reinforcement learning: meta-learning from a task distribution that is acquired automatically, rather than requiring manual design of the meta-training tasks. Developing effective unsupervised meta-reinforcement learning algorithms is challenging, since it requires solving two difficult problems together: meta-reinforcement learning with broad task distributions, and unsupervised exploration for proposing a wide variety of tasks for meta-learning. Since the assumptions of our method differ fundamentally from prior meta- reinforcement learning methods (we do not assume access to hand-specified meta-training tasks), the best points of comparison for our approach are learning the meta-test tasks entirely from scratch with conventional reinforcement learning algorithms. Our method can also be thought of as a data-driven initialization procedure for deep neural network policies, in a similar vein to data-driven initialization procedures explored in supervised learning [20].

The primary contributions of our work are to propose a framework for unsupervised meta- reinforcement learning, sketch out a family of unsupervised meta-reinforcement learning algorithms, and describe a possible instantiation of a practical algorithm from this family that builds on a recently proposed procedure for unsupervised exploration [7] and model-agnostic meta-learning (MAML) [9]. We discuss the design considerations and conceptual issues surrounding unsupervised meta-reinforcement learning, and provide an empirical evaluation that studies the performance of two variants of our approach on simulated continuous control tasks. Our experimental evaluation shows that, for a variety of tasks, unsupervised meta-reinforcement learning can effectively acquire reinforcement learning procedures that perform significantly better than standard reinforcement learning in terms of sample complexity and asympototic performance, and even rival the performance of conventional meta-learning algorithms that are provided with hand-designed task distributions.

2 Related Work



3 Unsupervised Meta-Reinforcement Learning
The goal of unsupervised meta-reinforcement learning is to take an environment and produce a learning algorithm specifically tailored to this environment that can quickly learn to maximize reward on any task reward in this environment. This learning algorithm should be meta-learned without requiring any human supervision. We can formally define unsupervised meta-reinforcement learning in the context of a controlled Markov process (CMP) – a Markov decision process without a reward function, C = (S,A,T,γ,ρ), with state space S, action space A, transition dynamics T, discount factor γ and initial state distribution ρ. Our goal is to learn a learning algorithm f on this CMP, which can subsequently learn new tasks efficiently in this CMP for a new reward function Ri, which produces a Markov decision processes Mi = (S, A, T, γ, ρ, Ri). We can, at a high level, denote f as a mapping from tasks to policies, f : T → Π, where T is the space of RL tasks defined by the given CMP and Ri, and Π is a space of parameterized policies, such that π ∈ Π is a probability distribution over actions conditioned on states, π(a|s). Crucially, f must be learned without access to any reward functions Ri, using only unsupervised interaction with the CMP. The reward is only provided at meta-test time.
3.1 A General Recipe
Our framework unsupervised meta-reinforcement learning consists of two components. The first component is a task identification procedure, which interacts with a controlled Markov process, without access to any reward function, to construct a distribution over tasks. Formally, we will define the task distribution as a mapping from a latent variable z ∼ p(z) to a reward function rz (s, a) : S × A → R. That is, for each value of the random variable z, we have a different reward function rz(s,a). The prior p(z) may be specified by hand. For example, we might choose a uniform categorical distribution or a spherical unit Gaussian. A discrete latent variable z corresponds to a discrete set of tasks, while a continuous representation could allow for an infinite task space. Under this formulation, learning a task distribution amounts to optimizing a parametric form for the reward function rz (s, a) that maps each z ∼ p(z) to a different reward function.
The second component of unsupervised meta-learning is meta-learning, which takes the family of reward functions induced by p(z) and rz (s, a), and meta-learns a reinforcement learning algorithm f that can quickly adapt to any task from the task distribution defined by p(z) and rz(s,a). The meta-learned algorithm f can then learn new tasks quickly at meta-test time, when a user-specified reward function is actually provided. This generic design for an unsupervised meta-reinforcement learning algorithm is summarized in Figure 1.
The nature of the task distribution defined by p(z) and rz (s, a) will affect the effectiveness of f on new tasks: tasks that are close to this distribution will be easiest to learn, while tasks that are far from this distribution will be difficult to learn. However, the nature of the meta-learning algorithm itself will also curcially affect the effectiveness of f. As we will discuss in the following sections, some meta-reinforcement learning algorithms can generalize effectively to new tasks, while some cannot. A more general version of this algorithm might also use f to inform the acquisition of tasks, allowing for an alternating optimization procedure the iterates between learning rz (s, a) and updating f , for example by designing tasks that are difficult for the current algorithm f to handle. However, in this paper we will consider the stagewise approach, which acquires a task distribution once and meta-trains on it, leaving the iterative variant for future work.

3.2 Unsupervised Task Acquisition

Task acquisition via diversity-driven exploration. We can acquire more varied tasks if we allow ourselves some amount of unsupervised environment interaction. Specifically, we consider a recently proposed method for unsupervised skill diversity method - Diversity is All You Need (DIAYN) [7] for task acquisition. DIAYN attempts to acquire a set of behaviors that are distinguishable from one another, in the sense that they visit distinct states, while maximizing conditional policy entropy to encourage diversity [15]. Skills with high entropy that remain discriminable must explore a part of the state space far away from other skills. Formally, DIAYN learns a latent conditioned policy πθ(a|s,z), with z ∼ p(z), where different values of z induce different skills. The training process promotes discriminable skills by maximizing the mutual information between skills and states (MI(s,z)), while also maximizing the policy entropy H(a|s, z):
F(θ) 􏰜 MI(s, z) + H[a | s] − MI(a, z | s) = H[a | s, z] + H[z] − H[z | s] (1)
A learned discriminator Dφ(z|s) maximizes a variational lower bound on Equation 1 (see [7] for proof). We train the discriminator to predict the latent variable z from the observed state, and optimize the latent conditioned policy to maximize the log-likelihood of the discriminator correctly classifying states which are visited under different skills, while maximizing policy entropy. Under this formulation, we can think of the discriminator as rewarding the policy for producing discriminable skills, and the policy visitations as informing the training of the discriminator.
After learning the policy and discriminator, we can sample tasks by generating samples z ∼ p(z) and using the corresponding task reward rz(s) = log(Dφ(z|s)). Compared to random discriminators, the tasks acquired by DIAYN are more likely to involve visiting diverse parts of the state space, potentially providing both a greater challenge to the corresponding policy, and achieving better coverage of the CMP’s state space. This method is still fully unsupervised, as it requires no handcrafting of distance metrics or subgoals, and does not require training generative model to generate goals [16].

3.5 Which Unsupervised and Meta-Learning Procedures Should Work Well?
Having introduced example instantiations of unsupervised meta-reinforcement learning, we discuss more generally what criteria each of the two procedures should satisfy - task acquisition and meta- reinforcement learning. What makes a good task acquisition procedure for unsupervised meta- reinforcement learning? Several criteria are desirable. First, we want the tasks that are learned to resemble the types of tasks that might be present at meta-test time. DIAYN receives no supervision in this regard, basing its task acquisition entirely on the dynamics of the CMP. A more guided approach could incorporate a limited number of human-specified tasks, or manually-provided guidance about valuable state space regions. Without any prior knowledge, we expect the ideal task distribution to induce a wide distribution over trajectories. As many distinct reward functions can have the same optimal policy, a random discriminator may actually result in a narrow distribution of optimal trajectories. In contrast, ... Unsupervised task acquisition procedures like DIAYN, which mediate the task acquisition process via interactions with the environment (which imposes dynamically consistent

We might then ask what kind of knowledge could possibly be “baked” into f during meta-training. There are two sources of knowledge that can be acquired. First, a meta-learning procedure like MAML modifies the initial parameters θ of a policy πθ(a|s). When πθ(a|s) is represented by an expressive function class like a neural network, the initial setting of these parameters strongly affects how quickly the policy can be trained by gradient descent. Indeed, this is the rationale behind research into more effective general-purpose initialization methods [19, 40]. Meta-training a policy essentially learns an effective weight initialization such that a few gradient steps can effectively modify the policy in functionally relevant ways.
The policy found by unsupervised meta-training also acquires an awareness of the dynamics of the given controlled Markov process (CMP). Intuitively, an ideal policy should adapt in the space of trajectories τ, rather than the space of actions a or parameters θ; an RL update should modify the policy’s trajectory distribution, which determines the reward function. Natural gradient algorithms impose equal-sized steps in the space of action distributions [31], but this is not necessarily the ideal adaptation manifold, since systematic changes in output actions do not necessarily translate into system changes in trajectory or state distributions. In effect, meta-learning prepares the policy to modify its behavior in ways that cogently affect the states that are visited, which requires a parameter setting informed by the dynamics of the CMP. This can be provided effectively through unsupervised meta-reinforcement learning.


















----------------------------------------------------

paper： MODEL-ENSEMBLE TRUST-REGION POLICY OPTIMIZATION


5 MODEL-ENSEMBLE TRUST-REGION POLICY OPTIMIZATION
Using the vanilla approach described in Section 4, we find that the learned policy often exploits regions where scarce training data is available for the dynamics model. Since we are improving
？？默认有探索的功能机制？？

不用bp？？

------------------------------------
paper： Model-Based Reinforcement Learning via Meta-Policy Optimization

In this paper we show that 1) model-based policy optimization can learn policies that match the asymptotic performance of model-free methods while being substantially more sample efficient, 2) MB-MPO consistently outperforms previous model-based methods on challenging control tasks, 3) learning is still possible when the models are strongly biased. The low sample complexity of our method makes it applicable to real-world robotics. For instance, we are able learn an optimal policy in high-dimensional and complex quadrupedal locomotion within two hours of real-world data. Note that the amount of data required to learn such policy using model-free methods is 10× - 100× higher, and, to the best knowledge of the authors, no prior model-based method has been able to attain the model-free performance in such tasks.

sac 速度也够快

Soft actor-critic solves all of these tasks quickly: the Minitaur locomotion and the block-stacking tasks both take 2 hours, and the valve-turning task from image observations takes 20 hours. We also learned a policy for the valve-turning task without images by providing the actual valve position as an observation to the policy. Soft actor-critic can learn this easier version of the valve task in 3 hours. For comparison, prior work has used PPO to learn the same task without images in 7.4 hours.



2


 We incorporate the idea of reducing model-bias by learning an ensemble of models. However, we show that these techniques do not suffice in challenging domains, and demonstrate the necessity of meta-learning for improving asymptotic performance.
Past work has also tried to overcome model inaccuracies through the policy optimization pro- cess. Model Predictive Control (MPC) compensates for model imperfections by re-planning at each step [30], but it suffers from limited credit assignment and high computational cost. Robust policy optimization [7, 8, 9] looks for a policy that performs well across models; as a result policies tend to be over-conservative. In contrast, we show that MB-MPO learns a robust policy in the regions where



three main approaches. First, differ- entiable trajectory optimization methods propagate the gradients of the policy or value function through the learned dynamics model [31, 32] . However, the models are not explicitly trained to approximate first order derivatives, and, when backpropagating, they suffer from exploding and vanishing gradients [10]. Second, model-assisted MF approaches use the dynamics models to aug- ment the real environment data by imagining policy roll-outs [33, 29, 34, 22]. These methods still rely to a large degree on real-world data, which makes them impractical for real-world applications. Thanks to meta-learning, our approach could, if needed, adapt fast to the real-world with fewer samples. Third, recent work fully decouples the MF module from the real environment by entirely using samples from the learned models [35, 10]. These methods, even though considering the model uncertainty, still rely on precise estimates of the dynamics to learn the policy. In contrast, we meta-



Current meta-learning algorithms can be classified in three categories. One approach in- volves training a recurrent or memory-augmented network that ingests a training dataset and outputs the parameters of a learner model [36, 37]. Another set of methods feeds the dataset followed by the test data into a recurrent model that outputs the predictions for the test inputs [12, 38]. The last cat- egory embeds the structure of optimization problems into the meta-learning algorithm [11, 39, 40]. These algorithms have been extended to the context of RL [12, 13, 15, 11]. Our work builds upon MAML [11]. However, while in previous meta-learning methods each task is typically defined by a different reward function, each of our tasks is defined by the dynamics of different learned models.

 previous meta-learning methods each task is typically defined by a different reward function, each of our tasks is defined by the dynamics of different learned models.

3 Background


4
MB-MPO), attains such goal by framing model-based RL as meta-learning a policy on a distribution of dynamic models, advocating to max- imize the policy adaptation, instead of robustness, when models disagree
This not only removes the arduous task of optimizing for a single policy that performs well across differing dynamic models, but also results in better exploration properties and higher diversity of the collected samples, which leads to improved dynamic estimates.









-------------------------------------------
maml

1
The process of training a model’s parameters such that a few gradient steps, or even a single gradient step, can pro- duce good results on a new task can be viewed from a fea- ture learning standpoint as building an internal representa- tion that is broadly suitable for many tasks. If the internal representation is suitable to many tasks, simply fine-tuning the parameters slightly (e.g. by primarily modifying the top layer weights in a feedforward model) can produce good results. In effect, our procedure optimizes for models that are easy and fast to fine-tune, allowing the adaptation to happen in the right space for fast learning. From a dynami- cal systems standpoint, our learning process can be viewed as maximizing the sensitivity of the loss functions of new tasks with respect to the parameters: when the sensitivity is high, small local changes to the parameters can lead to


2.2
we propose a method that can learn the parameters of any standard model via meta-learning in such a way as to prepare that model for fast adaptation. The intuition behind this approach is that some internal representations are more transferrable than others. For example, a neural network might learn internal features that are broadly applicable to all tasks in p(T ), rather than a single individual task. How can we en- courage the emergence of such general-purpose representa- tions? We take an explicit approach to this problem: since the model will be fine-tuned using a gradient-based learn- ing rule on a new task, we will aim to learn a model in such a way that this gradient-based learning rule can make rapid progress on new tasks drawn from p(T ), without overfit- ting. In effect, we will aim to find model parameters that are sensitive to changes in the task, such that small changes in the parameters will produce large improvements on the loss function of any task drawn from p(T ), when altered in the direction of the gradient of that loss (see Figure 1). We



实验领域及效果









-------------------------------
A SIMPLE NEURAL ATTENTIVE META-LEARNER

We propose a class of simple and generic meta-learner architectures that use a novel combination of temporal convolutions and soft attention; the former to aggregate information from past experience and the latter to pinpoint specific pieces of information.
 

1
Meta-learning can be formalized as a sequence-to-sequence problem; in existing approaches that adopt this view, the bottleneck is in the meta-learner’s ability to internalize and refer to past experience. Thus, we propose a class of model architectures that addresses this shortcoming: we combine temporal convolutions, which enable the meta-learner to aggregate contextual information from past experience, with causal attention, which allow it to pinpoint specific pieces of information within that context.

3
construct SNAIL by combining the two: we use temporal convolutions to produce the context over which we use a causal attention operation. By interleaving TC layers with causal attention layers, SNAIL can have high-bandwidth access over its past experience without constraints on the amount of experience it can effectively use. By using attention at multiple stages within a model that is trained end-to-end, SNAIL can learn what pieces of information to pick out from the experience it gathers, as well as a feature representation that is amenable to doing so easily. As an additional benefit, SNAIL architectures are easier to train than traditional RNNs such as LSTM or GRUs 
 
3.1 MODULAR BUILDING BLOCKS
 
 
实验领域及效果

5 EXPERIMENTS
Our experiments were designed to investigate the following questions:
• How does SNAIL’s generality affect its performance on a range of meta-learning tasks?
• How does its performance compare to existing approaches that are specialized to a particular
task domain, or have elements of a high-level strategy already built-in?
• HowdoesSNAILscalewithhigh-dimensionalinputsandlong-termtemporaldependencies?
5.1 FEW-SHOT IMAGE CLASSIFICATION






----------------------------------
Learning Unsupervised Learning Rules

Abstract
A major goal of unsupervised learning is to discover data representations that are useful for subsequent tasks, without access to supervised labels during training. Typically, this goal is approached by minimizing a surrogate objective, such as the negative log likelihood of a generative model, with the hope that representations useful for subsequent tasks will arise incidentally. In this work, we propose instead to directly target a later desired task by meta-learning an unsupervised learning rule, which leads to representations useful for that task. Here, our desired task (meta-objective) is the performance of the representation on semi-supervised classification, and we meta-learn an algorithm – an unsupervised weight update rule – that produces representations that perform well under this meta-objective. Additionally, we constrain our unsupervised update rule to a be a biologically- motivated, neuron-local function, which enables it to generalize to novel neural network architectures. We show that the meta-learned update rule produces useful features and sometimes outperforms existing unsupervised learning techniques. We further show that the meta-learned unsupervised update rule generalizes to train networks with different widths, depths, and nonlinearities. It also generalizes to train on data with randomly permuted input dimensions and even generalizes from image datasets to a text task.

1
 One explanation for this failure is that unsupervised representation learning algorithms are typically mismatched to the target task. Ideally, learned representations should linearly expose high level attributes of data (e.g. object identity) and perform well in semi-supervised settings. Many current unsupervised objectives, however, optimize for objectives such as log-likelihood of a generative model or reconstruction error and produce useful representations only as a side effect.
Unsupervised representation learning seems uniquely suited for meta-learning [1, 2]. Unlike most tasks where meta-learning is applied, unsupervised learning does not define an explicit objective, which makes it impossible to phrase the task as a standard optimization problem. It is possible, however, to directly express a meta-objective that captures the quality of representations produced by an unsupervised update rule by evaluating the usefulness of the representation for candidate tasks, e.g. semi-supervised classification. In this work, we propose to meta-learn an unsupervised update rule by meta-training on a meta-objective that directly optimizes the utility of the unsupervised representation.Unlike hand-designed unsupervised learning rules, this meta-objective directly targets the usefulness of a representation generated from unlabeled data for later supervised tasks.
By recasting unsupervised representation learning as meta-learning, we treat the creation of the unsupervised update rule as a transfer learning problem. Instead of learning transferable features, we learn a transferable learning rule which does not require access to labels and generalizes across both data domains and neural network architectures. 
2.1
In contrast to our work, each method imposes a manually defined training algorithm or loss function to optimize whereas we learn the algorithm that creates useful representations as determined by a meta-objective.


To our knowledge, we are the first meta-learning approach to tackle the problem of unsupervised representation learning
we are the first representation meta-learning approach to generalize across input data modalities as well as datasets, the first to generalize across permutation of the input dimensions, and the first to generalize across neural network architectures (e.g. layer width, network depth, activation function).




3.2
We wish for our update rule to generalize across architectures with different widths, depths, or even network topologies. To achieve this, we design our update rule to be neuron-local, so that updates are a function of pre- and post- synaptic neurons in the base model, and are defined for any base model architecture. This has the added benefit that it makes the weight updates more similar to synaptic updates in biological neurons, which depend almost exclusively on the pre- and post-synaptic neurons for each synapse [48].












appendix





--------------
An Approximation of the Error Backpropagation Algorithm in a Predictive Coding Network with Local Hebbian Synaptic Plasticity

To efficiently learn from feedback, cortical networks need to update synaptic weights on multiple levels of cortical hierarchy. An effective and well-known algorithm for computing such changes in synaptic weights is the error backpropagation algorithm. However, in this algorithm, the change in synaptic weights is a complex function of weights and activities of neurons not directly connected with the synapse being modified, whereas the changes in biological synapses are determined only by the activity of presynaptic and postsynaptic neurons. Several models have been proposed that approximate the backpropagation algorithm with local synaptic plasticity, but these models require complex external control over the network or relatively complex plasticity rules. Here we show that a network developed in the predictive coding framework can efficiently perform supervised learning fully autonomously, employing only simple local Hebbian plasticity. Furthermore, for certain parameters, the weight change in the predictive coding model converges to that of the backpropagation algorithm. This suggests that it is possible for cortical networks with simple Hebbian synaptic plasticity to implement efficient learning algorithms in which synapses in areas on multiple levels of hierarchy are modified to minimize the error on the output.







--------------------------------------


Transfer and Exploration via the Information Bottleneck 
Anirudh Goyal, Riashat Islam, DJ Strouse, Zafarali Ahmed, Hugo Larochelle, Matthew Botvinick, Sergey Levine, Yoshua Bengio

abs:
 In effect, the model learns the sensory cues that correlate with potential subgoals
 
ABSTRACT
A central challenge in reinforcement learning is discovering effective policies for tasks where rewards are sparsely distributed. We postulate that in the absence of useful reward signals, an effective exploration strategy should seek out decision states. These states lie at critical junctions in the state space from where the agent can transition to new, potentially unexplored regions. We propose to learn about decision states from prior experience. By training a goal-conditioned model with an information bottleneck on the goal-dependent encoder used by the agent’s policy, we can identify decision states by examining where the model actually leverages the goal state through the bottleneck. We find that this simple mechanism effectively identifies decision states, even in partially observed settings. In effect, the model learns the sensory cues that correlate with potential subgoals. In new environments, this model can then identify novel subgoals for further exploration, guiding the agent through a sequence of potential decision states and through new regions of the state space.



1 INTRODUCTION
Providing agents with useful signals to pursue in lieu of environmental reward becomes crucial in these scenarios. Here, we propose to incentive agents to learn about and exploit multi-goal task structure in order to efficiently explore in new environments. We do so by first training agents to develop useful habits as well as the knowledge of when to break them, and then using that knowledge to efficiently probe new environments for reward.
We focus on multi-goal environments and goal-conditioned policies (Foster and Dayan, 2002; Schaul et al., 2015; Plappert et al., 2018). In these scenarios, a goal G is sampled from p(G) and the beginning of each episode and provided to the agent. The goal G provides the agent with information about the environment’s reward structure for that episode.
We incentive agents to learn task structure by training policies that perform well under a variety of goals, while not overfitting to any individual goal. We achieve this by training agents that, in addition to maximizing reward, minimize the policy dependence on the individual goal, quantified by the conditional mutual information I(A; G | S). This approach is inspired by the information bottleneck approach 
This form of “information dropout” has been shown to promote generalization performance (Achille and Soatto, 2016; Alemi et al., 2017). Here, we show that minimizing goal information promotes generalization in an RL setting as well. We refer to this approach as InfoBot (from information bottleneck).

This approach to learning task structure can also be interpreted as encouraging agents to follow default policy, following this default policy should result in default behaviour which agent should follow in the absence of any additional task information (like the goal location or the relative distance
To see this, note that our regularizer can also be written as I(A;G | S) = Eπθ [DKL[πθ(A | S,G) | π0(A | S)]], where πθ(A | S,G) is the agent’s multi-goal policy, Eπθ denotes an expectation over trajectories generated by πθ, DKL is the Kuhlback-Leibler divergence, and π0(A | S) = 􏰀g p(g) πθ(A | S, g) is a “default” policy with the goal marginalized out. While the agent never actually follows the default policy π0 directly, it can be viewed as what the agent might do in the absence of any knowledge about the goal. Thus, our approach encourages the agent to learn useful behaviours and to follow those behaviours closely, except where diverting from doing so leads to significantly higher reward. Humans too demonstrate an affinity for relying on default behaviour when they can (Kool and Botvinick, 2018), which we take as encouraging support for this line of work (Hassabis et al., 2017).
We refer to states where diversions from default behaviour occur as decision states, based on the intuition that they require the agent not to rely on their default policy (which is goal agnostic) but instead to make a goal-dependent “decision.” Our approach to exploration then involves encouraging the agent to seek out these decision states in new environments. The intuition behind this approach is that such decision states are natural subgoals for efficient exploration because they are boundaries between achieving different goals (van Dijk and Polani, 2011). By encouraging an agent to visit them, the agent is encouraged to 1) follow default trajectories that work across many goals (i.e could be executed in multiple different contexts) and 2) uniformly explore across the many ”branches” of decision-making. We encourage the visitation of decision states by first training an agent to recognize them by training with the information regularizer introduced above. Then, we freeze the agent’s policy, and use DKL[πθ(A | S,G) | π0(A | S)] as an exploration bonus for training a new policy.

To summarize our contributions:
• We regularize RL agents in multi-goal settings with I(A; G | S), an approach inspired by the information bottleneck and the cognitive science of decision making, and show that it promotes generalization across tasks.
• We use policies as trained above to then provide an exploration bonus for training new poli- cies in the form of DKL[πθ(A | S, G) | π0(A | S)], which encourages the agent to seek out decision states. We demonstrate that this approach to exploration performs more effectively than other state-of-the-art methods, including a count-based bonus, VIME (Houthooft et al., 2016), and curiosity (Pathak et al., 2017b).

2 OUR APPROACH

3 RELATED WORK
van Dijk and Polani (2011) were the first to point out the connection between action-goal information and the structure of decision-making. They used information to identify decision states and use them as subgoals in an options framework (Sutton et al., 1999b). We build upon their approach by combining it with deep reinforcement learning to make it more scaleable, and also modify it by using it to provide an agent with an exploration bonus, rather than subgoals for options.
Our decision states are similar in spirit to the notion of ”bottleneck states” used to define subgoals in hierarchical reinforcement learning. A bottleneck state is defined as one that lies on a wide variety of rewarding trajectories (McGovern and Barto, 2001; Stolle and Precup, 2002) or one that otherwise serves to separate regions of state space in a graph-theoretic sense (Menache et al., 2002; S ̧ims ̧ek et al., 2005; Kazemitabar and Beigy, 2009; Machado et al., 2017). The latter definition is purely based on environmental dynamics and does not incorporate reward structure, while both definitions can lead to an unnecessary profileration of subgoals. To see this, consider a T-maze in which the agent starts at the bottom and two possible goals exist at either end of the top of the T. All states in this setup are bottleneck states, and hence the notion is trivial. However, only the junction where the lower and upper line segments of the T meet are a decision state. Thus, we believe the notion of a decision state is a more parsimonious and accurate indication of good subgoals than is the above notions of a bottleneck state. The success of our approach against state-of-the-art exploration methods (Section 4) supports this claim.

We use the terminology of information bottleneck (IB) in this paper because we limit (or bottleneck) the amount of goal information used by our agent’s policy during training. However, the correspon- dence is not exact: while both our method and IB limit information into the model, we maximize rewards while IB maximizes information about a target to be predicted. The latter is thus a supervised learning algorithm. If we instead focused on imitation learning and replaced E[r] with I(A∗; A | S) in Eqn 1, then our problem would correspond exactly to a variational information bottleneck (Alemi et al., 2017) between the goal G and correct action choice A∗ (conditioned on S).
Whye Teh et al. (2017) trained a policy with the same KL divergence term as in Eqn 1 for the purposes of encouraging transfer across tasks. They did not, however, note the connection to variational information minimization and the information bottleneck, nor did they leverage the learned task structure for exploration. Parallel to our work, Strouse et al. (2018) also used Eqn 1 as a training objective, however their purpose was not to show better generalization and transfer, but instead to promote the sharing and hiding of information in a multi-agent setting.
Popular approaches to exploration in RL are typically based on: 1) injecting noise into action selection (e.g. epsilon-greedy, (Osband et al., 2016)), 2) encouraging “curiosity” by encouraging prediction errors of or decreased uncertainty about environmental dynamics (Schmidhuber, 1991; Houthooft et al., 2016; Pathak et al., 2017b), or 3) count-based methods which incentivize seeking out rarely visited states (Strehl and Littman, 2008; Bellemare et al., 2016; Tang et al., 2016; Ostrovski et al., 2017). One limitation shared by all of these methods is that they have no way to leverage experience on previous tasks to improve exploration on new ones; that is, their methods of exploration are not tuned to the family of tasks the agent faces. Our transferrable exploration strategies approach in algorithm 1 however does exactly this. Another notable recent exception is Gupta et al. (2018), which took a meta-learning approach to transferrable exploration strategies.

4 EXPERIMENTAL RESULTS
In this section, we demonstrate the following experimentally:
• The policy-goal information bottleneck leads to much better policy transfer than standard RL training procedures (direct policy transfer).
• Using decision states as an exploration bonus leads to better performance than a variety of standard task-agnostic exploration methods (transferable exploration strategies).

4.1 MINIGRID ENVIRONMENTS
Solving these partially observable, sparsely rewarded tasks by random exploration is difficult because there is a vanishing probability of reaching the goal randomly as the environments become larger. Transferring knowledge from simpler to more complex versions of these tasks thus becomes essential. In the next two sections, we demonstrate that our approach yields 1) policies that directly transfer well from smaller to larger environments, and 2) exploration strategies that outperform other task-agnostic exploration approaches.
4.2 DIRECT POLICY GENERALIZATION ON MINIGRID TASKS
We first demonstrate that training an agent with a goal bottleneck alone already leads to more effective policy transfer
4.3 TRANSFERABLE EXPLORATION STRATEGIES ON MINIGRID TASKS
We now evaluate our approach to exploration (the second half of Algorithm 1). We train agents with a goal bottleneck on one set of environments (MultiRoomN2S6) where they learn the sensory cues that correspond to decision states. Then, we that knowledge to guide exploration one another set of environments 
4.4
TRANSFERABLE EXPLORATION STRATEGIES FOR CONTINUOUS CONTROL
4.5 TRANSFERABLE EXPLORATION STRATEGIES FOR ATARI
4.6 GOAL-BASED NAVIGATION TASKS
 For standard RL algorithms, these tasks are difficult to solve due to the partial observability of the environment, sparse reward (as the agent receives a reward only after reaching the goal), and low probability of reaching the goal via random walks (precisely because these junction states are crucial states where the right action must be taken and several junctions need to be crossed). This environment is more challenging as compared to the Minigrid environment, as this environment also has dead ends as well as more complex branching.
We first demonstrate that training an agent with a goal bottleneck alone leads to more effective policy transfer. We train policies on smaller versions of this goal based MiniPacMan environment environments ( 6 x 6 maze), but evaluate them on larger versions (11 X 11) throughout training.

4.7 MIDDLE GROUND BETWEEN MODEL BASED RL AND MODEL FREE RL
We further demonstrate the idea of decision states in a planning goal-based navigation task that uses a combination of model-based and model-free RL. Identifying useful decision states can provide a comfortable middle ground between model-free reasoning and model-based planning. For example, imagine planning over individual decision states, while using model-free knowledge to navigate between bottlenecks: aspects of the environment that are physically complex but vary little between problem instances are handled in a model-free way (the navigation between decision points), while the particular decision points that are relevant to the task can be handled by explicitly reasoning about causality, in a more abstract representation. We demonstrate this using a similar setup as in imagination augmented agents (Weber et al., 2017). In imagination augmented agents, model free agents are augmented with imagination, such that the imagination model can be queried to make predictions about the future. We use the dynamics models to simulate imagined trajectories, which are then summarized by a neural network and this summary is provided as additional context to a policy network. Here, we use the output of the imagination module as a “goal” and we want to show that only near the decision points (i.e potential subgoals) the agent wants to make use of the information which is a result of running imagination module.


5 CONCLUSION
In this paper, we proposed to train agents to develop “default behaviours” as well as the knowledge of when to break those behaviour, using an information bottleneck between the agent’s goal and policy. We demonstrated empirically that this training procedure leads to better direct policy transfer across tasks. We also demonstrated that the states in which the agent learns to deviate from its habits, which we call ”decision states”, can be used as the basis for a learned exploration bonus that leads to more effective training than other task-agnostic exploration methods.



--------------------------------


 


























下面公众号发过了。

-----------

Variational Option Discovery Algorithms   :: real hierarchical


We show that Variational Intrinsic Control (VIC) (Gregor et al. [2016]) and
the recently-proposed Diversity is All You Need (DIAYN) (Eysenbach et al. [2018]) are specific
instances of this template which decode from states instead of complete trajectories.

abs:
two algorithmic contributions. First: we highlight a tight connection between variational option discovery methods and variational autoencoders

In VALOR, the policy encodes contexts from a noise distribution into trajectories, and the decoder recovers the contexts from the complete trajectories. Second: we propose a curriculum learning approach 

1
Humans are innately driven to experiment with new ways of interacting with their environments.
This can accelerate the process of discovering skills for downstream tasks and can also be viewed as a primary objective in its own right. This drive serves as an inspiration for reward-free option discovery


In our analogy, a policy acts as an encoder, translating contexts from a noise distribution into trajectories; a decoder attempts to recover the contexts from the trajectories, and rewards the policies for making contexts easy to distinguish. Contexts are random vectors which have no intrinsic meaning prior to training, but they become associated with trajectories as a result of training; each context vector thus corresponds to a distinct option. Therefore this approach learns a set of options which are as diverse as possible, in the sense of being as easy to distinguish from each other as possible. We show that Variational Intrinsic Control (VIC) (Gregor et al. [2016]) and the recently-proposed Diversity is All You Need (DIAYN) (Eysenbach et al. [2018]) are specific instances of this template which decode from states instead of complete trajectories.
We make two main algorithmic contributions:
 1 .The idea is to encourage learning dynamical modes instead of goal-attaining modes, e.g. ‘move in a circle’ instead of ‘go to X’.
 2. We propose a curriculum learning approach where the number of contexts seen by the agent increases whenever the agent’s performance is strong enough (as measured by the decoder) on the current set of contexts.

show that, to the extent that our metrics can measure, all three of them perform similarly, except that VALOR can attain qualitatively different behavior because of its trajectory-centric approach, and DIAYN learns more quickly because of its denser reward signal. We show that our curriculum trick stabilizes and speeds up learning for all three methods, and can allow a single agent to learn up to hundreds of modes. Beyond our core comparison, we also explore applications of variational option discovery in two interesting spotlight environments: a simulated robot hand and a simulated humanoid. Variational option discovery finds naturalistic finger-flexing behaviors in the hand environment, but performs poorly on the humanoid, in the sense that it does not discover natural crawling or walking gaits. We consider this evidence that pure information-theoretic objectives can do a poor job of capturing human priors on useful behavior in complex environments
using a (particularly good) pretrained VALOR policy as the lower level of a hierarchy. In this experiment, we find that the VALOR policy is more useful than a random network as a lower level, and equivalently as useful as learning a lower level from scratch in the environment.

2
option Discovery
Several approaches for option discovery are primarily information-theoretic: Gregor et al. [2016], Eysenbach et al. [2018], and Florensa et al. [2017] train policies to maximize mutual information between options and states or quantities derived from states; by contrast, we maximize information between options and whole trajectories

Universal Policies
Universal Policies: Variational option discovery algorithms learn universal policies (goal- or instruction- conditioned policies)
By contrast, variational option discovery is unsupervised and finds its own instruction space.

Intrinsic Motivation:
However, none of these approaches were combined with learning universal policies, and so suffer from a problem of knowledge fade
Variational Autoencoders

Novelty Search:

3 Variational Option Discovery Algorithms



其他方法的互信息最大？？











--------------------------
DIVERSITY IS ALL YOU NEED:
LEARNING SKILLS WITHOUT A REWARD FUNCTION


DIVERSITY IS ALL YOU NEED: 充满睿智的论述。

perfect！ beautiful！ wonderful！

互信息从state action 到 state skill；action到skill就是提升一级抽象

cpc 视频预测也应该从之前的action到skill的。

：：：：
abs：
propose “Diversity is All You Need”(DIAYN), a method for learning useful skills without a reward function. Our proposed method learns skills by maximizing an information theoretic objective using a maximum entropy policy. On a variety of simulated robotic tasks, we show that this simple objective results in the unsupervised emergence of diverse skills, such as walking and jumping. In a number of reinforcement learning benchmark environments, our method is able to learn a skill that solves the benchmark task despite never receiving the true task reward. We show how pretrained skills can provide a good parameter initialization for downstream tasks, and can be composed hierarchically to solve complex, sparse reward tasks. Our results suggest that unsupervised discovery of skills can serve as an effective pretraining mechanism for overcoming challenges of exploration and data efficiency in reinforcement learning.
1
intelligent creatures can explore their environments and learn useful skills even without supervision, so that when they are later faced with specific goals, they can use those skills to satisfy the new goals quickly and efficiently.
Learning useful skills without supervision may help address challenges in exploration in these environments. For long horizon tasks, skills discovered without reward can serve as primitives for hierarchical RL, effectively shortening the episode length. In many practical settings, interacting with the environment is essentially free, but evaluating the reward requires human feedback (Christiano et al., 2017). Unsupervised learning of skills may reduce the amount of supervision necessary to learn a task. While we can take the human out of the loop by designing a reward function, it is challenging to design a reward function that elicits the desired behaviors from the agent

A skill is a latent-conditioned policy that alters that state of the environment in a consistent way.
 
learning objective that ensures that each skill individually is distinct and that the skills collectively explore large parts of the state space.
These skills are useful for a number of applications, including hierarchical reinforcement learning and imitation learning.
    
A key idea in our work is to use discriminability between skills as an objective. Further, skills that are distinguishable are not necessarily maximally diverse – a slight difference in states makes two skills distinguishable, but not necessarily diverse in a semantically meaningful way. To combat problem, we want to learn skills that not only are distinguishable, but also are as diverse as possible. By learning distinguishable skills that are as random as possible, we can “push” the skills away from each other, making each skill robust to perturbations and effectively exploring the environment. By maximizing this objective, we can learn skills that run forward, do backflips, skip backwards, and perform face flops (

five contributions. First, we propose a method for learning useful skills without any rewards. We formalize our discriminability goal as maximizing an information theoretic objective with a maximum entropy policy. Second, we show that this simple exploration objective results in the unsupervised emergence of diverse skills
Third, we propose a simple method for using learned skills for hierarchical RL and find this methods solves challenging tasks. Four, we demonstrate how skills discovered can be quickly adapted to solve a new task. Finally, we show how skills discovered can be used for imitation learning.

2
Previous work on hierarchical RL has learned skills to maximize a single, known, reward function by jointly learning a set of skills and a meta-controller 
One problem with joint training (also noted by Shazeer et al. (2017)) is that the meta-policy does not select “bad” options, so these options do not receive any reward signal to improve.
Our work prevents this degeneracy by using a random meta-policy during unsupervised skill-learning, such that neither the skills nor the meta-policy are aiming to solve any single task. A second importance difference is that our approach learns skills with no reward. Eschewing a reward function not only avoids the difficult problem of reward design, but also allows our method to learn task-agnostic.

and Jung et al. (2011) use the mutual information between states and actions as a notion of empowerment for an intrinsically motivated agent. Our method maximizes the mutual information between states and skills, which can be interpreted as maximizing the empowerment of a hierarchical agent whose action space is the set of skills
and Gregor et al. (2016) showed that a discriminability objective is equivalent to maximizing the mutual information between the latent skill z and some aspect of the corresponding trajectory

Three important distinctions allow us to apply our method to tasks significantly more complex than the gridworlds in Gregor et al. (2016). First, we use maximum entropy policies to force our skills to be diverse. Our theoretical analysis shows that including entropy maximization in the RL objective results in the mixture of skills being maximum entropy in aggregate. Second, we fix the prior distribution over skills, rather than learning it. Doing so prevents our method from collapsing to sampling only a handful of skills. Third, while the discriminator in Gregor et al. (2016) only looks at the final state, our discriminator looks at every state, which provides additional reward signal. These three crucial differences help explain how our method learns useful skills in complex environments.
we aim to acquire complex skills with minimal supervision to improve efficiency (i.e., reduce the number of objective function queries) and as a stepping stone for imitation learning and hierarchical RL. We focus on deriving a general, information-theoretic objective that does not require manual design of distance metrics and can be applied to any RL task without additional engineering.
While these previous works use an intrinsic motivation objective to learn a single policy, we propose an objective for learning many, diverse policies.

3
the aim of the unsupervised stage is to learn skills that eventually will make it easier to maximize the task reward in the supervised stage. Conveniently, because skills are learned without a priori knowledge of the task, the learned skills can
be used for many different tasks.
3.1
 
3.1 HOW IT WORKS
Our method for unsupervised skill discovery, DIAYN (“Diversity is All You Need”), builds off of three ideas. First, for skills to be useful, we want the skill to dictate the states that the agent visits. Different skills should visit different states, and hence be distinguishable. Second, we want to use states, not actions, to distinguish skills, because actions that do not affect the environment are not visible to an outside observer. For example, an outside observer cannot tell how much force a robotic arm applies when grasping a cup if the cup does not move. Finally, we encourage exploration and incentivize the skills to be as diverse as possible by learning skills that act as randomly as possible. Skills with high entropy that remain discriminable must explore a part of the state space far away from other skills, lest the randomness in its actions lead it to states where it cannot be distinguished.

公式看论文吧。

4 EXPERIMENTS
4.1 ANALYSIS OF LEARNED SKILLS
Question 1. What skills does DIAYN learn?
Question 2. How does the distribution of skills change during training?
Question 3. Does discriminating on single states restrict DIAYN to learn skills that visit disjoint sets of states?
Our discriminator operates at the level of states, not trajectories.   ref Variational Option Discovery Algorithms
Question 4. How does DIAYN differ from Variational Intrinsic Control (VIC)
 In contrast, DIAYN fixes the distribution over skills, which allows us to discover more diverse skills.
4.2 HARNESSING LEARNED SKILLS
 Three less obvious applications are adapting skills to maximize a reward, hierarchical RL, and imitation learning.
4.2.1 ACCELERATING LEARNING WITH POLICY INITIALIZATION
we propose that DIAYN can serve as unsupervised pre-training for more sample-efficient finetuning of task-specific policies.
Question 5. Can we use learned skills to directly maximize the task reward?

4.2.2 USING SKILLS FOR HIERARCHICAL RL
In theory, hierarchical RL should decompose a complex task into motion primitives, which may be reused for multiple tasks. In practice, algorithms for hierarchical RL can encounter many problems: (1) each motion primitive reduces to a single action (Bacon et al., 2017), (2) the hierarchical policy only samples a single motion primitive (Gregor et al., 2016), or (3) all motion primitives attempt to do the entire task. In contrast, DIAYN discovers diverse, task-agnostic skills, which hold the promise of acting as a building block for hierarchical RL.

Question 6. Are skills discovered by DIAYN useful for hierarchical RL?
To use the discovered skills for hierarchical RL, we learn a meta-controller whose actions are to choose which skill to execute for the next k steps (100 for ant navigation, 10 for cheetah hurdle). The meta-controller has the same observation space as the skills.
VIME attempts to learn a single policy that visits many states.
Figure 7: DIAYN for Hierarchical RL: By learning a meta-controller to compose skills learned by DIAYN, cheetah quickly learns to jump over hurdles and ant solves a sparse-reward navigation task.

Question 7. How can DIAYN leverage prior knowledge about what skills will be useful?
4.2.3 IMITATING AN EXPERT
Question 8. Can we use learned skills to imitate an expert?



5 CONCLUSION
In this paper, we present DIAYN, a method for learning skills without reward functions. We show that DIAYN learns diverse skills for complex tasks, often solving benchmark tasks with one of the learned skills without actually receiving any task reward. We further proposed methods for using the learned skills (1) to quickly adapt to a new task, (2) to solve complex tasks via hierarchical RL, and (3) to imitate an expert. As a rule of thumb, DIAYN may make learning a task easier by replacing the task’s complex action space with a set of useful skills. DIAYN could be combined with methods for augmenting the observation space and reward function. Using the common language of information theory, a joint objective can likely be derived. DIAYN may also more efficiently learn from human preferences by having humans select among learned skills. Finally, the skills produced by DIAYN might be used by game designers to allow players to control complex robots and by artists to animate characters.





















-----------------------
EMI：

2 RELATED WORKS
Our work is related to the following strands of active research:
Unsupervised representation learning via mutual information estimation
cpc mine DIM 。。。

2.2

VIME (Houthooft et al. (2016)) approximates the environment dynamics, uses the information gain of the learned dynamics model as intrinsic rewards, and showed encouraging results on robotic locomotion problems. However, the method needs to update the dynamics model per each observation and is unlikely to be scalable for complex tasks with high dimensional states such as Atari games.

Other approaches utilize more ad-hoc measures (Pathak et al., 2017; Tang et al., 2017) that aim to approximate surprise. ICM (Pathak et al. (2017)) transforms the high dimensional states to feature space and imposes cross entropy and euclidean loss so the action and the feature of the next state are predictable. However, ICM does not utilize the mutual information like VIME to directly measure the uncertainty and is limited to discrete actions. Our method (EMI) is also reminiscent of (Kohonen & Somervuo, 1998) in a sense that we seek to construct a decoder-free latent space from the high dimensional observation data with a topology in the latent space. In contrast to the prior works on exploration, we seek to construct the representation under linear topology and does not require decoding the full observation but seek to encode the essential predictive signal that can be used for guiding the exploration.











