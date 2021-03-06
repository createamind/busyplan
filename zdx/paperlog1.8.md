


https://scholar.google.com/citations?hl=en&user=8R35rCwAAAAJ&view_op=list_works&sortby=pubdate


公众号未深入看的

concept learning

infobot
VALOR

Imagination-Augmented Agents for Deep Reinforcement Learning

UNSUPERVISED CONTROL THROUGH NON-PARAMETRIC DISCRIMINATIVE REWARDS







LEARNING A PRIOR OVER INTENT VIA META-INVERSE REINFORCEMENT LEARNING
ABSTRACT
A significant challenge for the practical application of reinforcement learning to real world problems is the need to specify an oracle reward function that correctly defines a task. Inverse reinforcement learning (IRL) seeks to avoid this challenge by instead inferring a reward function from expert behavior. While appealing, it can be impractically expensive to collect datasets of demonstrations that cover the variation common in the real world (e.g. opening any type of door). Thus in practice, IRL must commonly be performed with only a limited set of demon- strations where it can be exceedingly difficult to unambiguously recover a reward function. In this work, we exploit the insight that demonstrations from other tasks can be used to constrain the set of possible reward functions by learning a “prior” that is specifically optimized for the ability to infer expressive reward functions from limited numbers of demonstrations. We demonstrate that our method can efficiently recover rewards from images for novel tasks and provide intuition as to how our approach is analogous to learning a prior.

1
Our approach relies on the key observation that related tasks share common structure that we can leverage when learning new tasks. To illustrate, considering a robot navigating through a home.
While the exact reward function we provide to the robot may differ depending on the task, there is a structure amid the space of useful behaviours, such as navigating to a series of landmarks, and there are certain behaviors we always want to encourage or discourage, such as avoiding obstacles or staying a reasonable distance from humans. This notion agrees with our understanding of why humans can easily infer the intents and goals (i.e., reward functions) of even abstract agents from just one or a few demonstrations Baker et al. (2007), as humans have access to strong priors about how other humans accomplish similar tasks accrued over many years. Similarly, our objective is to discover the common structure among different tasks, and encode the structure in a way that can be used to infer reward functions from a few demonstrations.



3.1
Learning in general energy-based models of this form is common in many applications such as structured prediction. However, in contrast to applications where learning can be supervised by millions of labels (e.g. semantic segmentation), the learning problem in Eq. 3 must typically be performed with a relatively small number of example demonstrations. In this work, we seek to address this issue in IRL by providing a way to integrate information from prior tasks to constrain the optimization in Eq. 3 in the regime of limited demonstrations.


4 LEARNING TO LEARN REWARDS
Our goal in meta-IRL is to learn how to learn reward functions across many tasks such that the model can infer the reward function for a new task using only one or a few expert demonstrations. Intuitively, we can view this problem as aiming to learn a prior over the intentions of human demon- strators, such that when given just one or a few demonstrations of a new task, we can combine the learned prior with the new data to effectively determine the human’s reward function. Such a prior is helpful in inverse reinforcement learning settings, since the space of relevant reward functions is much smaller than the space of all possible rewards definable on the raw observations.

Erin Grant, Chelsea Finn, Sergey Levine, Trevor Darrell, and Thomas Griffiths. Recasting gradient- based meta-learning as hierarchical bayes. International Conference on Learning Representations (ICLR), 2018.

4.2 INTERPRETATION AS LEARNING A PRIOR OVER INTENT
The objective in Eq. 6 optimizes for parameters that enable that reward function to adapt and gener- alize efficiently on a wide range of tasks. Intuitively, constraining the space of reward functions to lie within a few steps of gradient descent can be interpreted as expressing a “locality” prior over reward function parameters. This intuition can be made more concrete with the following analysis.
By viewing IRL as maximum likelihood estimation, we
can take the perspective of Grant et al. (2018) who
showed that for a linear model, fast adaptation via a few
steps of gradient descent in MAML is performing MAP
inference over φ, under a Gaussian prior with the mean
θ and a covariance that depends on the step size, num-
ber of steps and curvature of the loss. This is based on
the connection between early stopping and regularization
previously discussed in Santos (1996), which we refer
the readers to for a more detailed discussion. The in-
terpretation of MAML as imposing a Gaussian prior on
the parameters is exact in the case of a likelihood that is
quadratic in the parameters (such as the log-likelihood of
a Gaussian in terms of its mean). For any non-quadratic
likelihood, this is an approximation in a local neighbor-
hood around θ (i.e. up to convex quadratic approxima-
tion). In the case of very complex parameterizations, such
as deep function approximators, this is a coarse approximation and unlikely to be the mode of a pos- terior. However, we can still frame the effect of early stopping and initialization as serving as a prior in a similar way as prior work (Sjo ̈berg & Ljung, 1995; Duvenaud et al., 2016; Grant et al., 2018).
More importantly, this interpretation hints at future extensions to our approach that could benefit from employing more fully Bayesian approaches to reward and goal inference.





then ref by  Few-Shot Goal Inference for Visuomotor Learning and Planning

then ref by visual foresight















Meta-Reinforcement Learning of Structured Exploration Strategies  https://www.arxiv-vanity.com/papers/1802.07245/
Abstract
Exploration is a fundamental challenge in reinforcement learning (RL). Many of the current exploration methods for deep RL use task-agnostic objectives, such as information gain or bonuses based on state visitation. However, many practical applications of RL involve learning more than a single task, and prior tasks can be used to inform how exploration should be performed in new tasks. In this work, we explore how prior tasks can inform an agent about how to explore effectively in new situations. We introduce a novel gradient-based fast adaptation algorithm – model agnostic exploration with structured noise (MAESN) – to learn exploration strategies from prior experience. The prior experience is used both to initialize a policy and to acquire a latent exploration space that can inject structured stochasticity into a policy, producing exploration strategies that are informed by prior knowledge and are more effective than random action-space noise. We show that MAESN is more effective at learning exploration strategies when compared to prior meta-RL methods, RL without learned exploration strategies, and task-agnostic exploration methods. We evaluate our method on a variety of simulated tasks: locomotion with a wheeled robot, locomotion with a quadrupedal walker, and object manipulation.





Solar: Deep structured latent representations for model-based reinforcement learning
Authors
Marvin Zhang, Sharad Vikram, Laura Smith, Pieter Abbeel, Matthew J Johnson, Sergey Levine
Publication date
2018/8/28
Journal
arXiv preprint arXiv:1808.09105
Description
Model-based reinforcement learning (RL) methods can be broadly categorized as global model methods, which depend on learning models that provide sensible predictions in a wide range of states, or local model methods, which iteratively refit simple models that are used for policy improvement. While predicting future states that will result from the current actions is difficult, local model methods only attempt to understand system dynamics in the neighborhood of the current policy, making it possible to produce local improvements without ever learning to predict accurately far into the future. The main idea in this paper is that we can learn representations that make it easy to retrospectively infer simple dynamics given the data from the current policy, thus enabling local models to be used for policy learning in complex systems. To that end, we focus on learning representations with probabilistic graphical model (PGM) structure, which allows us to devise an efficient local model method that infers dynamics from real-world rollouts with the PGM as a global prior. We compare our method to other model-based and model-free RL methods on a suite of robotics tasks, including manipulation tasks on a real Sawyer robotic arm directly from camera images. Videos of our results are available at this https URL






https://arxiv.org/pdf/1812.10157.pdf
Motion Selective Prediction for Video Frame Synthesis
Veronique Prinet ´
The Hebrew University of Jerusalem
vprinet@gmail.com
Abstract
Existing conditional video prediction approaches train a
network from large databases and generalise to previously
unseen data. We take the opposite stance, and introduce a
model that learns from the first frames of a given video and
extends its content and motion, to, e.g., double its length.
To this end, we propose a dual network that can use in a
flexible way both dynamic and static convolutional motion
kernels, to predict future frames. The construct of our model
gives us the the means to efficiently analyse its functioning
and interpret its output. We demonstrate experimentally the
robustness of our approach on challenging videos in-thewild and show that it is competitive w.r.t. related baselines












Automata Guided Reinforcement Learning With Demonstrations





DEEP VARIATIONAL BAYES FILTERS: UNSUPERVISED
LEARNING OF STATE SPACE MODELS FROM RAW
DATA
Maximilian Karl, Maximilian Soelch, Justin Bayer, Patrick van der Smagt
Data Lab, Volkswagen Group, 80805, München, Germany
zip([maximilian.karl, maximilian.soelch], [@volkswagen.de])
ABSTRACT
We introduce Deep Variational Bayes Filters (DVBF), a new method for unsupervised learning and identification of latent Markovian state space models. Leveraging recent advances in Stochastic Gradient Variational Bayes, DVBF can overcome
intractable inference distributions via variational inference. Thus, it can handle
highly nonlinear input data with temporal and spatial dependencies such as image
sequences without domain knowledge. Our experiments show that enabling backpropagation through transitions enforces state space assumptions and significantly
improves information content of the latent embedding. This also enables realistic
long-term prediction.








https://blog.openai.com/quantifying-generalization-in-reinforcement-learning/





Causal Confusion in Imitation Learning





-------------------------------------------
PVEs: Position-Velocity Encoders for Unsupervised Learning of Structured State Representations






---------------------------------------------------------------

UPN  https://arxiv.org/pdf/1804.00645.pdf




SE3-Pose-Nets: Structured Deep Dynamics Models for Visuomotor Planning and Control  https://rse-lab.cs.washington.edu/se3-structured-deep-ctrl/




Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
https://github.com/wiewel/LatentSpacePhysics





https://github.com/dazhang-cv/S3D  
S3D: Single Shot multi-Span Detector via Fully 3D Convolutional Networks





ICML 2018 深度学习论文及代码集锦(1)  微信文章





---------------------------------------------------------
SFV: Reinforcement Learning of Physical Skills from Videos




-------------------------------
TACO: Learning Task Decomposition via Temporal Alignment for Control
Abstract
Many advanced Learning from Demonstration (LfD) methods consider the decomposition of complex, real-world tasks into simpler sub-tasks. By reusing the corresponding sub-policies within and between tasks, they provide training data for each policy from different high-level tasks and compose them to perform novel ones. However, most existing approaches to modular LfD focus either on learning a single high-level task or de- pend on domain knowledge and temporal seg- mentation. By contrast, we propose a weakly supervised, domain-agnostic approach based on task sketches, which include only the sequence of sub-tasks performed in each demonstration. Our approach simultaneously aligns the sketches with the observed demonstrations and learns the re- quired sub-policies, which improves generalisa- tion in comparison to separate optimisation proce- dures. We evaluate the approach on multiple do- mains, including a simulated 3D robot arm control task using purely image-based observations. The approach performs commensurately with fully su- pervised approaches, while requiring significantly less annotation effort, and significantly outper- forms methods which separate segmentation and imitation.






-----------------------------------------------------

Improvements on Hindsight Learning https://arxiv.org/abs/1809.06719




Hindsight policy gradients https://arxiv.org/abs/1711.06006






key paper:
LEARNING AN EMBEDDING SPACE FOR TRANSFERABLE ROBOT SKILLS  2018  https://openreview.net/pdf?id=rk07ZXZRb

abs：
skill embedding space. We learn such skills by taking advantage of latent variables and exploiting a connection between reinforcement learning and variational inference. The main contribution of our work is an entropyregularized policy gradient formulation for hierarchical policies, and an associated, data-efficient and robust off-policy gradient algorithm based on stochastic
value gradients

 solving the control problem in the embedding space rather
than the raw action space.
main contribution of our work is an entropyregularized policy gradient formulation for hierarchical policies, and an associated, data-efficient
and robust off-policy gradient algorithm based on stochastic value gradients.
Our formulation draws on a connection between reinforcement learning and variational inference
and is a principled and general scheme for learning hierarchical stochastic policies. We show how
stochastic latent variables can be meaningfully incorporated into policies by treating them in the
same way as auxiliary variables in parametric variational approximations in inference (Salimans
et al. 2014; Maaløe et al. 2016; Ranganath et al. 2016). The resulting policies can model complex
correlation structure and multi-modality in action space. We represent the skill embedding via such
latent variables and find that this view naturally leads to an information-theoretic regularization
which ensures that the learned skills are versatile and the embedding space is well formed.


???? deepmind互信息压缩技能的论文。










Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review
Sergey Levine
(Submitted on 2 May 2018 (v1), last revised 20 May 2018 (this version, v3))
The framework of reinforcement learning or optimal control provides a mathematical formalization of intelligent decision making that is powerful and broadly applicable. While the general form of the reinforcement learning problem enables effective reasoning about uncertainty, the connection between reinforcement learning and inference in probabilistic models is not immediately obvious. However, such a connection has considerable value when it comes to algorithm design: formalizing a problem as probabilistic inference in principle allows us to bring to bear a wide array of approximate inference tools, extend the model in flexible and powerful ways, and reason about compositionality and partial observability. In this article, we will discuss how a generalization of the reinforcement learning or optimal control problem, which is sometimes termed maximum entropy reinforcement learning, is equivalent to exact probabilistic inference in the case of deterministic dynamics, and variational inference in the case of stochastic dynamics. We will present a detailed derivation of this framework, overview prior work that has drawn on this and related ideas to propose new reinforcement learning and control algorithms, and describe perspectives on future research.




Advances in Variational Inference                Cheng Zhang Judith Bu ̈tepage Hedvig Kjellstro ̈m Stephan Mandt
Abstract—Many modern unsupervised or semi-supervised machine learning algorithms rely on Bayesian probabilistic models. These models are usually intractable and thus require approximate inference. Variational inference (VI) lets us approximate a high-dimensional Bayesian posterior with a simpler variational distribution by solving an optimization problem. This approach has been successfully used in various models and large-scale applications. In this review, we give an overview of recent trends in variational inference. We first introduce standard mean field variational inference, then review recent advances focusing on the following aspects: (a) scalable VI, which includes stochastic approximations, (b) generic VI, which extends the applicability of VI to a large class of otherwise intractable models, such as non-conjugate models, (c) accurate VI, which includes variational models beyond the mean field approximation or with atypical divergences, and (d) amortized VI, which implements the inference over local latent variables with inference networks. Finally, we provide a summary of promising future research directions.












-------------------------------------------
The Implicit Preference Information in an Initial State  https://openreview.net/forum?id=rkevMnRqYQ

Rohin Shah, Dmitrii Krasheninnikov, Jordan Alexander, Pieter Abbeel, Anca Dragan
28 Sep 2018 (modified: 21 Dec 2018)ICLR 2019 Conference Blind SubmissionReaders:  EveryoneShow BibtexShow Revisions
Abstract: Reinforcement learning (RL) agents optimize only the features specified in a reward function and are indifferent to anything left out inadvertently. This means that we must not only specify what to do, but also the much larger space of what not to do. It is easy to forget these preferences, since these preferences are already satisfied in our environment. This motivates our key insight: when a robot is deployed in an environment that humans act in, the state of the environment is already optimized for what humans want. We can therefore use this implicit preference information from the state to fill in the blanks. We develop an algorithm based on Maximum Causal Entropy IRL and use it to evaluate the idea in a suite of proof-of-concept environments designed to show its properties. We find that information from the initial state can be used to infer both side effects that should be avoided as well as preferences for how the environment should be organized.
Keywords: Preference learning, Inverse reinforcement learning, Inverse optimal stochastic control, Maximum entropy reinforcement learning, Apprenticeship learning
TL;DR: When a robot is deployed in an environment that humans have been acting in, the state of the environment is already optimized for what humans want, and we can use this to infer human preferences.

Maximum Causal Entropy IRL

























-------------------------------------------
VMAV-C: A Deep Attention-based Reinforcement Learning Algorithm for Model-based Control
Abstract
Recent breakthroughs in Go play and strategic games have witnessed the great potential of reinforcement learning in intelligently scheduling in uncertain environment, but some bottlenecks are also encountered when we generalize this paradigm to universal complex tasks. Among them, the low efficiency of data utilization in model-free reinforcement algorithms is of great concern. In contrast, the model-based reinforcement learning algorithms can reveal underlying dynamics in learning environments and seldom suffer the data utilization problem. To address the problem, a model-based reinforcement learning algorithm with attention mechanism embedded is proposed as an extension of World Models in this paper. We learn the environment model through Mixture Density Network Recurrent Network(MDN-RNN) for agents to interact, with combinations of variational auto-encoder(VAE) and attention incorporated in state value estimates during the process of learning policy. In this way, agent can learn optimal policies through less interactions with actual environment, and final experiments demonstrate the effectiveness of our model in control problem.



ta




Dynamic Movement Primitives in Latent Space of Time-Dependent  ？? sfv   ?？跳舞VAE；  https://github.com/jsn5/dancenet
Variational Autoencoders
Nutan Chen, Maximilian Karl, Patrick van der Smagt
Abstract— Dynamic movement primitives (DMPs) are powerful for the generalization of movements from demonstration.
However, high dimensional movements, as they are found in
robotics, make finding efficient DMP representations difficult.
Typically, they are either used in configuration or Cartesian
space, but both approaches do not generalize well. Additionally,
limiting DMPs to single demonstrations restricts their generalization capabilities.
In this paper, we explore a method that embeds DMPs into
the latent space of a time-dependent variational autoencoder
framework. Our method enables the representation of highdimensional movements in a low-dimensional latent space.
Experimental results show that our framework has excellent
generalization in the latent space, e.g., switching between movements or changing goals. Also, it generates optimal movements
when reproducing the movements.






Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations
Francesco Locatello, Stefan Bauer, Mario Lucic, Sylvain Gelly, Bernhard Schölkopf, Olivier Bachem
(Submitted on 29 Nov 2018 (v1), last revised 2 Dec 2018 (this version, v2))
In recent years, the interest in unsupervised learning of disentangled representations has significantly increased. The key assumption is that real-world data is generated by a few explanatory factors of variation and that these factors can be recovered by unsupervised learning algorithms. A large number of unsupervised learning approaches based on auto-encoding and quantitative evaluation metrics of disentanglement have been proposed; yet, the efficacy of the proposed approaches and utility of proposed notions of disentanglement has not been challenged in prior work. In this paper, we provide a sober look on recent progress in the field and challenge some common assumptions. 
We first theoretically show that the unsupervised learning of disentangled representations is fundamentally impossible without inductive biases on both the models and the data. Then, we train more than 12000 models covering the six most prominent methods, and evaluate them across six disentanglement metrics in a reproducible large-scale experimental study on seven different data sets. On the positive side, we observe that different methods successfully enforce properties "encouraged" by the corresponding losses. On the negative side, we observe in our study that well-disentangled models seemingly cannot be identified without access to ground-truth labels even if we are allowed to transfer hyperparameters across data sets. Furthermore, increased disentanglement does not seem to lead to a decreased sample complexity of learning for downstream tasks. 
These results suggest that future work on disentanglement learning should be explicit about the role of inductive biases and (implicit) supervision, investigate concrete benefits of enforcing disentanglement of the learned representations, and consider a reproducible experimental setup covering several data sets.







Towards a Definition of Disentangled Representations
Irina Higgins, David Amos, David Pfau, Sebastien Racaniere, Loic Matthey, Danilo Rezende, Alexander Lerchner
(Submitted on 5 Dec 2018)
How can intelligent agents solve a diverse set of tasks in a data-efficient manner? The disentangled representation learning approach posits that such an agent would benefit from separating out (disentangling) the underlying structure of the world into disjoint parts of its representation. However, there is no generally agreed-upon definition of disentangling, not least because it is unclear how to formalise the notion of world structure beyond toy datasets with a known ground truth generative process. Here we propose that a principled solution to characterising disentangled representations can be found by focusing on the transformation properties of the world. In particular, we suggest that those transformations that change only some properties of the underlying world state, while leaving all other properties invariant, are what gives exploitable structure to any kind of data. Similar ideas have already been successfully applied in physics, where the study of symmetry transformations has revolutionised the understanding of the world structure. By connecting symmetry transformations to vector representations using the formalism of group and representation theory we arrive at the first formal definition of disentangled representations. Our new definition is in agreement with many of the current intuitions about disentangling, while also providing principled resolutions to a number of previous points of contention. While this work focuses on formally defining disentangling - as opposed to solving the learning problem - we believe that the shift in perspective to studying data transformations can stimulate the development of better representation learning algorithms.









-------------------------------------------
Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
Steffen Wiewel, Moritz Becher, Nils Thuerey
(Submitted on 27 Feb 2018 (v1), last revised 15 Jun 2018 (this version, v2))
Our work explores methods for the data-driven inference of temporal evolutions of physical functions with deep learning techniques. More specifically, we target fluid flow problems, and we propose a novel LSTM-based approach to predict the changes of the pressure field over time. The central challenge in this context is the high dimensionality of Eulerian space-time data sets. Key for arriving at a feasible algorithm is a technique for dimensionality reduction based on convolutional neural networks, as well as a special architecture for temporal prediction. We demonstrate that dense 3D+time functions of physics system can be predicted with neural networks, and we arrive at a neural-network based simulation algorithm with significant practical speed-ups. We demonstrate the capabilities of our method with a series of complex liquid simulations, and with a set of single-phase buoyancy simulations. With a set of trained networks, our method is more than two orders of magnitudes faster than a traditional pressure solver. Additionally, we present and discuss a series of detailed evaluations for the different components of our algorithm.






Deep Variational Bayes Filters: Unsupervised Learning of State Space Models from Raw Data            http://brml.org/projects/dvbf/      https://arxiv.org/pdf/1605.06432.pdf

Maximilian Karl, Maximilian Soelch, Justin Bayer, Patrick van der Smagt
(Submitted on 20 May 2016 (v1), last revised 3 Mar 2017 (this version, v3))
We introduce Deep Variational Bayes Filters (DVBF), a new method for unsupervised learning and identification of latent Markovian state space models. Leveraging recent advances in Stochastic Gradient Variational Bayes, DVBF can overcome intractable inference distributions via variational inference. Thus, it can handle highly nonlinear input data with temporal and spatial dependencies such as image sequences without domain knowledge. Our experiments show that enabling backpropagation through transitions enforces state space assumptions and significantly improves information content of the latent embedding. This also enables realistic long-term prediction.






Variational Encoding of Complex Dynamics  https://arxiv.org/abs/1711.08576
Carlos X. Hernández, Hannah K. Wayment-Steele, Mohammad M. Sultan, Brooke E. Husic, Vijay S. Pande
(Submitted on 23 Nov 2017 (v1), last revised 2 Dec 2017 (this version, v2))
Often the analysis of time-dependent chemical and biophysical systems produces high-dimensional time-series data for which it can be difficult to interpret which individual features are most salient. While recent work from our group and others has demonstrated the utility of time-lagged co-variate models to study such systems, linearity assumptions can limit the compression of inherently nonlinear dynamics into just a few characteristic components. Recent work in the field of deep learning has led to the development of variational autoencoders (VAE), which are able to compress complex datasets into simpler manifolds. We present the use of a time-lagged VAE, or variational dynamics encoder (VDE), to reduce complex, nonlinear processes to a single embedding with high fidelity to the underlying dynamics. We demonstrate how the VDE is able to capture nontrivial dynamics in a variety of examples, including Brownian dynamics and atomistic protein folding. Additionally, we demonstrate a method for analyzing the VDE model, inspired by saliency mapping, to determine what features are selected by the VDE model to describe dynamics. The VDE presents an important step in applying techniques from deep learning to more accurately model and interpret complex biophysics.














A Disentangled Recognition and Nonlinear Dynamics Model for Unsupervised Learning               https://github.com/simonkamronn/kvae
Marco Fraccaro, Simon Kamronn, Ulrich Paquet, Ole Winther
(Submitted on 16 Oct 2017 (v1), last revised 30 Oct 2017 (this version, v2))
This paper takes a step towards temporal reasoning in a dynamically changing video, not in the pixel space that constitutes its frames, but in a latent space that describes the non-linear dynamics of the objects in its world. We introduce the Kalman variational auto-encoder, a framework for unsupervised learning of sequential data that disentangles two latent representations: an object's representation, coming from a recognition model, and a latent state describing its dynamics. As a result, the evolution of the world can be imagined and missing data imputed, both without the need to generate high dimensional frames at each time step. The model is trained end-to-end on videos of a variety of simulated physical systems, and outperforms competing methods in generative and missing data imputation tasks.

two latent representations: an object's representation, coming from a recognition model, and a latent state describing its dynamics. 







Episodic Curiosity through Reachability
Nikolay Savinov, Anton Raichuk, Raphaël Marinier, Damien Vincent, Marc Pollefeys, Timothy Lillicrap, Sylvain Gelly
(Submitted on 4 Oct 2018 (v1), last revised 1 Dec 2018 (this version, v2))
Rewards are sparse in the real world and most today's reinforcement learning algorithms struggle with such sparsity. One solution to this problem is to allow the agent to create rewards for itself - thus making rewards dense and more suitable for learning. In particular, inspired by curious behaviour in animals, observing something novel could be rewarded with a bonus. Such bonus is summed up with the real task reward - making it possible for RL algorithms to learn from the combined reward. We propose a new curiosity method which uses episodic memory to form the novelty bonus. To determine the bonus, the current observation is compared with the observations in memory. Crucially, the comparison is done based on how many environment steps it takes to reach the current observation from those in memory - which incorporates rich information about environment dynamics. This allows us to overcome the known "couch-potato" issues of prior work - when the agent finds a way to instantly gratify itself by exploiting actions which lead to hardly predictable consequences. We test 








-----------------------------------------------------------
https://arxiv.org/pdf/1802.04181.pdf
State Representation Learning for Control: An Overview

表格不错

As an example, infants expect inertial objects to follow principles of persistence, continuity, cohesion and solidity before appearance-based elements such as color, texture and perceptual goodness. At the same time, these principles help guide later learnings such as object’ rigidity, softness and liquids properties. 
Later, adults will reconstruct perceptual scenes using internal representations of the objects and their physically relevant properties (mass, elas- ticity, friction, gravity, collision, etc.) [Lake et al., 2016].

2.2
Using prior knowledge to constrain the state space: A last approach is to handle SRL by using specific constraints or prior knowledge about the functioning, dynamics or physics of the world (besides the constraints of forward and inverse models) such as the temporal continuity or the causality principles that generally reflect the interaction of the agent with objects or in the environment [Jonschkowski and Brock, 2015]. Priors are defined as objective or loss functions L, applied on a set of states s1:n (Fig. 5), to be minimized (or maximized) under specific condition c. An example of condition can be enforcing locality or time proximity within the set of states.
Loss = Lprior(s1:n;θφ|c) (5) All these approaches are detailed in Section 3.


2.3 State representation characteristics
Besides the general idea that the state representation has the role of encoding essential information (for a given task) while discarding irrelevant aspects of the original data, let us detail what the characteristics of a good state representation are.
In a reinforcement learning framework, the authors of [Böhmer et al., 2015] defines a good state representation as a representation that is:
• Markovian, i.e. it summarizes all the necessary information to be able to choose an action within the policy, by looking only at the current state.
• Able to represent the true value of the current state well enough for policy improvement.
• Able to generalize the learned value-function to unseen states with similar futures.
• Low dimensional for efficient estimation.

3 Learning objectives
In this section, we review what objectives can be used to learn a relevant state representation. A schema detailing the core elements involved in each model’s loss function was introduced in Fig. 2 – 5, which highlights the main approaches to be described here. This section touches upon machine learning tools used in SRL such as auto-encoders or siamese networks. A more detailed description of these is later addressed in Section 4.
3.1 Reconstructing the observation
3.2 Learning a forward model
3.3 Learning an inverse model
3.4 Using feature adversarial learning
3.5 Exploiting rewards
3.5 Exploiting rewards
3.6 Other objective functions
Slowness Principle
Variability
Proportionality
Repeatability
Dynamic verification
Selectivity
3.7 Using hybrid objectives
4 Building blocks of State Representation Learning
In this section, we cover various implementation aspects relevant to state representation learning and its evaluation. We refer to specific surrogate models, loss function specification tools or strategies that help constraining the information bottleneck and generalizing when learning low- dimensional state representations,
4.1 Learning tools
We first detail a set of models that through an auxiliary objective function, help learning a state representation. One or several of these learning tools can be integrated in broader SRL approaches as was previously described.
4.1.1 Auto-encoders (AE)4.1.2 Denoising auto-encoders (DAE)
4.1.3 Variational auto-encoders (VAE)
4.1.4 Siamese networks






State Abstraction as Compression in Apprenticeship Learning        https://github.com/david-abel/rl_info_theory             ref infobot；

State Abstractions for Lifelong Reinforcement Learnin  https://david-abel.github.io/papers/lifelong_sa_icml_18.pdf











A Theory of State Abstraction for Reinforcement Learning
David Abel
Department of Computer Science
Brown University
david_abel@brown.edu
Abstract
Reinforcement learning presents a challenging problem:
agents must generalize experiences, efficiently explore the
world, and learn from feedback that is delayed and often
sparse, all while making use of a limited computational budget. Abstraction is essential to all of these endeavors. Through
abstraction, agents can form concise models of both their surroundings and behavior, supporting effective decision making
in diverse and complex environments. To this end, the goal
of my doctoral research is to characterize the role abstraction plays in reinforcement learning, with a focus on state
abstraction. I offer three desiderata articulating what it means
for a state abstraction to be useful, and introduce classes of
state abstractions that provide a partial path toward satisfying these desiderata. Collectively, I develop theory for state
abstractions that can 1) preserve near-optimal behavior, 2) be
learned and computed efficiently, and 3) can lower the time
or data needed to make effective decisions. I close by discussing extensions of these results to an information theoretic
paradigm of abstraction, and an extension to hierarchical abstraction that enjoys the same desirable properties.








-------------------------------------------------------------------
Learning Actionable Representations from Visual Observations  https://sites.google.com/view/actionablerepresentations

In this work we explore a new approach for robots to teach themselves about the world simply by observing it. In particular we investigate the effectiveness of learning task-agnostic representations for continuous control tasks. We extend Time-Contrastive Networks (TCN)

 using only the learned embeddings as input
 


The contributions of this paper are:
• Introducing a multi-frame variant of TCN which works better for both static and motion attributes classification.
• Showing RL policies can be learned from pixels using mfTCN while outperforming pixel-based ones learned
from scratch or using PVEs.
• We also show that the learned policies are competitive
with true (proprioceptive) state based policies. Hence, we refer to our representations as actionable as they not only encode both static and motion information present in the proprioceptive state but can also be used directly for continuous control tasks.
IV. EXPERIMENTS
A. Regression to velocities and positions of Cartpole
B. Policy learned on TCN embedding from Self-Observation
C. Policy learned on TCN embedding from Observing Other Agents

D. Classification results on Pouring dataset
In the original dataset, frames are manually labeled with answers to questions that should be pertinent to the task of pouring. The original dataset had the following questions and answers:
1) Is the hand in contact with the container? (yes or no)
2) Is the container within pouring distance of the recipi-
ent? (yes or no)
3) What is the tilt angle of the pouring container? (90,
45, 0 and -45 degrees)
4) Is the liquid flowing? (yes or no)
5) Does the recipient contain liquid? (yes or no)
These questions, though relevant to the task, are restricted to information contained only in a single-frame. We augment the set of questions with motion based questions that are also relevant to pouring but will typically require more than 1 frame to answer. The motion based questions are all binary questions (yes or no):
1) Is the hand reaching towards the container? 2) Is the hand receding from the container?
3) Is the container in contact with the hand going up? 4) Is the container in contact with the hand coming down?