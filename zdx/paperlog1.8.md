




公众号未深入看的
预览 整理




Causal Confusion in Imitation Learning






PVEs: Position-Velocity Encoders for Unsupervised Learning of Structured State Representations






---------------------------------------------------------------

UPN  https://arxiv.org/pdf/1804.00645.pdf




SE3-Pose-Nets: Structured Deep Dynamics Models for Visuomotor Planning and Control






--------------
SFV: Reinforcement Learning of Physical Skills from Videos





Improvements on Hindsight Learning https://arxiv.org/abs/1809.06719




Hindsight policy gradients https://arxiv.org/abs/1711.06006




The Implicit Preference Information in an Initial State  https://openreview.net/forum?id=rkevMnRqYQ

Rohin Shah, Dmitrii Krasheninnikov, Jordan Alexander, Pieter Abbeel, Anca Dragan
28 Sep 2018 (modified: 21 Dec 2018)ICLR 2019 Conference Blind SubmissionReaders:  EveryoneShow BibtexShow Revisions
Abstract: Reinforcement learning (RL) agents optimize only the features specified in a reward function and are indifferent to anything left out inadvertently. This means that we must not only specify what to do, but also the much larger space of what not to do. It is easy to forget these preferences, since these preferences are already satisfied in our environment. This motivates our key insight: when a robot is deployed in an environment that humans act in, the state of the environment is already optimized for what humans want. We can therefore use this implicit preference information from the state to fill in the blanks. We develop an algorithm based on Maximum Causal Entropy IRL and use it to evaluate the idea in a suite of proof-of-concept environments designed to show its properties. We find that information from the initial state can be used to infer both side effects that should be avoided as well as preferences for how the environment should be organized.
Keywords: Preference learning, Inverse reinforcement learning, Inverse optimal stochastic control, Maximum entropy reinforcement learning, Apprenticeship learning
TL;DR: When a robot is deployed in an environment that humans have been acting in, the state of the environment is already optimized for what humans want, and we can use this to infer human preferences.

Maximum Causal Entropy IRL





VMAV-C: A Deep Attention-based Reinforcement Learning Algorithm for Model-based Control




Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow






-----------------------------------------------------------
https://arxiv.org/pdf/1802.04181.pdf
State Representation Learning for Control: An Overview

表格不错
















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