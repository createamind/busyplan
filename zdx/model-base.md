

MODEL-ENSEMBLE TRUST-REGION POLICY OPTIMIZATION



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
 





















---------------------
RND：


2.2.2 RELATION TO UNCERTAINTY QUANTIFICATION

If we specialize the regression targets yi to be zero, then the optimization problem arg minθ E(xi,yi)∼D∥fθ(xi) + fθ∗ (xi)∥2 is equivalent to distilling a randomly drawn function from the prior. Seen from this perspective, each coordinate of the output of the predictor and target net- works would correspond to a member of an ensemble (with parameter sharing amongst the ensemble), and the MSE would be an estimate of the predictive variance of the ensemble (assuming the ensemble is unbiased). In other words the distillation error could be seen as a quantification of uncertainty in predicting the constant zero function.


4


Other methods of exploration include adversarial self-play (Sukhbaatar et al., 2018), maximizing empowerment (Gregor et al., 2017), parameter noise (Plappert et al., 2017; Fortunato et al., 2017), identifying diverse policies (Eysenbach et al., 2018; Achiam et al., 2018), and using ensembles of value functions (Osband et al., 2018; 2016; Chen et al., 2017).


