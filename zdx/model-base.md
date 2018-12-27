

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





-----------

Variational Option Discovery Algorithms

We show that Variational Intrinsic Control (VIC) (Gregor et al. [2016]) and
the recently-proposed Diversity is All You Need (DIAYN) (Eysenbach et al. [2018]) are specific
instances of this template which decode from states instead of complete trajectories.


---------------------
RND：


2.2.2 RELATION TO UNCERTAINTY QUANTIFICATION

If we specialize the regression targets yi to be zero, then the optimization problem arg minθ E(xi,yi)∼D∥fθ(xi) + fθ∗ (xi)∥2 is equivalent to distilling a randomly drawn function from the prior. Seen from this perspective, each coordinate of the output of the predictor and target net- works would correspond to a member of an ensemble (with parameter sharing amongst the ensemble), and the MSE would be an estimate of the predictive variance of the ensemble (assuming the ensemble is unbiased). In other words the distillation error could be seen as a quantification of uncertainty in predicting the constant zero function.


4


Other methods of exploration include adversarial self-play (Sukhbaatar et al., 2018), maximizing empowerment (Gregor et al., 2017), parameter noise (Plappert et al., 2017; Fortunato et al., 2017), identifying diverse policies (Eysenbach et al., 2018; Achiam et al., 2018), and using ensembles of value functions (Osband et al., 2018; 2016; Chen et al., 2017).


