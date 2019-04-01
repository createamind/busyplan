

STCN：

3 like lvae  https://github.com/ermongroup/Variational-Ladder-Autoencoder--Learning Hierarchical Features from Generative Models；











Never ForgetNever Forget: Balancing Exploration and Exploitation
via Learning Optical Flow
Hsuan-Kung Yang * 1 Po-Han Chiang * 1 Kuan-Wei Ho 1 Min-Fong Hong 1 Chun-Yi Lee 1
Abstract
Exploration bonus derived from the novelty of the
states in an environment has become a popular
approach to motivate exploration for deep reinforcement learning agents in the past few years.
Recent methods such as curiosity-driven exploration usually estimate the novelty of new observations by the prediction errors of their system
dynamics models. Due to the capacity limitation
of the models and difficulty of performing nextframe prediction, however, these methods typically fail to balance between exploration and exploitation in high-dimensional observation tasks,
resulting in the agents forgetting the visited paths
and exploring those states repeatedly. Such inefficient exploration behavior causes significant
performance drops, especially in large environments with sparse reward signals. In this paper,
we propose to introduce the concept of optical
flow estimation from the field of computer vision
to deal with the above issue. We propose to employ optical flow estimation errors to examine the
novelty of new observations, such that agents are
able to memorize and understand the visited states
in a more comprehensive fashion. We compare
our method against the previous approaches in
a number of experimental experiments. Our results indicate that the proposed method appears
to deliver superior and long-lasting performance
than the previous methods. We further provide
a set of comprehensive ablative analysis of the
proposed method, and investigate the impact of
optical flow estimation on the learning curves of
the DRL agents.







这篇不一定好
UNSUPERVISED DISCOVERY OF
PARTS, STRUCTURE, AND DYNAMICS
Zhenjia Xu∗
MIT CSAIL, Shanghai Jiao Tong University
Zhijian Liu∗
MIT CSAIL
Chen Sun
Google Research
Kevin Murphy
Google Research
William T. Freeman
MIT CSAIL, Google Research
Joshua B. Tenenbaum
MIT CSAIL
Jiajun Wu
MIT CSAIL
ABSTRACT
Humans easily recognize object parts and their hierarchical structure by watching
how they move; they can then predict how each part moves in the future. In this
paper, we propose a novel formulation that simultaneously learns a hierarchical,
disentangled object representation and a dynamics model for object parts from
unlabeled videos. Our Parts, Structure, and Dynamics (PSD) model learns to,
first, recognize the object parts via a layered image representation; second, predict
hierarchy via a structural descriptor that composes low-level concepts into a
hierarchical structure; and third, model the system dynamics by predicting the
future. Experiments on multiple real and synthetic datasets demonstrate that our
PSD model works well on all three tasks: segmenting object parts, building their
hierarchical structure, and capturing their motion distributions








