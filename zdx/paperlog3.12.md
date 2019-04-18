

STCN：

3 like lvae  https://github.com/ermongroup/Variational-Ladder-Autoencoder--Learning Hierarchical Features from Generative Models；






Split-brain autoencoders: Unsupervised learning by cross-channel prediction
Unsupervised learning of depth and ego-motion from video
Unsupervised video object segmentation with motion-based bilateral networks
Unsupervised Online Video Object Segmentation with Motion Property Understanding
Video Object Segmentation using Space-Time Memory Networks

FEELVOS: Fast End-to-End Embedding Learning for Video Object Segmentation
Paul Voigtlaender, Yuning Chai, Florian Schroff, Hartwig Adam, Bastian Leibe, Liang-Chieh Chen
(Submitted on 25 Feb 2019 (v1), last revised 8 Apr 2019 (this version, v2))
Many of the recent successful methods for video object segmentation (VOS) are overly complicated, heavily rely on fine-tuning on the first frame, and/or are slow, and are hence of limited practical use. In this work, we propose FEELVOS as a simple and fast method which does not rely on fine-tuning. In order to segment a video, for each frame FEELVOS uses a semantic pixel-wise embedding together with a global and a local matching mechanism to transfer information from the first frame and from the previous frame of the video to the current frame. In contrast to previous work, our embedding is only used as an internal guidance of a convolutional network. Our novel dynamic segmentation head allows us to train the network, including the embedding, end-to-end for the multiple object segmentation task with a cross entropy loss. We achieve a new state of the art in video object segmentation without fine-tuning with a J&F measure of 71.5% on the DAVIS 2017 validation set. We make our code and models available at this https URL.




https://github.com/wenguanwang/AGS 无监督代码 Learning Unsupervised Video Object Segmentation through Visual Attention



Multigrid Predictive Filter Flow for Unsupervised Learning on Videos
Shu Kong, Charless Fowlkes
(Submitted on 2 Apr 2019)
We introduce multigrid Predictive Filter Flow (mgPFF), a framework for unsupervised learning on videos. The mgPFF takes as input a pair of frames and outputs per-pixel filters to warp one frame to the other. Compared to optical flow used for warping frames, mgPFF is more powerful in modeling sub-pixel movement and dealing with corruption (e.g., motion blur). We develop a multigrid coarse-to-fine modeling strategy that avoids the requirement of learning large filters to capture large displacement. This allows us to train an extremely compact model (4.6MB) which operates in a progressive way over multiple resolutions with shared weights. We train mgPFF on unsupervised, free-form videos and show that mgPFF is able to not only estimate long-range flow for frame reconstruction and detect video shot transitions, but also readily amendable for video object segmentation and pose tracking, where it substantially outperforms the published state-of-the-art without bells and whistles. Moreover, owing to mgPFF's nature of per-pixel filter prediction, we have the unique opportunity to visualize how each pixel is evolving during solving these tasks, thus gaining better interpretability.

Multigrid Predictive Filter Flow for Unsupervised Learning on Videos
Shu Kong, Charless Fowlkes
Dept. of Computer Science, University of California, Irvine
{skong2, fowlkes}@ics.uci.edu
[Project Page], [Github], [Demo], [Slides], [Poster]

https://github.com/aimerykong/predictive-filter-flow

image：
Image Reconstruction with Predictive Filter Flow


Shu Kong, Charless Fowlkes
Dept. of Computer Science, University of California, Irvine
{skong2, fowlkes}@ics.uci.edu
[Project Page], [Github], [Slides], [Poster]
Abstract
We propose a simple, interpretable framework for solving a wide range of image reconstruction problems such as
denoising and deconvolution. Given a corrupted input image, the model synthesizes a spatially varying linear filter
which, when applied to the input image, reconstructs the
desired output. The model parameters are learned using
supervised or self-supervised training. We test this model
on three tasks: non-uniform motion blur removal, lossycompression artifact reduction and single image super resolution. We demonstrate that our model substantially outperforms state-of-the-art methods on all these tasks and is
significantly faster than optimization-based approaches to
deconvolution. Unlike models that directly predict output
pixel values, the predicted filter flow is controllable and interpretable, which we demonstrate by visualizing the space
of predicted filters for different tasks.1

Using a CNN to directly predict a
well regularized solution is orders of magnitude faster than
expensive iterative optimization.

During training, we use a loss defined
over the transformed image (rather than the predicted flow).
This is closely related to so-called self-supervised techniques that learn to predict optical flow and depth from unlabeled video data

Predictive filter flow differs from other CNNbased approaches in this regard since the intermediate filter
flows are interpretable and transparent [50, 12, 32], providing an explicit description of how the input is transformed
into output. 


To summarize our contribution: (1) we propose a novel,end-to-end trainable, learning framework for solving various low-level image reconstruction tasks; (2) we show this
framework is highly interpretable and controllable, enabling
direct post-hoc analysis of how the reconstructed image is
generated from the degraded input; (3) we show experimentally that predictive filter flow outperforms the state-of-theart methods remarkably on the three different tasks, nonuniform motion blur removal, compression artifact reduction and single image super-resolution.

Filter locality In principle, each pixel output I2 in Eq. 3 can depend on all input pixels I2. We introduce the struc- tural constraint that each output pixel only depends on a corresponding local neighborhood of the input. The size of this neighborhood is thus a hyper-parameter of the model. We note that while the predicted filter flow Tˆ acts locally, the estimation of the correct local flow within a patch can depend on global context captured by large receptive fields in the predictor fw (·).
















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








