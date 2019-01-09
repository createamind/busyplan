 人
-----------------------------
Plan   正式的：  日记 反思； 最重要！  明确的！具体！！！具体！！！

梳理一个实现AGI的简单框架

互信息 熵 信息熵；

更好智能的各个维度： 下面所列方法均有开源代码。  思想思路：

0 《人工智能的未来》（On Intelligence）一书，是由杰夫•霍金斯介绍了大脑的智能属性之一是预测， prediciton（预测的各个角度：4d时空预测，DFP多传感器相互的属性等信息预测-6表示方法，
发展progressive着提高预测的精度，cGAN各种条件状态情况下的预测）；监督学习的标签预测，特定环境的特定行动的特定结果的一致性cGAN， 
Curiosity-driven Exploration by Self-supervised Prediction http://mp.weixin.qq.com/s/A3pYqzakPCYn68wcQfDtCQ AGI:我与世界的互动是不是如我所愿。动作条件CycleGAN好奇心探索，
熟悉环境的预训练。https://github.com/pathak22/noreward-rl

1更多的维度-空间(2d到3d)加时间监督 videogan；
1我们生活在四维的时空中，真正的智能必须感知这四个维度，3维空间和时间，所有接受的信息应该是视频形式，视频是非常好的输入素材。只通过图片训练的智能有其先天缺陷。
基于四维时空的预测学习或记忆学习。所以我们依据做了基于视频的预测。
单张图片歧义-通过视频进行歧义消除：slam14讲2.1图片 小人是否是模型 是否远近； 图片是一个拖把-视频看其实是一个多毛的小狗。 飞行的鸟（单张图片就是混乱的线条，视频就是一个运动的鸟。）

视觉的运动感知在视觉的认知发展是很早起就开始发育的-公众号菜单生物智能相关文章。

+BI ?


2更多传感器，视听触等, 压力的感知和 reward 动作的感知学习。   宽泛讲就是多维度的信息，比如虚拟环境各种指标信息--DFP，不同传感器互相监督即cGAN；
2视频的信息从像素出发维度太高，而且视频中很多内容是完全不同的，区分视频中的不同物体应该不仅仅是从视觉区分（使用的时候是从视觉来区分，但是训练的时候是不同传感器直接区分的）
视频中不同物体的本质是物体的不同属性的区分，通过视觉联想其他感知信息的内容来区分。从视觉中区分出不同远近，不同软硬的物体，不同用途的物体，不同冷热危险的物体。现在只做了深度的学习，其他属性还未开始训练学习。
生成模型的人脸比如头发的密度温柔属性和脸的肉体属性温度是很不同的。
小数据-大任务--人工智能的架构与统一。大任务---多感知不同属性的多任务关联映射。

多属性和注意力的关系：注意力就是讲有明显特点属性的物体快速提取进行更细致的认识，有一个粗略的网络进行比如危险属性的识别，然后有专门网络进行特定属性的特定处理，比如跟踪，准确的识别认识，或逃跑。

2.1 强化学习学习disentangeling factor，通过动作互动，动作也是感知方式之一， 二维平面学习图片中的disentangleing feature先天缺陷，深度学习，及不同触觉信息都进行了多维度的属性拆解。
betavae等仅仅通过视觉已经非常厉害了，扩展方式应该是多属性及动作对属性的探测。

2.2 动态跟踪物体的移动；深度学习跟踪已经有不错的项目。

2.3 选取动作及视野就是注意力，好奇心，就是关注特定数据。
选准特定视野，选择特定的目标，选择特定的。。。。  底层信息的竞争机制。

2.4 disentangling 学到的z的变化规律记忆下来，z在训练时候也会被覆盖掉的。  学习要适可而止，
2.4.1  加动作或位置信息等一起多传感器进行disentangling的学习。   道路先学习转换到俯视图，然后学习俯视图的disentangling。

2.5 记忆的meoey，记忆规律disentangling的z；记忆某个强化学习的技能，记忆。。。。  强化学习的经验如何自动选择记忆，记忆提取，小范围的自动disentangling的规律自动选择记忆，记忆提取机制？  这个临时记忆通过反复重复播放训练形成长期记忆。


3 Progressive Growing of GANs 智力的成长发展，2分类或3分类进行简单功能的精准实现  --7-- 上下左右，前后远近 先离散后回归。
3学习开始可以从简单开始，就像progressive growing gans 图像越来越清晰，分类或其他功能的准确度也可以越来越精确，先从2分类或3分类开始训练即可。
将特征空间压缩到非常低的维度，比如高低，胖瘦，大小，快慢，前后，左右，上下，



4 conditon gan；条件生成 外界环境的特定条件下的特定反应。  最新的cGAN 论文。https://github.com/pfnet-research/sngan_projection
condition：不同时间，不同空间，不同条件，不同的传感器属性，不同物体属性。
cGAN 各种不同的条件就是标签，条件监督可以用在传感器监督，标签监督，更细致的监督可以参考pix2pix-vid2vid-两幅图像大小一致的情况, vectro2vector 

生物各种不同情况的反应就是各种condition GAN的 动作输出。语言的很多限定词都是conditon，无人驾驶的红绿灯，各种异常情况都是各种conditonGAN。
因果推理和cGAN的预测也是相关的。

5 人脑发育到一定阶段神经元增长不多，增加的是神经元的链接，通过不同区域神经网络的链接的增加，实现了不同的关系属性及概念的互相链接。这个可以对应进行神经网络的highway的不断增加。
5 more and more highway for 概念关联。 

densenet！！入职培训提到的densenet多层连接。  3d unet的一半

容量

6capsnet 胶囊 (Capsule) 是一个包含多个神经元的载体，每个神经元表示了图像中出现的特定实体的各种属性 多属性表示。 



下面是相对重复的一些关键点。
7极低的特征空间维度  beta-vae distangle:bayesgan   https://github.com/ermongroup/Variational-Ladder-Autoencoder； 这些都是图像。
动作也可以很低，上下左右，前后远近，   ref 2.1

8  cnn filter  reinforcement learning 选择特定filter进行处理。  or densenet？ 3d unet的一半
9 DFP   Human certainly possess the innate ability to switch goals based on different circumstances  -----联系 4 
imitation carla； Curiosity-driven

10 vid2vid prediction  https://github.com/createamind/vid2vid  ;   Curiosity-driven
序列预测是智能非常重要的能力，对于AI非常重要，完全符合公司目标通用智能，做出了能增强现有神经网络的智能。
具体场景：大家一起想！避障，其他车辆意图的预测，torcs游戏验证？机器人自己动作的预测，常识学习。条件反射，躲避飞来的石头，场景记忆，右转注意行人等，
原型验证ok，完善中再继续找应用的场景和产品的具体完善。
无人驾驶决策是快决策、条件反射的决策，不是高层推理的决策，所以是感知和决策和执行几乎是一体的。保守的异常情况刹车处理。


----------------------------------------------------------------------

EMI ：  互信息的embedding  forward model; backwordk model; model-base in mutual information;  替换之前的vae 模型 扩展到 mode base；and chap5 reward ；  code？好奇心代码？？改造？？
empower 的 code；  https://github.com/navneet-nmk/pytorch-rl  and tf version    动作跟stat的互信息；  https://navneet-nmk.github.io/2018-08-26-empowerment/

UNSUPERVISED CONTROL THROUGH NON-PARAMETRIC DISCRIMINATIVE REWARDS

概念学习 能量函数。Concept Learning with Energy-Based Models

SAC-----
Plan Online, Learn Offline: Efficient Learning and Exploration via Model-Based Control   
https://bair.berkeley.edu/blog/2018/11/30/visual-rl/


智能 framework and key paper  code;
--------------------------
0 slow  deeper; go most deeper;

1
bayes；certain uncertain；   polo探索部分引用了prior ref by rnd and rnd have openai code。好奇心：；RND code
exploration  curiosity  paper:polo 三个应用说明背后的方法是一样的通用的。 mpc，global value； explorer

反馈，信息的确认？bayes uncertain减少？互信息？

2表示学习； 概念学习  gan-qp
2.1 生成模型做分类，先能生成再进行分类。
vdb
互信息用在编码控制上面，有编码的都可以进行控制？？应用vdb的bottleneck进行信息控制 编码控制？+  EMI ？

3progressive grow；  pg
环境的reward 需要动态变化，不同阶段学习不同的重点能力，比如先站，再跑，在目标跑，再避障，再。。。。。

4 4d space time   densenet-tc A SIMPLE NEURAL ATTENTIVE META-LEARNER         ； BI

5memory： data reuse；  her

6control；   


7熵-互信息； 信息熵 最大 or 最小
互信息最大   empower EMI  动作控制影响 互信息最大。   ++ vdb？？
互信息最小  vdb   信息瓶颈？？  y z; x z  决策信息最小最关键，信息瓶颈 最大动作相关信息？？

红绿灯时一种cnoditon的的动作，比如  论文 deepmind 互信息压缩很多技能在一个网络里面。
stat 决定的action；   stat + goal的  conditon 动作；   

goal conditon 的  model base ； 
goal 是车道线  是红绿灯  是 其他异常情况，其他行人 其他的 车 其他的自行车灯
需要对车道线  红绿灯进行概念的学习；
概念学习------------  概念是一个conditon的goal key；

or  模仿学习； inverse rl  不学习概念，示范来做。后面升级到概念的语言沟通来做。


最后开个脑洞：人脑对于少样本训练的范化误差是远比机器学习模型的效果要好的，那么对于任何一个新概念 Y，其在各层抽象级的表述分别为，人脑必然有非常高效的计算机制，对于所有之前已经学习到的概念，最大程度利用已有的知识。同时对于同一抽象级的不同概念，尽量让它们描述不同的信息，即减小，很可能频繁用到离散化的技巧来实现互信息压缩。 




8model base planning-- pred  predictonve ，mpc

9multimodel: cycle(video audio language )  cycle-sensor-motor     gan-qp 苏剑林
Multimodal Densenet  https://arxiv.org/pdf/1811.07407.pdf    3d unet的一半
cnn的特征使用的改进！
9.1 2.1 darla model base ; world model; darla vae densenet;
densenet的vae！！   
unet的vae；
unet的video prediction；

10 
语言： 交流沟通 通信编码
动作描述---动作的语言之前的模型语言模型文章语言论文互信息 IB；Efficient human-like semantic representations via the Information Bottleneck principle

Unsupervised perceptual rewards for imitation learning .   gan .  vae .  自动学习视频的分割--------苏剑林 自动的vae 聚类   3dvae聚类，

Mutual alignment transfer learning， and ref it paper；  自动分段  ； Ddco: Discovery of deep continuous options for robot learning from demonstrations

TACO: Learning Task Decomposition via Temporal Alignment for Control

Variational Option Discovery Algorithms -- real hierarchial --- 非常重要的预训练
每个skill的完整性；skill的完整确认。  diversity is all you need .

novel uncertain  ----- bayes prior function。
模仿的基础是自己会，自己有基础，类似这样先学习基础后 再 demo imitation；



11  学习方式；自己学  学别人
手把手教---有动作的直接学习动作的模仿学习
只看就学 ---  无动作的模仿学习或     强化学习：
vdb 互信息
 imitation  meta-learning。 https://zhuanlan.zhihu.com/p/33789604
模仿学习是从视觉推断动作的学习，而好奇心探索就是动作产生的环境变化的互动开始学习，好奇心是可以作为模仿学习的预训练的。

oneshot--学习通用的架构结构-可以快速泛化。

https://sites.google.com/view/one-shot-imitation  code https://github.com/tianheyu927/mil    SFV: Reinforcement Learning of Physical Skills from Videos

Deep Meta-Learning: Learning to Learn in the Concept Space

Inverse reinforcement learning for video games   https://github.com/HumanCompatibleAI/atari-irl   加上vdb互信息的约束试试  +EMI + empower

Automata Guided Reinforcement Learning With Demonstrations， HER


A Simple Neural Attentive Meta-Learner


Model-Based Reinforcement Learning via Meta-Policy Optimization+ 实时反馈 -- mpc？？

mb-mpo +. sac 是不是会很厉害？mode base+free；


12 env： carla ； real world


13   小数据  大任务； 多任务；   meta learning 是不是就是多loss，的确是，不过是一类loss； 人 ，跟基础的原则道理是跨类的任务loss。
maml meta learning 跟表示学习的联系。  meta learning 多任务跟多任务互信息压缩到一个网络的关系？ tc softattention ref bottleneck mutual info;
：：：：：多任务 多loss ？？ what loss：  
13.1
13.2
.3
.4
.5
各种loss 列举出来！！！！ 数据需要什么样的？？？？




多任务loss：meta learning；



多传感器loss：


空间loss：

https://github.com/tensorflow/models/tree/master/research/vid2depth depth loss
https://github.com/tensorflow/models/tree/master/research/struct2depth

uncertain vae loss
角度 位置  gqn loss
4d segmentation  loss

4D空间 关系  重建关系位置 

双目视察 loss
3d conv 运动信息 loss



tcn2  actionable loss？
https://sites.google.com/view/actionablerepresentations
actionable2 paper  LEARNING ACTIONABLE REPRESENTATIONS WITH GOAL-CONDITIONED POLICIES


13.0 多算法，多方法方法训练。


14


15 transfer learning  与多任务 meta learning 的关系。 sim to real



16 模型的容量最够大。



actionable
https://github.com/VinF/deer
mine
DIM
cpc
cpc-action
norl near 最优表示的强化学习






欢迎大家批评留言交流，提出不同的智能点的各种方法的实现思路想法，

持续完善阅读原文  https://github.com/createamind/busyplan/blob/master/zdx/Plan-thinkout.md




code: 
cGAN https://github.com/pfnet-research/sngan_projection,
Curiosity https://github.com/pathak22/noreward-rl,
vid2vid，
carla imitation  https://github.com/mvpcom/carlaILTrainer



架构：
carla + Curiosity A3c + 数据存储replay memory（神经网络压缩图像）+ 后台的rgb2depth + beta-vae + 多属性。



模仿开源：
https://sites.google.com/site/imitationfromobservation/
https://sermanet.github.io/imitation/  tcn

https://github.com/tianheyu927/mil
https://sites.google.com/view/daml  数据泛化！ 只有视频和








--------------------------------------------------------------------------------------


结合强化学习的注意力的任务相关特征抽取的独立网络

多个生成模型的生成效果叠加为一个完整视频效果，还是独立的一个大网络一起训练？ 自动分化出子网络




混合pix2pix vid2vid 视频预测和cyclegan 视频预测-------------短期记忆！！！

视频vid2vid action reward ?Direct Future Prediction rl? 声音到图像？ time contrast 跟 vid2vid 的关系？？

Time Contrastive Networks    https://sites.google.com/site/imitationfromobservation/
内在的多种任务-视觉求索朱松纯

memory：视频重放，深度学习对图片压缩的相关谷歌工作，哈希也是智能的一部分的文章。存储z？多感知维度--倒水后杯子轻-重力感知-声音感知-
action condition video pred！



cardemo-autoware；   vide GAN; gan车道线，pix2pix;  深度到seg；3d；
5年  1年  1月 1周 今天  现在  今天最重要的目标计划：？？ 
两个推进思路： 1  自动驾驶真车使用的需求   2 通用智能的构建思路推进。  videogan+vae
机械臂？？？




车道线  opencv识别用pix2pix生成普通图片到车道线道路概念的映射训练，车道方向的

预测车的方向后，自己何时转方向盘是另外一个问题。-- videogan就可以学习何时转向！！ 动作和图片的pix2pix？？
继续训练压缩z的每个维度到一个语义含义，
1  特征对应的向量会有
2  特征对应的向量是否为独立的维度？？

beta-vae  avb  

videogan学习到特征但是是否每个隐变量是独立特征语义？

------------------------------------------



---------------------------------------------------

1 结构： 无监督分类网络对输入视频场景人物等的无监督分类。 分类后的每一类进来后分布输入不同的类比的训练网络进行训练。

3d空间结构： 背景，前景（内容-运动（外部运动-自身我运动））  分阶段训练不同的方面重点



-----------------------------
即时思考记录

22 z就是结论本身；生成是认知，机器学习是预测，生成就是可视化内部信息，就是白盒分析，z是决策的认知元素。

10.20  mocogan的人脸动作；   dcgan用视频的序列图片训练会如何？？   AVB视频序列图片训练？？
https://arxiv.org/abs/1706.08033 from mocogan ref to this to encode vidoe, 视频输入的分解生成latent  var； 


10.15  16
深度：室外，室内； 双目视差、雷达、物体的摆放前后逻辑-视线阻挡。
cyclegan视频，单帧or 连续 ； 深度以外的cyclegan学习。  
cyclegan是成对的学习，或成对的数据集，pix2pix是成对的数据；能否不用成对数据集，直接从数据集中学习各种属性，遇到其他数据集继续类似cyclegan学习对应数据集的对应属性变化？？
或图灵机自己选择某些数据组成成对的数据集。


语言到图像：

pix2pix  到 sound2video  （have text to image 这个的相关介绍 ） 
Ambient Sound Provides Supervision for Visual Learning  https://arxiv.org/pdf/1608.07017.pdf

深度到rgb： Self-Paced Cross-Modality Transfer Learning for Efficient Road Segmentation. In ICRA2017


video 视频的context信息，声音和图像的对应关系也有。 ego-motion   3D重建？

通用智能的构建思路： 1 运动（参考生物视觉发展），  videgan(avb?)     1.1 我不动，周围运动，1.2自我运动的检测。   



如何表示一个物体的很多属性？？？不同传感器的属性，不同的硬度，导热，高低，亮度等等。
liyanran160pptGAN,domain transfer:吧隐变量的几个结构进行了更改，其他属性或维度保持不变。





test只用视觉，train 多传感器训练
激光雷达价格下降的速度，只用视觉的效率。雷达实时深度探测到效率。
实际在线训练，雷达做备份-在rgb深度信息预测准确度降低到某个值时候启用实时雷达探测及收集数据训练新场景。

































--------------------------------------------------
智能架构思路：

方向：
1 哪些需要我们自己去突破（我们有积累和思考，有自己的思路方向）:  更好的模型，cyclegan；
2 哪些是我们去跟进即可（别人比我们更专业更有资源和实力），比如apollo，比如硬件改装车，比如px2
精力注意力首先放到要突破的工作去。



todo： 
招人：之前跑的成果展示

指导思想：集成已有算法代码而非自己研究算法代码。
集成：整个系统的集成：各个子功能的实现。

无监督技术调研：

模型：
1  pix2pix应用到多传感器、多场景、每个单词都是一个conditon
之前的pix2pix 文章：
最近比较火的三个GAN应用及代码--Pix2pix
论文 pix2pix  最近比较火应用背后的技术思想
cyclegan、discogan
CycleGAN: pix2pix without input-output pairs
Discover Cross-Domain Relations with GAN



多传感器：
为什么从视觉里我们看到石头和苹果就能体会到他们的不同的硬度和口味？
看到小狗和麻雀为什么会有不同的联想。铁丝和粉丝感觉更不同。
视觉看到和我们体会物体的属性是有关联的，那么如何进行这种关联呢？
如何训练能联想的模型。
segmentation的本质是不同物体的有核心不同的属性，这些不同的属性从哪里来？从其他传感器维度来，视觉图像和这些属性建立了关联、联觉、联想最终我们通过视觉就能进行这种属性的区分
pix2 pix   多传感器 互为 不同方面进行互相生成。视觉对热感受的识别；通感训练。通感视觉对其他属性的感知，
视觉和其他很多物体的属性都有关联：视觉与听觉，视觉与触觉（软硬、凉热等属性）嗅觉（味道属性等）
及不同属性所需要的不同的注意力：活动的蛇与静止的蛇。
部分文字就是对这种属性的标注link等调用。


多条件condition：
pix2pix 应用到自身运动估计，摄像头移动或静止。
condition：我走，环境是这样变化，condition：我静止，环境其他物体的运动应该是怎么样的。学习背景与物体（videogan 周末轻松一刻，欣赏完全由程序自己回忆的视频片段）。


摄像头，深度双目摄像头，雷达，能不能学出来独立物体？？从不同的深度区分，不同的属性区分（雷达测试密度）
视觉和雷达点云深度信息的互相学习。视觉从雷达探测的物体属性进行学习。
学习不同场景、不同物体、不同大小、不同特点、不同物体展示的深度信息。


视觉与深度、雷达  rgbd； 与imu；移动速度的估计，与红外，热量、声呐；等等其他传感器。



测试：任务分解细化：


这个和sfm-net有类似地方，sfm是手动设置loss函数。
以上全部在一个大网络里面训练，或如何自动拆分为小网络，或网络如何自动扩展、扩容。



-1  betavae 单特征连续变化学习

通用智能架构：

11  运动的监督，视频的监督，时间的监督-------就是视频里的学习

2  光流论文代码(监督训练)和sfm-net论文（无监督）

1 光流检测运动   ： 1 自动标注的作用  https://github.com/msracver/Deep-Feature-Flow
光流对segmentation分割的学习可以结合fasterrcnn对区域合并的网络训练。
视频和光流对的pix2pix。
视频和segmentation的pix2pix
视频里运动物体的预测感知。


2 sfm
最近的无监督学习： sfm learning   自我运动学习

Unsupervised Learning of Depth and Ego-Motion from Video  及代码
sfmlearning   https://github.com/tinghuiz/SfMLearner
https://github.com/mrharicot/monodepth  场景深度学习


TCN: 端到端机器人自学习算法
3 time contrast 
Time-Contrastive Networks: Self-Supervised Learning from Multi-View Observation


光流做分割的数据准备—-论文 光流视频 一篇论文！！https://github.com/msracver/Deep-Feature-Flow

slam—pix2pix 生成卫星图的道路位置！  细节信息如何用生成模型来预测！！？？

4 ros 多机器，小车采集数据到服务器训练，实时或bag文件保存

5 ros传感器类型数据： 视频，雷达3d，红外、声呐、imu，的数据保存；
5.2开放数据的使用进行 pix2pix训练！包括kitti，滴滴、百度，udacity等数据的训练。

6 增强学习

7 图灵机模型   最新deepmind时间生成


自动驾驶技术：305篇论文调研；

视觉论文总结的文章；视觉的大厚书  
生物视觉了解：
论文解读：主视觉大脑皮层的深度层级模型：机器视觉可以从中学到些什么？

数据集：kittiudaciytbaidu  awesome

参考自动驾驶第一本书---相关的12篇文章


版本升级测试：应该是智能的测试，类似考题了，语言交互进行测试（1 语言学习及语言接口）。






--------------------------------------------------
语义学习思路：  公众号是最新的

​

一：概述
二：通俗介绍
三：技术相关模块资源



一：概述：

《思想本质》一书通过语言和认知对人的思想认知等进行了分析，前部分有一个核心观点是(李德毅院士也提过)：语言是思想认知体系的语义索引，语言只是符号，语言表达的含义即语言背后的认知体系是决定语言的根本。任何一个具体的词语，比如桌子，‘桌子’是两个字，可以翻译为其他语言，但是桌子这个概念代表了一种家庭工具，通常是木制的用来放物品的一个工具，这个工具是看得到，有总量，有形状的一个具体物体，这个桌子概念代表了一种具体的认知，‘桌子’这个词本身只是一个词，背后是一个认知概念体系。


语言是语义索引，背后有一个很大的认知体系为基础，认知基础之一来自视觉对现实世界的认识，物理规律，场景的学习理解，跨越长时间的事件的记忆理解及社会互动理解等等。但是需要从基础的基本的语义概念开始学习和实现。

人思考以概念为基础单位，每个概念都是一个语义认知的单位，人智慧高于其他动物就是因为人的抽象思维层次更高，思维活动是在语义认知层面进行思维。所以现在如果做通用人工智能应该从视觉开始，语言其次；首先通过视觉对外部世界达到语义认识，然后在语义认识的基础之上进行思维，进行其他认识活动的构建。

deepmind也有类似思路，见deepmind 做通用人工智能的思路，及理论 | 暑期课程最后一讲：理论神经科学和深度学习理论，且现在的深度网络已经可以学习语义特征，只是技术还非常原始初级。ref   beta-vae  超越 infogan的无监督学习框架--效果更新   ！    使用infogan学习可解释的隐变量特征学习-及代码示例（代码和官方有差异）

上面链接展示的语义认知，还仅限于静态图像，视频中跨时间维度的如动作（运动信息向量的神经网络学习 code、ppt、视频ok）等事件抽象 (还没有做到)。





感知作为外部信息输入的第一步，作用是抽取核心语义特征属性。高层次的逻辑判断基于底层的基本语义单位的认知元素。比如翻译就是对同一事件的不用语言描述，事件是唯一认知，描述可以用不同语言描述，不同语言的描述都是基于同一事件的多个元素在不同语言里对应的概念名称进行描述。
思考运算基于物体的核心属性推理 ref deepmind 做通用人工智能的思路

拿驾驶来说
驾驶的核心是认路，路的概念是核心，路的方向是驾驶的根本。
路的一个属性是路的延伸方向，驾驶的核心既是控制车辆的前进的方向跟路的延伸方向相同即可，方向盘需要和路的方向一致，如果学习到路的方向这个抽象概念，那方向控制就太容易了，路的方向本质是路延伸的指向。如果能吧路抽象成一条线，而且是把各种不同的路（高速路，山路，街道）统一抽象能很好的泛化，方向盘的控制就解决了。


深度学习的动机与挑战之-流形学习 提到道路就是嵌入在三维空间中的一维流形。
GAN对流形的优化提高稳定性：
理论|来聊聊最近很火的WGAN
通俗|令人拍案叫绝的Wasserstein GAN 及代码（WGAN两篇论文的中文详细介绍）


刹车控制的基础是危险的概念，危险的识别和认知就比较复杂了。要认识这个世界的危险，先认识这个世界，认识就是找出事物发展的规律，需要先总结多个事件，至少需要先能描述多个事情，先分门别类的区分多个事件，多个到一个，先能描述一个事件，先重现一个事件，落实到具体技术就是：gan生成视频

认识需要 1 区分不同的属性，是否危险（学习独立事物，独立个体，独立场景，个体的颜色属性，声音属性，触觉属性，气温属性），什么条件能更方便找到食物。什么环境下的什么特征？颜色特征，天气特征（这个太大，天气的其他属性），这些属性会通过drl强化学习关联到reward，出现部分动物迷信的特点。部分动物或鸟类只根据某些颜色等特征进行固定的行为。其实条件反射也类似。更细致的学习这些特征就类似GAN学习到眼镜、笑脸、男性、女性等概念一样。

另外物体跟踪算法可以为物体的识别提供大量样本进行训练，视觉中连续的物体 https://github.com/msracver/Deep-Feature-Flow 


不同场景路对应不同的概念，比如城市中的路概念对应"街道"。街道的概念其实有很多的相关概念：行人、建筑、交通标识、路口等等。为了理解街道的概念需要以上相关概念的学习。当然概念的学习是一个渐进的过程，概念可以分化，细化。

深度学习学习到的概念如果能类似人实时改进学习提高，且不同的概念能及时学习提高，快速适应环境变化，这样才能有更好的适应能力。
考虑概念的层级组合关系，子概念的概念学习可以在不影响架构的情况下，直接提高系统的认知能力。

传统自动驾驶软件如果将各种条件提前做好配置，则失去了不同环境的适应能力。


现存技术障碍：语义学习技术处于非常初级阶段，模型的稳定性，扩展性，训练模型结合其他功能的协调，如和动作控制进行结合，属性的相关操作比如手臂的协调控制指示等，注意力等。

语义学习可以从beta VAE入手  ref 谷歌：beta-vae 可以媲美infogan的无监督学习框架-多图-及代码；

#对deepmind强化学习与通用智能的个人理解；强化学习对好坏利弊的追求适用于任何智能层次的主体；即适用于任何层次的生物，但是类人智能需要在认知上有更高的抽象，所以更好的无监督抽象学习将提高强化学习的效果（比如抽象学习到路的方向）。所以强化学习使用学习到更抽象概念的的GAN模型进行特征提取的辅助能更高效。


二：通俗介绍
为什么视觉重要：
这个链接有丰富的内容：论文解读：主视觉大脑皮层的深度层级模型：机器视觉可以从中学到些什么？

视觉是动物对世界认识极其重要的一个入口，通过视觉，动物对世界进行认识，把世界在自己的大脑中进行建模。
视觉无论从生物的进化，还是从个体生物的发育来看，视觉都是基础，其他的思维很大比例都是构建在视觉基础之上，很多语言都是首先反映普通的视觉对象，抽象词语也大都是从具体的词语抽象而来
智慧的高低是抽象思维的深度，是理解概念的抽象深度，概念最基础的应该是具体的物体的概念，视觉层次的物体形体概念，
人比其他动物更强的是大脑的其他高级功能更多更抽象，视觉神经系统相比其他感觉系统在整个大脑中所占比重也更大。

实现方法的深刻认识：才是改变未来的能力！向大自然已经实现的智能学习：动物的认知神经发育发展过程，这样实现过程才是有一定保证的。

在与世界的互动水平上面反映了地球生物的智能水平，对这个世界认识越深刻那么这个生物越智能。人作为这个世界最高智慧生物，学习人的认知发展不会导致类人智能的研发出现很大的偏差。
想更智能就要对世界有更深入的认识，想真正的认识世界，需要从最底层开始认识这个世界，从最底层开始构建对世界的认知，像一个新生婴儿，甚至更早的小婴儿一样开始认识世界，积累对世界的认识
反观生物对世界认知的发展进化，发育演化，视觉是认识世界的重要基础。

视觉是动物对世界认识极其重要的一个入口，通过视觉，动物对世界进行认识，把世界在自己的大脑中进行建模。视觉神经系统在整个大脑中所占比重很大。
视觉无论从生物的进化，还是从个体生物的发育来看，视觉都是基础（视觉在人的成长中的作用，在人成长中对人智力的发展作用，视觉不好对人的智力影响之大不容置疑。的确生物中高级动物都是视觉认知为主。），其他的思维很大比例都是构建在视觉基础之上，很多语言都是首先反映普通的视觉对象，抽象词语也大都是从具体的词语抽象而来（立规矩的立就是从具体物体的站立而引申到抽象概念规矩的站立。规矩规则这种抽象概念是看不到的，但是能感受到，能理解它的存在，所以立规矩的立就从具体站立的概念延伸到了抽象领域。）
智慧的高低是抽象思维的深度，是理解概念的抽象深度，最基础的概念是具体的物体的概念，视觉中具体物体的形体概念，




三：技术模块资源：

-2 3d数据集有图像和深度！

-1 commaai 的gan 预测 ，

0    摄像头输入图像视频的WGAN训练

1    从视频中学习运动信息，
        运动信息向量的神经网络学习 code、ppt、视频ok
        周末轻松一刻，欣赏完全由程序自己回忆的视频片段
！videogan code

2    区分背景与主体；
        周末轻松一刻，欣赏完全由程序自己回忆的视频片段 自动学习区分主题背景及运动
        attribute-disentangled-VAE手动设置主体与背景等论文有涉及。

3    GAN的Dis可以是非常多的子网络分别对整体和局部进行监督学习-需用到2从运动中学到的主体。 
        wayaai有部分 pyramid层级生成；分类识别网络的多层级集成-每个概念的识别才有对应的每个概念的生成。
        https://github.com/255BITS/HyperGAN


4    从注意力机制引导学习焦点到特定的主体的局部。结合3自概念的学习

5    学习到的概念的语义标注即语言联系
        通过文字生成图片的神经网络技术
        文字生成图片的相关，训练过程应该先由简单物体学习再到复杂场景的学习。
语义标注betaVae INFOGan。

6    训练过程的多类别loss函数训练：比如先WGAN训练稳定然后beta-vae训练；不同维度互为监督-视觉听觉信息的互相监督学习：猫狗的叫声和形态听觉视觉互相监督学习。pix2pix  cycleGAN 等。
cyclegan 学习深度信息的效果  相当于从图像联想到深度，使用记忆联想，陌生环境才需要更实时的雷达深度探测，熟悉环境使用记忆提高效率（运算效率，使用效率）

！vae算法和WGAN算法的深入比较。
！wgan 输出二维是否会类似vae？
！lsgan效果测试
    
7    模型扩展技术-子结构学习比如人脸的眼镜-鼻子；子结构的自动学习

8    结合强化学习的特征利用，强化学习使用betaVAE特征。强化学习验证智能学习的水平程度。deepdrive自动驾驶的架构，
！强化学习代码 开源强化学习程序架构

9    复合模型，比如yolo9000中使用了很多技巧；模型整合；代码阅读分解。

10    时间序列学习；神经图灵机；GTMM；

11 自监督特征学习： https://mp.weixin.qq.com/s/DOxsLm3bYTNp3-uke0dVTA

12 apollo


end.





 对视觉概念学习有兴趣的可以联系我微信zdx3578；



         关注公众号获取更多内容：
