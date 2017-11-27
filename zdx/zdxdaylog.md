计划 总结 日记 反思； 最重要！ 目标   明确的！
推进慢-缺少数学、程序、深度学习、方面高手，做出东西吸引。  qq 群

cardemo-autoware；
vide GAN; gan车道线，pix2pix;  深度到seg；3d；


5年  1年  1月 1周 今天  现在  今天最重要的目标计划：？？ 



think; plan; log
短期、近期任务：videogan+vae进行编码

视频vid2vid action reward ?? 声音到图像？ time contrast 跟 vid2vid 的关系？？









cyclegan 预测  视频，cyclegan 有车情况，无车情况。
视频预测想到的是，关注点更少，可以在更少的数据或不同的小方面简历语义关联。类人的注意力模型。

avb大图片； beta vae avb running  

commaai  z interpolations   z1-z2差值的应用？？？

5 1024 preotrain interpolation

run bayesgan????

vid2vid？思路分解，张炜配合测试。
1，长视频 2 优化效果-ref mocogan   3 保存视频结果。 4 wgan or  bayesgan 改进。

prefnet：  model’s learned representation in this setting supports decoding of the current steering angle.






todo: 
1 优化效果 dual motion papere; mocogan  bayesgan; beta-vae处理抽象特征；图片是否独立生成然后增加图片的Dis；
几篇论文需要看看 dual motion；

2 视频预处理放到程序里面，可以自动设置训练预测的间隔。； 输出图片拼接正常和生成的，或转为视频发公众号。
3 训练不要抽取中间帧，训练帧数加大比如16帧；循环训练一条路，网络会不会记住这条路？？？？   隐变量会有路的概念？？

4 多传感器的属性学习---导致理解背景路 树木 电线杆等 不动，车动。不同物体不同属性不同运动属性。

5 beta vae avb训练时候就自动间隔输出每个维度的图片，即手动运行的结果自动输出。  
从总数据集里面随机找一张进行处理。range也在一个范围随机，20张插值。
取值写到文件名上

todo：公众号文章 思路 等等。   多传感器---只抽取某一传感器的特征的相关数据的预测。
老熊的论文。












结果为导向吗？

9点沟通计划，12点说明进展及问题。5点继续说明一次。或者书面  这个时间点，提交github。

安装正常情况进行计划，意外困难遇到了在家时间，不能预估了困难的时间，然后做的时候用顺利的思路去使用时间。

必须回答的问题写给我。不回答问题就不能做事那就沟通，要不就把问题写下了回答沟通。大方向的讨论以做出具体进展来争取机会。周末可以讨论，平时做事无大方向讨论。
！！沟通大方向的前提是什么？？需要什么基础来沟通？

没听懂自己也努力想，我也要学习学习；没听懂自己努力学。

如果事情做好了先休息旅游。或先放两天假。























11月见month11.md文件



11.2 周四： pggans 代码学习及运行。同事工作跟进；

pggans 找到可以运行的代码，运行ok； 代码的分析学习！。
h5文件可以分析；我们的文件没问题，分析代码！
新代码可以跑我们的数据； 3号ok；
gpu可以启用


11.1 周三： pggans 论文；同事问题跟进；





31  新数据集训练 avb； 图片隐变量。   数据多样性不够？？

周二：练习ppt；下午盛大 bp及公司头脑风暴  ；晚上锻炼；celbea 1024；


30 周一  插值看中间图片生成效果；提取特征向量+——图片看效果

优化思路：1 
关键问题：数据集特征不同，人脸和道路，网络调整是否会比较大
2

上午 下午 晚上ppt  bp；




avb commaai 两个模型
1 插值更多。效果变化不好。  1.1插值方法  弧形。
2 选不同的图片插值
3  图片的隐变量分析
4 特征使用效果






29 周日 ppt
28 周六 baiud会议及ppt

27  周五   评审准备； load 后图片生成问题；  统一的思路改进。
todo： 1 commaai 差值 2   avb 大图片！

沈先生沟通很多建议改进ppt

周四生成图片为test，今天增加batch norm 后图片基本正常。晚上到周末一直重新训练


26 周四：  图片的特征向量生成，特征向量的分析，特征向量的加减做道路偏向的变化。 分解
1 reload weight； 文档，参考代码过一下；  checkpoint reload；  for  1 interpolation 2  特征提取  nvidia blog 过一遍
1.1  网络比较，commaai 图像大，avb图像小，网络结构优化

找图片，reload weight，生成图片向量，找向量里面道路相关特征的维度，此特征维度变化后生成图片的变化。

2 看论文
3 跟进度。

中午盛大天地苗圃，回来跟进各个问题处理。load weight memory growth；图片/255; 前天图片显示，今天不显示；代码变化过？？代码管理！！


25 周三：  知道  做到   锻炼，用非当前紧急任务锻炼，任务分解后的子任务锻炼，python基础知识资料分享。
当前需要完成的任务：  

改进
1 数据集 找更好的
1.1 udaciyt baidu是高速路
1.2 torcs 新数据集

现有模型
2 现有数据集找合适的图片集  左前  右前 分布10张图片
3 图片集 interpolation 看效果
4  2 3 分别是  research  和  avb  两个模型的测试

中午见刘自良


24 周二  log
 
电线杆预测不连续的解决办法是？？
todo  1 avb结构优化，训练车道更合适   2  research 的特征提取使用！  选一个！！！！
3 数据集 val test  需要认真选择 ！

Allowing GPU memory growth

By default, TensorFlow maps nearly all of the GPU memory of all GPUs (subject to CUDA_VISIBLE_DEVICES) visible to the process. This is done to more efficiently use the relatively precious GPU memory resources on the devices by reducing memory fragmentation.

In some cases it is desirable for the process to only allocate a subset of the available memory, or to only grow the memory usage as is needed by the process. TensorFlow provides two Config options on the Session to control this.

The first is the allow_growth option, which attempts to allocate only as much GPU memory based on runtime allocations: it starts out allocating very little memory, and as Sessions get run and more GPU memory is needed, we extend the GPU memory region needed by the TensorFlow process. Note that we do not release memory, since that can lead to even worse memory fragmentation. To turn this option on, set the option in the ConfigProto by:

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config, ...)
The second method is the per_process_gpu_memory_fraction option, which determines the fraction of the overall amount of memory that each visible GPU should be allocated. For example, you can tell TensorFlow to only allocate 40% of the total memory of each GPU by:

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config=config, ...)
This is useful if you want to truly bound the amount of GPU memory available to the TensorFlow process.

CUDA_VISIBLE_DEVICES=  这个可以设置tensorflow使用的GPU；


24 周二 计划：模型的向量提取，avb的训练是不是23晚上跑跑？？
commaai训练新的数据G 失败，


10.23 周一  plan: 我和亮亮做车道线， nvida  博客文章 公众号；torcs之前截取视频的方法，及训练模型用avb？或之前的 commaai research基础都行。
之前训练的模型的人脸实验，或直接车辆场景的信息抽取。

1 torcs数据，之前记录torcs数据的程序  torcs 生成视频，是否保持为图片？ 查看avb是否图片？？
2 之前跑的模型，跑commaai simu torcs的程序 or avb?
之前torcs 训练结果？？   avb跑一个结果

avb数据输入方式：

3 如何跑起来，train and test
4 数据整理出相关用于提前特征的图片
5 测试去特征后叠加的效果。









10.22 休息；认知与机器学习的区别？

10.21 周六 上午面试  王全， 虚拟机安装？？？？； 下午yushikeji 面试沟通一位；论文看一点，锻炼，论文
 
10.20 周五 宋总 无人驾驶资料邮件查看；加群内容回复（最重要？？？）
1 传统和深度学习的本质区别。
2 是否是与众不同，
3 真正追求的是什么？
4 实现目标的方法，途径，是最合适的吗？

10.19 周四几乎一天沟通,通用人工智能思路，实现通用智能的思路分享，分析，梳理，表达自己的实现思路，问题是存在很大解释不清楚的点，大家依然对实现路径不清晰，无法自我驱动前进。上午到下午4点。 传感器配置； 公众号

10.18 周三：videogan： 时空卷积结构
paper16 videogan: principles in mind. Firstly, we want the network to be invariant to translations in both space and time. Secondly, we want a low-dimensional z to be able to produce a high-dimensional output (video). Thirdly, we want to assume a stationary camera and take advantage of the the property that usually only objects move.
摄像头移动和背景的变化是一致的。 摄像头移动跟踪物体移动
TGAN：address this problem by decomposing it into background generation and foreground generation, this approach has a drawback that it cannot generate a scene with dynamic background due to the static background assumption [44]. To the best of our knowledge, there is no study that tackles video generation without such assumption and generates diversified videos like natural videos.
We believe that the nature of time dimension is essentially different from the spatial dimensions in the case of videos so that such approach has difficulty on the video generation problem. The relevance of this assumption has been also discussed in some recent studies [33, 24, 46] that have shown good performance on the video recognition task.
contributions are summarized as follows. (i) The generative model that can efficiently capture the latent space of the time dimension in videos. It also enables a natural extension to an application such as frame interpolation. (ii) The alternative parameter clipping method for WGAN that significantly stabilizes the training of the networks that have advanced structure.

mocogan

10.17 周二：  log 上午访谈，下午沟通工作思路。 

计划： gan video  两周 第一1周 熟悉已有论文，跑效果，
第二周，分析隐变量的语义相关信息，自动驾驶的转向角度和z的关系 。

10.16 周一  视频无监督调研； 自监督ppt扩展开来， 父母买护膝；明天访谈准备。
todo 视频相关论文代码

探索深度广度





10.15  
上午 自监督 ，下午体验特斯拉-面试李亮亮。  https://arxiv.org/pdf/1608.07017.pdf ; 云服务器网络谷歌学术折腾1.5小时，自己网站恢复耽误1小时。
log png16位格式深度 or 32深度？
分享： text2image 

深度图对比效果做出来了。
10.14 休息看小孩
10.13  周五
think: 今天焦点不集中！
plan  1 chanal 参数，看代码已经确认ok； 2 test 视频一个看看深度的生成效果， 3 
改进  1 用pix2pix  2  代码熟悉   3数据量大  epoch大
log  代码； 测试，数据集！ 

10.12  周四   think videogan videovae  ；  paper: implicit  generator

cyclegan 直接输入数据， 2  1ch 变 3ch  gan后 3ch 1 ch  3  gan网络直接用1ch    4 pix2pix  

周四 cyclegan代码及深度图片的训练改进， 效果可以

tensorflow pytorch 比较：https://mp.weixin.qq.com/s/y5LLraGWuYUbOted9N-GWQ https://mp.weixin.qq.com/s/u1aaOxPA_-nGwKD7AQZ7Bw https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650726576&idx=3&sn=4140ee7afc67928333e971062d042c59  https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650723769&idx=1&sn=17565e650771699ceddabb214d485626
结论  PyTorch更有利于研究人员、爱好者、小规模项目等快速搞出原型。而TensorFlow更适合大规模部署，特别是需要跨平台和嵌入式部署时。

10.11 周三 如何提高数据下载和使用的速度？ aws 桌面； cyclegan

语言到图像的gan生成，语言图像可以互相学习，那么声音和图像，图像和雷达，等等。

http://www.k4ai.com/depth/index.html  Monocular Depth Perception with cGAN  两个小数据集

cyclegan： --gpu_ids 0,1,2 for multi-GPU  pytorch  应该凌晨跑完一次。

发票拍照仔细，时间久肯定忘，生物记忆不可靠。

10.10  周二 plan：内部sample；   账物！  传感器！
10.10  上午沟通：视觉增加很多非视觉属性，  确定gan cyclegan训练 深度信息和图像的互相转换，联想。;中午下午  账物梳理及报销有发票的

https://mp.weixin.qq.com/s?src=11&timestamp=1507627964&ver=444&signature=YcEj5vp8T24jw3P0gWbBzga8x2aN4gDrmrrGzIw3NvYVmv2kLTq6cFLAz8OzCaWsa0RwmyBemrv0uD*ZK4YmvJcsdL*o86ZxkJucc3X5HgftBqWas-mxsRp7U88Ngi4v&new=1
比较有创新点的一篇论文‘Unsupervised Monocular Depth Estimation WithLeft-Right Consistency’，基本理论可以这样简单理解：既然假设双目看的是同一个场景，那么已知左目图像和视差，就可以求得右目图像，同理知道右目图像和视差就可以求得左目图像。假设训练时，我们同时有左目和右目，然后只根据左目经过神经网络求得左右视差，然后右图根据求得的视差去预测左图，然后再求差，然后循环往复直至收敛。

https://mp.weixin.qq.com/s?src=11&timestamp=1507627964&ver=444&signature=xx3suuOkPwr4eRj9SHukjSBPAnBrNbopdbTFvBvPhU0iO2r44RqHNXR3uKDttvrgsg4ifQw3sZOjWGMIpxzjqXqlplILBuYT8*AlpCbHjU-OYS64abIqVnsw28jq5Crk&new=1
学界 | 用单张图片推理场景结构：UC Berkeley提出3D景深联合学习方法 2017-07-13 机器之心

https://mp.weixin.qq.com/s?src=11&timestamp=1507627964&ver=444&signature=xx3suuOkPwr4eRj9SHukjSBPAnBrNbopdbTFvBvPhU0zbBptKvTJZxlAGtc9eeSnKZzBJD89CjXp81IXTjMh496*iF6vasL9fL*ClmZ3kBdyCINnrKSO2T-B0XGti9RG&new=1
专栏 | CVPR 2017论文解读：基于视频的无监督深度和车辆运动估计 2017-07-27 机器之心

Paper Reading | VINet 深度神经网络架构
https://github.com/createamind/SfMLearner

kitti ref dataset  http://make3d.cs.cornell.edu/data.html
NYU RGB-D Dataset: Indoor dataset captured with a Microsoft Kinect that provides semantic labels.
TUM RGB-D Dataset: Indoor dataset captured with Microsoft Kinect and high-accuracy motion capturing.


车展仿真环境效果！ 
10.9 上午王一舒沟通，下午，材料配合小王，工作沟通自动驾驶autoware cluster视频；  专注最重要一件书的学习；6：30看宝宝，锻炼10：00，改模型runceleba 11点20
10.8 interpretable  paper; avb celeba test run;avb celeba z 16dim;   下午；mutualinformationI(x;y)
gan paper for driver sad;
10.7 下午公司看论文代码，avb，吃饭后沟通，先生，汽车资深人士，但是太自信，什么都懂；；锻炼；公众号及avb run test
空杯心态，接受不同的观点，自己什么都懂，来了都要指导你公司的策略方向
10.6 白天休息，晚上陆家嘴，回来跑celeba ； 昨天的脸是什么意思？？
10.5  脸，下午电影，傍晚论文interpretable一点，锻炼，改gpu，run avb mnist；，avb ok； 其他机器环境耽误一小时；

10.4  休息一天，晚上发公众号
10.3  上午休息，下午论文 avb，晚上锻炼，陪小孩吃饭，
10.2  下雨，白天休息，晚上锻炼，论文 avb
10.1  上午世纪公园休息，晚上听歌，看了些得到文章
9.30 早上沟通，工资，commaai效果发公众号，代码提交github； 讨论沟通

2017-9.29 计划： 熟悉apollo；
总结：周五：上午计划讨论；

2017-9.28 周四： 51job简历查看一周的情况，电话沟通了两位。 训练模型跟进查看；

2017-9.27 周三： Learning a Driving Simulator   1608.01230v1;
