

11月

第一周
11.3 周五：
0   pggans code 学习
1   gpu可以启用  pggans
2   digits 自己的两个models
3   pggans pytorch版本运行？？？
4   pggans 版本比较？？？ 可以运行的版本和官方版本？？？

1 pggans 数据集从200张扩大！
2 dcgan web 跑起来！
log：工作沟通；训练效果跟进，吃早饭思考休息，亮亮聊聊，发工资； 

11.4  周六
事故：   10点  rm  删除了正在训练的程序文件！    
事故2 ： 11点   kill掉了同事训练的digits ！！
下午讨论gan的隐变量，开会反应出了一些问题，进行后续思考。

11.5 周日：工作反思如何沟通，如何开会，中间陪小孩，通用智能视觉文章学习


第二周 

11.6 周一：去车站送宝宝回姥姥家
？？？pggans如何训练更快。
11.7 周二：  训练更快
gammar 分布；
pggans代码； 
avb mnist run。

z分析不理想

周三 11.8
todo 1  avb notebook run； 2  commaai notebook 车道方向test，找图片，找z；找z的差值，测试差值应用的效果。
2  beta vae  两篇论文； 3 beta vae 作者  论文。
3 avb-betavae  代码改进。
4 昨天微信公众号记录的 commaai model 采样空间不同 内容不同。
5 1024 preotrain interpolation

z分析失败。


周四 9 log1 亮亮 命令行没看全，认为自己看了，然后认为认为机器坏了。 学linux多久？   年轻人自己没达到效果，更多会任务环境或外部因素问题，还极少会认为认为自己努力不够。
问然后呢，是我吧各种思路都梳理出来，做和思路梳理如何平衡，，你去做，我就去梳理下一步了，
没搞好，认为小车问题，要拆。  小车是不是坏了？？我是不是没搞清楚，没弄对呢？
一批评就反驳，前置过滤，   拒绝承认错误是人的本性。
做了这一步就知道然后了，
小车使用时间：----------------------------------- 晚上，影响大家，工作重点
pix2pix 2 bigan 比较？？？

上午讨论做vid2vid ;下午 小寒来访，亮亮沟通；傍晚vid2 沟通；晚上张炜沟通；然后看电影没锻炼。


周五 10： 
我可以随时问工作进展；日报必须写，
我说我现在没能力做出最好的选择，选择最重要的事情。，现在没有足够更好的能力，不是我一直都是个..。  我话一出就是：没能力找人啊，找更强的人啊，这就是前置过滤。言之有据，不是张嘴就来。

刚才讨论 videogan不做了，你们说 是你们认为找不到z，所有不做了，不是我要求不做的，另外中间插入了改装车传感器事情，插入了bp事情。

log: 工作沟通； 论文；工作思路反思，z问题，等等

周六 11：
1   1024 interpolation  2 vid2vid  

vid2vid
mocogan - 2个D ，一个D ->3d convolution 视频，2个D-image 1frame
G 直接生成视频，还是先生成图片，再生成视频？  大概是直接输入z；
pix2pix  扩展 vid2vid 
pix2pix d 输入是两张图片，  扩展为输入两个视频片段，

12  上午见李政 康宁。下午休息，晚上看电影， 失眠是因为不累？


第三周

周一 11.13   计划： vid2vid？思路分解，张炜配合测试。  2  1024 interpolation  3 avb infer
vid2vid
mocogan - 2个D ，一个D ->3d convolution 视频，2个D-image 1frame
G 直接生成视频，还是先生成图片，再生成视频？  大概是直接输入z；
pix2pix  扩展 vid2vid 
pix2pix d 输入是两张图片，  扩展为输入两个视频片段，

简单的可以用conv3d or upsample3d
视频生成参考mocogan；mocogan图片生成可以用cyclegan混合训练。
pix2pix的d是否可以使用mocogan的视频判别D。   mocogan的rnn没有理解。

log pix2pix concatrealAfakeA； 3d 文章看。 下午代码等吧

周二 14  pix2pix  除unet还有renset的结构
pix2pix 的G结构，论文；prednet； 文献google；视频综述。
下午：  A Survey on Deep Video Prediction  论文   time contrast; prednet
4点半锻炼到6点多

？结果导向吗？环境搞坏了，不恢复好就下班了？而且时间还不到6点半，环境仍那里也不能继续训练？至少主动说一下。
先pip uninstall 再pip install 就好了。

晚上：paper继续，小孟沟通， 视频预测思路继续看论文
video_pred_survey 觉得收获不大，是我们做的看的更往前吧，论文最新的gan视频就一篇。 time contrast ; prednet; self-supervised;

However, other experiments have shown that incorporating model-based search techniques can achieve breakthroughs in domains where purely model-free methods would have little success [19] [48] [54]. Having an accurate action-conditional video prediction module would enable RL agents to explore new strategies based on whether it would lead to novel outcomes [20] [40] [51], perform long term planning [14] [65], and determine visual regions that are contingent on the agent’s actions [1] [40].



周三  15 杂事，展会，沟通智能孟晓宇；公众号；亮亮终止合作。
conda install --channel https://conda.anaconda.org/menpo opencv3
不同beta参数的model weight。

周四：16 todo 1  张炜 mnist mode weithigt测试；
vid2vid 计划：
log1 工作事情思考，得到李笑来文章；2 betavae 继续看效果 训练beta 400； 3 宝宝视频。4 跟进v2v;5 v2v问题跟进。



周五 17  中午 张炜问题解决，模型可以跑起来。
todo：1，长视频 2 优化效果-ref mocogan   3 保存为视频结果 ok。。 
4 wgan or  bayesgan 改进。

不同数据集效果及程序

想了很久的想法，想法的独占，放下开始做。
prednet： model’s learned representation in this setting supports decoding of the current steering angle.
 
周六 18  Time-Contrastive Networks  继续周五任务。
写论文：？？  
  
todo: 
1 优化效果 dual motion papere; mocogan  bayesgan; beta-vae处理抽象特征；图片是否独立生成然后增加图片的Dis；
几篇论文需要看看 dual motion；

2 视频预处理放到程序里面，可以自动设置训练预测的间隔。； 输出图片拼接正常和生成的，或转为视频发公众号。
3 训练不要抽取中间帧，训练帧数加大比如16帧；
4 多传感器的属性学习---导致理解背景路 树木 电线杆等 不动，车动。不同物体不同属性不同运动属性。

5 beta vae avb训练时候就自动间隔输出每个维度的图片，即手动运行的结果自动输出。  
从总数据集里面随机找一张进行处理。range也在一个范围随机，20张插值。
取值写到文件名上

todo：公众号文章，论文 思路 等等。
老熊的论文。




周一 11.20

plan todo：
2 视频预处理放到程序里面，可以自动设置训练预测的间隔。； 输出图片拼接正常和生成的，或转为视频发公众号。
！！
5 beta vae avb训练时候就自动间隔输出每个维度的图片，即手动运行的结果自动输出。  
从总数据集里面随机找一张进行处理。range也在一个范围随机，20张插值。
取值写到文件名上

plan：  论文 dual；         展会准备。 公众号 微信扫一扫。 createamind。 展会的文章准备。



周二周三 去常熟车站，中间跟进了视频生成的事情
周四：plan：
细节1：  左侧还是两个A吧，A 前后都对比！   2  帧数24？？好像不是啊！   3 MORE GPU MORE !!

什么样的好效果：视频生成？ 2   车辆的预测？？车辆的视频生成。直接用之前的预处理也可以，
3 公众号上传视频  王亚会挑选视频上传

5  beta vae； 论文dual； fader network 
 
nvidia端对端训练驾驶和视频预测驾驶+动作  区别？

视频动作的视频预测 反过来 视频视频的要求或目的或期望的下一步视频效果比如往前走的 动作预测或动作输出
往前走如何定义及设计训练？？

加入深度信息和动作信息的视频预测！！


周五11.24   细节： convolution 3d kernel  时间是否还必须是1 ？？     
plan todo !!   论文dual； fader network  ；  bayesgan 中文翻译代码等。
multi sclae  16 lecun paper； prednet；  paper
我的思路介绍。
现在我们预测学习的不足和缺陷？
是学习或记忆？语义记忆如何实现？？

预测加入深度信息。


!!日报
周六周日休息；发公众号，出去转转










周一 11.27  todo 1 betavae ;深度图片。
视频vid2vid action reward ?? 声音到图像？ time contrast paper读-- 跟 vid2vid 的关系？？


周二： time contrast paper; imitation paper;

空挡 自己学习还是看前沿论文。

周三： 日报？？要求不合理？
torcs 录制下来都有哪些信息？ 多任务； 动作到视频？unet就不行了，网络结构？？！声音到视频？？

memory：视频重放-存储z？多感知维度--倒水后杯子轻-重力感知-声音感知-

condtion action pred video！

imitation 都是直道，就都是相同的指令。
多目标：路中央，action condition,action超路中央跑，拐弯刹车，冲出去-惯性？
图像跟reward的关系，图像跟速度的关系，图像跟位置的关系，图像跟动作的关系？图像跟是否在直道，是否在弯道，是否

关系 决策 技术（视觉 imitation，强化学习，video，认知） 场景 现有的各种（技术、认知、）缺陷

场景（路的不同位置，不同的路，车道线，路牙子，十字路口，多车的情况、行人、其他动物）

自身情况（速度、位置、决策动作）

偏离补偿-偏离的视觉确认。


周四--周日  回日照休息




























