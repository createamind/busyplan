




及：第一本无人驾驶技术书 的目录 可以看到无人驾驶的整体技术框架。及 https://mp.weixin.qq.com/s?__biz=MjM5ODY2MzIyMQ%3D%3D&mid=2652428995&idx=1&sn=e06840fd2d30e2d5756a7d3f3f566c14&scene=45#wechat_redirect


### 什么更重要？？ 为什么是这个？？  更多牛人- 清晰的目标
找到正确的发展路径！行业认知！人员成长！当前最重要的是什么？为什么？   自动驾驶当前迅速发展的环节是哪个？    做哪个是可以快速发展的



### 行业生态  技术环节   应用场景  车厂合作

### 商业应用探索：
应用场景的探索及商业模式的开拓我们跟进各个自动驾驶公司，百度 驭势等，以当前的资源和市场能力我们还没有探索商业应用场景的能力，但是我们做好自动驾驶技术是完全有价值的。做地图我们比不过千寻，做其他技术难度低有市场的产品但是比我们资源多，人力多，经验多的公司太多了，我们比不过这些公司。我们人员经验少，做最难的，没有思路限制，大胆试错才是我们的出路。改进或突破自动驾驶技术是一条无人走过的新路。挑战很大，需要我们踏实快速试错。

### 自动驾驶的算法智能分析：
自动驾驶要想部分替代人的功能，需要与人部分相当的智能，当前自动驾驶的问题：  智能算法智能程度不够：比如对环境的感知非常初级，表现：不能像人一样教他学习及处理相关问题。 1 智能不够，学习速度慢，不能举一反三，必然训练时间过长，依赖大数据进行训练，2 智能不够，识别环境不精准，所以不够安全。很多公司改进fasterrcnn等算法离程序用视觉理解世界还差比较远，需要更好的进展。当然fasterrcnn很有参考学习价值。

### 生态定位：
自动驾驶生态：虽然有高校，但是缺少算法合作伙伴；我们在生态中的定位：最接近生物智能的视觉解决方案。 CreateAMind，服务自动驾驶的视觉系统，而非自动驾驶任务特定的视觉解决方案。现阶段定位为技术公司，为自动驾驶的普及做好感知基础。  技术如何在某一点，超越baidu 谷歌

### 方向和中期目标：
在自动驾驶整个技术框架中，我选择的切入点是perception感知和决策。视觉的精准感知，视觉输入即可精准的感知（准确的认知）其他非视觉属性，如触觉属性、压力、深度、速度等可以通过视觉感知到，
视觉感知训练阶段会结合多传感器进行训练。输入是摄像头和雷达的原始信息输入进行训练。输出目标会逐渐完善，从基本概念的学习开始逐渐分化，运动物体的跟踪，感知分辨率的逐渐提升，感知物体类别的逐渐扩大，认知概念的抽象逐渐提升。

### 提高智能学的快需要：
1 特征学习更准确和完整及更抽象的高层次特征 参考公众号生成模型系统相关文章。 2 人分析驾驶任务相对简单是因为人对周围环境的认知非常抽象准确，是在低纬度的空间进行运算分析，如 路 角度 速度等等。


### 短期目标9-11月：  

仿真环境，及仿真环境里基础adas相关功能实现（用我们自己的视觉系统思路来进行实现，及通用视觉系统，而非驾驶的专用视觉任务系统。）
仿真测试的必要性： 扩大自动驾驶环境完善参与人群，扩大自动驾驶算法训练参与人群，降低驾驶训练爱好者参与门槛，提高测试频率，提高测试效率，提高模型验证速度，仿真方便更改各种环境条件
仿真环境可以快速了解各种真实传感器数据类型，相关数据的处理训练算法，及多传感器数据fusion方法，各种模型的验证训练也更快。

仿真环境的学习模型先增加视觉cnn等处理功能，再改进。


深度学习"视觉"和强化学习。简单 gan vae等等



强化学习的必要性：自动学习，而非全部人力规则，人工指导即可。

仿真软件： 使用gazebo及torcs等仿真端对端的方式。 天欧商业软件

### why
目标相关性！
仿真的价值：
角色






路径一：新算法改进应用：
自动驾驶问题分解： 

单车：1区别路和非路 2 路的抽象路线概念，区分无人道路与有参与者的道路，无障碍道路与有障碍道路，沿着路线开 3 直路的速度和稳定性问题  4 拐弯imu 速度和稳定性问题。 一个神经网络学习的概念越来越多，在这个基础模型上面进行持续学习和扩展，技术思路有两种—模型能否扩展 2 用更大模型从头开始训练学习各种概念。
2 多车互动，强化学习博弈。
3 路人场景互动，交通规则

深度学习： 特征学习，概念学习，概念应用 beta vae 论文，infogan，video gan；
强化学习 自己探索环境习得一定的经验。

视觉精准感知作为切入点，包括雷达传感器感知。
	视觉改进点  beta-vae； 视频gan； pix2pix联想
决策切入点是用强化学习及rl-teacher 等；acktr a3c
	决策突破点  决策学习 rlteacher 强化学习等技术。mobileeye也有决策技术分享可以参考 todo 1。

参考：

概念抽象特征等学习：
1 beta vae 论文 ，学习独立视觉特征  https://mp.weixin.qq.com/s?__biz=MzA5MDMwMTIyNQ==&mid=2649291563&idx=1&sn=52517fba9fc521c430a025a59b318937&chksm=8811e
Abstract: Learning an interpretable factorised representation of the independent data generative factors of the world without supervision 
is an important precursor for the development of artificial intelligence that is able to learn and reason in the same way that humans do. 
We introduce beta-VAE, a new state-of-the-art framework for automated discovery of interpretable factorised latent representations from raw image data in a completely unsupervised manner.   详细参考论文

2 video-gan 场景理解，背景 运动物体的学习。 https://mp.weixin.qq.com/s?__biz=MzA5MDMwMTIyNQ==&mid=2649286943&idx=1&sn=26d0978fb432f831dca4b67d51dee109&m
TGAN

3 commaai的场景预测学习。https://mp.weixin.qq.com/s?__biz=MzA5MDMwMTIyNQ==&mid=2649285914&idx=1&sn=0b3a8088a6f6c0932bcb4aebe247bdda
https://github.com/commaai/research/

7 运动vae  https://mp.weixin.qq.com/s?__biz=MzA5MDMwMTIyNQ==&mid=2649290203&idx=1&sn=2712f62ab22a3a952ec89b5b45712f22&chksm=88


视觉认知：多传感器特征互相感知
1 pix2pix 视觉特征映射转换  cyclegan  https://mp.weixin.qq.com/s?__biz=MzA5MDMwMTIyNQ==&mid=2649291376&idx=1&sn=c4b4b567b144731781d5c2955286c1eb&chks 
ref   https://mp.weixin.qq.com/s?__biz=MzA5MDMwMTIyNQ==&mid=2649291640&idx=1&sn=8feacad9cde0f813296dc33825110717&chksm=
不同感官的联想，不同维度信息的区分。



仿真环境模拟器： 
5 gazebo https://github.com/osrf/car_demo  各种硬件传感器可模拟

视觉认知构建思路分析：
6  语义学习-通用智能的切入点-实现路径v0.01
 https://mp.weixin.qq.com/s?__biz=MzA5MDMwMTIyNQ==&mid=2649290721&idx=3&sn=c2acecd52b6526973a283ef730925b26&chk 












-----------------------------------------------------------------------------------
进展参考baidu开源，在baidu开放的场景上面，我们的感知超过baidu开源的感知即可，todo 测试对比标准

路径二： 常规算法的集成。autoware 等。

联想的功能 数据多维度差异。  
提高智能带来的  1 降低数据依赖，2 加速训练  rlteacher  rlt 2.0 指定行为3 识别精准   beta vae， videogan
更远距离感知到相关物体依赖硬件。感知更准确依赖软件
汽车保有量-和厂车合作，合作门槛是什么？？ 生成模型摄像头车道保持做到芯片低成本车厂合作，算法部署到车上去需要什么条件？  	摸清adas的研发部署情况！
rlteacher torcs更好，torcs特技，gazebo 特技；物理小车特技，物理真车特技。
