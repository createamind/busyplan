
请 总批评


tesla发布会报告分析  

我们判断tesla已经具备robo taxi落地能力；需要尽快实现我们落地的能力（算法 算力 数据需要一起推进）。

我们如何更快？


Tesla的全自动驾驶视频：
 长视频 https://www.youtube.com/watch?v=nfIelJYOygY    压缩短版本 https://www.youtube.com/watch?v=tlThdr3O5Qo 
Tesla发布会芯片视觉等介绍
https://mp.weixin.qq.com/s?src=11&timestamp=1556264066&ver=1569&signature=8elC0C*1N11q-hqtpN9IBJzYHH04XoP73-eP5JZfKZ3-q*DAlH5trOfiIKXvxHZItVjx*kiWooBgvCargTfsdcBai8KjWUCvC39uIW*j2Dsolsf6SfjvPU2doyC0AG7v&new=1

算法+算力+网络（数据）将是最核心的生产力。
公众号群发了视频及上述链接
https://mp.weixin.qq.com/s?__biz=MzA5MDMwMTIyNQ==&mid=2649294151&idx=1&sn=13f006a53ceb29bf47a9d5a2b6da988a&chksm=88101301bf679a17b2d68e25826dc92a28f33b65d4c297867732cfbeed1115555f59cb09830d&token=893529594&lang=zh_CN#rd

Openai Five 对算力、算法、数据的关系描述：We were expecting to need sophisticated algorithmic ideas, such as hierarchical reinforcement learning, but we were surprised by what we found: the fundamental improvement we needed for this problem was scale. Achieving and utilizing that scale wasn’t easy and was the bulk of our research effort!
在较小的算力和数据下，远远没有达到算法的天花板，限制还是在数据和计算力。算法持续改进但是进展速度不稳定。openai five的45000年的游戏经验说明也需要继续提升算法。

如果传统方法可行，为什么AlphaGo需要深度学习；现在只有tesla实现了数据收集的闭环，如果所有场景都是训练集数据，实现的无人驾驶效果会怎么样？

做好无人驾驶的算力数据的基础设施，持续提升算法能力。
落地紧迫感
1 Tesla让我们感到自动驾驶行业的威胁，而不是waymo apollo； tesla 明年落地robo taxi ！
其他威胁来自会跟进telsa自动驾驶思路的大公司 百度 车厂  商汤 旷世等，当然跟进者面临共同的门槛：完整的大量的驾驶数据。
tesla有无人驾驶行业流程及技术迭代的闭环，车辆收集数据量大，算法迭代速度快，常规算法（模仿学习监督学习检测识别等）实现了落地应用。
2. 威胁的启示：用AI做无人驾驶； 智能的发展思路是正确的，激光雷达高精地图的路线是没前途的。



对策：更智能速度更快的AI算法及tesla技术路线的技术基础设施。    数据 算力 算法 的生产力。
商业模式：  telsa 是无人驾驶的苹果（软硬件算法数据闭环），  我们需要无人驾驶的 android 联盟；
中国场景数据 + 联合waymo工程能力  + deepmind createamind算法  对抗 openai+Tesla 


我们的技术路线、发展路线及需要的资源。

数据： 完整的大量的驾驶数据来源的解决思路：
	目标车厂：   找对telsa有威胁感的中外车厂及有合作点的无人驾驶公司+ 有行动有投入的； 上汽  小鹏  蔚来  车和家 （pass 无电动车的车厂）
算力： 算力及分布式训练软件基础设置人才等：猎头挖人。
算法：现在的planet有数据就能学好！持续改进其他的相关各种算法—todo技术架构分解  1。  2 认同telsa落地robo taxi 的分析。

人员结构目标：
数据  收集清洗标注，核心case的整理 等人员
算法  DRL CV方面的人才；提升算法及泛化能力；
计算  高性能计算、分布式架构人才；


数据闭环ok，可以走telsa的路线，没有数据则还是算法优先的思路、同时加强数据和算力的投入。
telsa从基础算法+数据是否是适合我们的路线？
高级算法迭代



路径：时间 目标 计划    
0 planet效果
1 车厂合作沟通，拿到数据
2 人员招聘常规算法的改进版实现及训练
3 算力增加实现效果
4 更好的算法的效果提升
5 循环往复进行改进。

效果计划分解： 博士；


功能：全部基于端对端的
1 红绿灯最强的泛化能力
2 行人避让，异常情况保守处理。
3 换车道，车道保持，超车加塞
4 高德地图导航，抽象位置地图的理解；4.1 会看地图；4.2 会用地图；
5 指令控制的高层动作执行，拐弯换道停车等


todo list：

提升训练速度：
1 多传感器互相监督，轮胎发动机感知传感器
2



无监督效果：深度效果，视频无监督分割，
大网络的效果
模仿学习效果
model base的表示学习 stcn   Predictive Filter Flow
注意力 好奇心 互信息
value fuction
priority replay buffer
多传感器监督 双目    
动作抽象 action repeat；
听指令 command reach goal；  
数据并行的训练


效果 投入 预算

算法强了都容易，算法不强都不容易。




1  Tesla 发展说清楚，

仿真demo：    carla 规则10个的效果 到真实环境。
真车demo      carla 规则10个的效果 到真实环境。
 
，tesla的竞争对手。


todo技术架构分解  1。  2 认同telsa落地robo taxi 的分析。