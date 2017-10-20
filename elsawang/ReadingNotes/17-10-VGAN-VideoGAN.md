### **paper 阅读笔记**

***17-10-VGAN-生成视频的GAN***

[**Generating Videos with Scene Dynamics**](http://papers.nips.cc/paper/6194-generating-videos-with-scene-dynamics)

#### 应用

识别：动作分类

预测

表示学习：

#### 实验计划


#### 论文概述

模型结构

![VGAN](../images/vgan.png)

将前景及背景分解，利用两个G分别生成动态的前景和静态的背景

>>  We use a two-stream architecture where the generator is governed by the combination:
>>
G2(z) = m(z) ** f(z) + (1 ** m(z)) ** b(z). 
>>
... the foreground f(z) model or the background model b(z) for each pixel location and timestep. To enforce a background model in the generations, b(z) produces a spatial image that is replicated over time, while f(z) produces a spatio-temporal cuboid masked by m(z). By summing the foreground model with the background model, we can obtain the final generation. Note that   is element-wise multiplication, and we replicate singleton dimensions to match its corresponding tensor. During learning, we also add to the objective a small sparsity prior on the mask  km(z)k1 for   = 0.1, which we found helps encourage the network to use the background stream.

视频表征学习

替换D最后一层，K-way softmax classifier
