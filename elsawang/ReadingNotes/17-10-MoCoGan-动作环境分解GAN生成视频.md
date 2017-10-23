### **paper 阅读笔记**

***17-10-MoCoGan-动作环境分解GAN生成视频***


Z --> Zc & Zm

4个网络

动作分布Rm - GRU - > Zm
生成器Gi - DCGAN
判别器Di - DCGAN - 图像生成质量
判别器Dv - spatio-temporal CNN 视频&动作生成质量

随机噪音向量被分为环境和动作Zc和Zm对应content subspace和motion subspace，分别是背景内容和动作变化。

利用RNN从动作空间获得Zm分布，随机变量e --> RNN --> Zm

Conditional

Label -> RNN -> concatnent Label with Zm

