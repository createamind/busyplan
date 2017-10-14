***Week 41 of 2017***


### 14.10 Sat

找到一个视频，[CNN-SLAM](https://www.youtube.com/watch?v=z_NJxbkQnBU)，用单眼图像还原3D精深。抽空看下这篇[论文](https://arxiv.org/abs/arXiv:1704.03489v1)

明天主要体验大老板的Tesla S，按照官方介绍，Tesla 的SDC重点技术在视觉领域，现在基本实现L3级别的自动驾驶。


### 13.10 Fri

**Summary**

CycleGAN 和 pix2pix 在深度检定上表现良好，虽然训练集用的virtual kitti，但在真实视频图像的深度表现仍然很好，且做的人较少，有发展潜力。pix2pix因为有aligned数据不断在像素级别上纠正，轮廓更清晰，训练效果也更好。

待解决问题：

- [ ] 生成数据**量化**检测问题
- [ ] 模型优化和如何优化

***日常吐槽:wink:***

**Schedule**

- [x] run video with Cycle-Gan-Depth-Model

- [ ] deploy text2img env

- [ ] familiar code and structure

- [ ] find research of GAN image segmenation, organise state-of-art tech in semantic image segmentation

- [ ] >> read [char-CNN-RNN](./papers/1605.05395.pdf)

---

### 12.10 Thu

**Schedule**

- [x] read [Generative Adversarial Text to Image Synthesis](pages/1605.05396.pdf) [https://arxiv.org/abs/1605.05396](https://arxiv.org/abs/1605.05396)

- [ ] > deploy text2img env -> try paper's model

**Tomorrow Plan**

- [ ] deploy text2img env

- [ ] familiar code and structure

- [ ] find research of GAN image segmenation

- [ ] read [char-CNN-RNN](./pages/1605.05395.pdf)

---

#### 一点感想

- 有尝试利用Conditional GAN pair 连续值（e.g. 弯道曲线）的价值
- 初步搜索，没有利用GAN做segmentation的先例，可以尝试用GAN进行semantic segmentation和image localize
- GAN探索方向：1.风格迁移领域的扩展，如音乐，文字 2.高分辨图像的生成 3.应用 参考[动漫图像](https://hiroshiba.github.io/girl_friend_factory/index.html)
- ？用GAN进行3D渲染

##### 未完成认为原因

- 论文环境部署失败（原文使用porch）
- 替代的TensorFlow部署时环境没网络，明天继续尝试
- 解决方法：明天尝试TensorFlow的环境 --> 考虑迁移pytorch平台 -->实在不行就自己重写！

---

