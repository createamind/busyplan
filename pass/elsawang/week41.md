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
<<<<<<< HEAD
- 解决方法：明天尝试TensorFlow的环境 --> 考虑迁移pytorch平台 -->实在不行就自己重写！
||||||| merged common ancestors
- 解决方法：明天尝试TensorFlow的环境 --> 考虑迁移pytorch平台 -->实在不行就自己重写！

---

##### **paper 阅读笔记**

*基于文字描述生成相应图片的GAN*


- GAN-CLS

增加一个D的loss function，为G提供新信号

> In addition to the real / fake inputs to the discriminator during train- ing, we add a third type of input consisting of real im- ages with mismatched text, which the discriminator must learn to score as fake.By learning to optimize image / text matching in addition to the image realism, the discrimina- tor can provide an additional signal to the generator.


| GAN-CLS  | real image  | fake image|
|:----- |:--------:|:----------:|
| right text | D- |  D+ |
| wrong text | **new** D+ |  -- |


- GAN-INT manifold interpolation

融合条件文字描述

> Motivated by this property, we can generate a large amount of additional text embeddings by simply interpolat- ing between embeddings of training set captions.

```
beta*t1 + (1-beta)*t2
```

- 风格迁移

![style encoder](./images/text2img-styleencoder.png)

```
s <- S(x), x_hat <- G(s, psi(t))
```

*x_hat* is the result image and *s* is the predicted style.

- 风格迁移的评估方法

> **ROC curves using cosine distance** between predicted style vector on same vs. different style image pairs

> To recover z, we inverted the each generator network as described in subsection 4.4. To construct pairs for verifica- tion, we grouped images into 100 **clusters using K-means** where images from the same cluster share the same style. For background color, we clustered images by the average color (RGB channels) of the background; for bird pose, we clustered images by 6 keypoint coordinates (beak, belly, breast, crown, forehead, and tail).


本质上依然是pair的conditionalGAN，特点是由pix2pix的图像表征生成转换到*文字表征*生成，由利用GAN学习图像分布，到学习文字描述映射图像的分布。

基于DC-GAN, text encoder [char-CNN-RNN](./pages/1605.05395.pdf) [http://arxiv.org/abs/1605.05395](http://arxiv.org/abs/1605.05395)

> a deep convolutional generative adversarial network (DC-GAN) conditioned on text fea- tures encoded by a hybrid character-level convolutional- recurrent neural network. 

模型现有问题，生成图像解析度小 64x64
=======
- 解决方法：明天尝试TensorFlow的环境 --> 考虑迁移pytorch平台 -->实在不行就自己重写！

---

>>>>>>> 9a92635df27bdb831662ed027e6323840283c679
