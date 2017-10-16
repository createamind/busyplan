***Week 41 of 2017***


### 13.10 Fri

**Schedule**

[] run video with Cycle-Gan-Depth-Model

[] deploy text2img env

[] familiar code and structure

[] find research of GAN image segmenation, organise state-of-art tech in semantic image segmentation

[] read [char-CNN-RNN](./pages/1605.05395.pdf)

---

### 12.10 Thu

**Schedule**

[x] read [Generative Adversarial Text to Image Synthesis](pages/1605.05396.pdf) [https://arxiv.org/abs/1605.05396](https://arxiv.org/abs/1605.05396)

[ ] > deploy text2img env -> try paper's model

**Tomorrow Plan**

[] deploy text2img env

[] familiar code and structure

[] find research of GAN image segmenation

[] read [char-CNN-RNN](./pages/1605.05395.pdf)

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