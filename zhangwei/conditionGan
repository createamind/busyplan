
在
image-to-image 的图像生成的问题中采用conditional adversarial networks

训练图像到图像的映射
学习该映射的目标loss function，通用的方法，避免在不同问题中手工定制loss
1.introduction

CNN的传统loss function，prediction 和 ground truth 之间的像素级的欧式距离，对每个像素误差取平均值，将会导致生成的图像模糊。如何使cnn学到你想要的信息，仍然是个开放的问题，需要高超的技巧。

所以生成图像的问题中最好能抛开像素级误差，使用更高层次的目标，例如’和真实图像无法区分‘，自动学习满足这一目标的损失函数。GAN中的判别器和生成器结构，判别器可以为生成器提供gradient，使其能够调整每一像素值，产生更真实的图像。

为了将GAN应用到图像翻译的问题中，还需要引入conditional setting，根据输入的图像生成对应的另一领域的图像。

2.相关工作

2.1Structured losses

传统的Image-to-image translation 通常定义为像素级的分类或者回归 。

这将忽略输出的图像空间的结构信息，在给定输入图像后，输出的图像每个像素相对其他像素都是条件独立的。

Conditional GANs将会学习structured loss. Structured losses 惩罚输出像素间的条件概率分布。

Structured losses

conditional random fields ，Semantic image segmentation with deep con- volutional nets and fully connected crfs
the SSIM metric ，Image quality assessment: from error visibility to struc- tural similarity
feature matching ，Generating images with per- ceptual similarity metrics based on deep networks
nonparametric losses，Combining markov random fields and convolutional neural networks for image synthesis.
the convolutional pseudo-prior，Top-down learning for struc- tured labeling with convolutional pseudoprior.
losses based on matching covariance statistics，Perceptual losses for real-time style transfer and super-resolution
2.2 Conditional GANs

discrete labels，Conditional generative adversar- ial nets
text，Generative adversarial text to image synthesis
inpainting，Feature learning by inpainting
image prediction from a normal map，Generative image modeling using style and structure adversarial networks
image manipulation guided by user constraints
future frame prediction
future state prediction
product photo generation
and style transfer
2.3 网络结构

生成器采用U-Net，U-net: Convolu- tional networks for biomedical image segmentation

判别器采用“PatchGAN” ，Precomputed real-time texture synthe- sis with markovian generative adversarial networks.

针对输出图像每个局部分块判断惩罚其structure loss

3.问题定义

传统的GAN，将从正泰分布采样的noise z，映射到输出的图像y ，G ：z ->y

conditional GAN 还会观察到原始图像 x，在这个条件下降 x，z 映射到输出图像y，G: (x,z)->y,



4.优化目标



G tries to minimize this objective against an adversarialDthat tries to maximize it。

结合传统的L1和L2loss，有利于保持生成的图像和输入图像的相似性。L1比L2更利于生成清晰的图像。



在没有z的条件下，仍然可以学习到x到y的映射，但是这个映射是决定性的，无法逼近目标的概率分布。

但是在试验中，z一般被当做噪音忽略掉了，输出的图像只有很小的随机性。

在本文提出的模型中，采用dropout noise的方式将噪音加到某些layer上，train和test的过程都要加。

Designing conditional GANs that produce stochastic out- put, and thereby capture the full entropy of the conditional distributions they model, is an important question left open by the present work.

5.网络结构

5.1 生成器，在DCGAN的基础上优化设计

图像到图像的翻译问题，可以看做同样的结构有不同的渲染纹理，输入图像的结构特征和输出图像的结构特征应该是对齐的，并且能共享。

在传统的自动编码器的瓶颈网络结构中，所有的信息流经每一层向后传输，可以采用skip shortcut将编码层和解码器层直接连接。



5.2 判别器 Markovian discriminator (PatchGAN)

L1，L2 loss 会使输出的图像平均像素重建误差更小，但是更模糊。

GAN捕获高频特征，L1loss捕获低频特征。

高频特征主要是局部的结构特征，可以将图片分块，分布判断每一块图片的局部纹理结构特征的真实度。最终的真实度为每块的平均值。

本质上是一个全卷积网络。

将图像建模为 Markov random field, 假设xiang每一像素的只和patch内的像素相关，相对patch之外的像素独立。

6.训练和预测

训练过程中batchsize设置为1，instance normalization，

《实例正则化：快速风格化缺失的成分》

http://www.jianshu.com/p/d77b6273b990

一个简单的观测是：一般来说，图像风格化的结果不应该取决于内容图的对比度（见fig 2）事实上，风格上的差距被设计用来将元素从风格图转移到内容图，因此风格化后图的对比度应该是近似于风格图的对比度的。因此，生成器网络必须忽视内容图的对比度信息，问题就在于，对比度正则化是不是可以有效，通过结合在标准的CNN块中，或者说直接实现在结构中。

实例正则化能消除内容图的对比度的影响？？
预测过程中要使用dropout，和batchnormal
