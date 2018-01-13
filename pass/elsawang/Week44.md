## Work Diary

#### Week 44 of 2017

03/11 Tue

!!*张总：* 如果CeleA GAN 训练完成的话，请先 commit 保存docker， 然后按照issues中提到的教程训练 CeleA Encoder


主要工作：完成 DIGITS 中 MNIST GAN and Encoder 复现，训练 CeleA GAN
完成情况：完成
工作记录：
很顺利，代码问题根据昨天找到的解决方案解决。

明日计划安排：
完成 CeleA Encoder 训练
复现 CeleA Inference


02/11 Mon

主要工作：安装配置DIGITS_GAN
完成情况：完成
工作记录：
安装解决方案，下载其他人修改好的docker
image 地址
https://hub.docker.com/r/ai4sig/digits6mod/

使用说明
https://ai4sig.org/docs/get-modified-digits-docker/

本地安装DIGITS， caffe仍然有compile问题

其他进展：

根据post作者[教程](https://github.com/gheinrich/DIGITS-GAN/tree/master/examples/gan#handwritten-digits)学习使用DIGITS
DIGITS MNIST数据集分裂网络成功

MNIST GAN visualise 出现问题：
分析原因[code](https://github.com/gheinrich/DIGITS-GAN/blob/master/examples/gan/network-mnist.py)和digits使用的是TensorFlow 0.12， 版本叫老，`tf.concat` 命令顺序不同

明日计划安排：
1. 明天尝试按照 https://stackoverflow.com/questions/41813665/tensorflow-slim-typeerror-expected-int32-got-list-containing-tensors-of-type 或 https://github.com/carpedm20/DCGAN-tensorflow/issues/99 改code
2. 争取完成celeA教程

---

01/11 Mon

主要工作：安装Nvidia DIGITS
完成情况：未完成
主要原因：

1. local安装需要Caffe，两台计算机ubuntu及sdc，c++ compile过程均失败，无法安装Caffe，DIGITS不能运行
2. Docker官方环境安装成功，但缺乏gan module，尝试modified docker，下载失败。

明天计划：直接在官方docker下保存修改，调试encrytpoint。
