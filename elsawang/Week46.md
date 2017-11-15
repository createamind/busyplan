## Work Diary Week 46

### 15/11 Wed
- 熟悉 pytorch 框架
- 仔细阅读 pix2pix pytorch 代码，确定 3D API 代码位置
- 基本确定几个3D卷积API可以直接更换使用
- 问题，数据输入。原始代码输入 2D 图片，两个方案：
 1. 抛弃原始数据输入接口，重新编写 3D 视频接口
 2. 在原始接口上更改


### 14/11 Tue

- 调试 Keras 3D Unet，可以运行
- 下午安装 Keras 插件过程中出现于 TensorFlow 1.4 不兼容清空
- 明天尝试在 已有 TensorFlow 版本 pix2pix 基础上进行 3D 修改
