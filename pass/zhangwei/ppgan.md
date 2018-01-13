- 1.研读nvidia生成模型论文
https://github.com/createamind/busyplan/blob/master/zhangwei/nvidia-gan-paper.md
- 2.收集1024*1024图像，由于图像较大，模拟器运行产生的数据占用很大的内存又无法自动释放，最后内存爆满。
- 3.考虑分段运行模拟器，或者查找出问题的代码。

1.训练数据集准备，按要求格式输出H5文件
2.theano编译成功，运行时出现shape mis-match的错误
3.检查输入数据格式没问题，需要继续深入代码，或者换已有的正常运行的代码测试
