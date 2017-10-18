1.Temporal Generative Adversarial Nets with Singular Value Clipping

从随机采样的z0生成整个视频序列
A uniform distribution is used to sample z0.
- temporal generator ，一维反卷积，将初始隐变量映射到一个隐变量序列，G0:z0-->{z1,z2,z3,z4....zn}
- image generator 一维向量到二维图像反卷积，G1:(z0,z1)->frame1,(z0,z2)->frame2...
- 生成的video序列可表示为[G1(z0,z1),G1(z0,z2),G1(z0,z3),G1(z0,z4),G1(z0,z5)...G1(z0,zt)]
- 真实的video序列可表示为[x1,x2,....xt]
- 判别器spatio-temporal 3D convolu- tional layers，判别两个视频序列的真假

