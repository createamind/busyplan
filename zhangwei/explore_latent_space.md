Latent Space Oddity: on the Curvature of Deep Generative Models
探索隐变量空间分布
Deep generative models provide a systematic way to learn nonlinear data distributions, through a set of latent variables and a nonlinear “generator” function that maps latent points into the input space. The nonlinearity of the generator imply that the latent space gives a distorted view of the input space.

Since the generator defines a surface in the input space, we can alternatively seek the shortest curve along this surface that connects the two points; this is perhaps the most natural choice of interpolant.

如果要找到图像特征的渐进性变化和z变量的关系，即在隐变量空间找到两点之间的最短路径。确定两张只有一个特征变化的图像和对应的z，当沿着这条路径采样得到一系列，生成的图像将会是渐进变化的。

如果假设隐变量空间是个欧式空间，两点之间的最短路径是两点之间的直线。
由插值实验的不同方法和产生的图像变化效果可以得出一个结论，线性插值容易产生奇怪的图像，并且变化并不平滑。

最常采用的插值方法是球形插值，这在尼曼几何空间是符合最短路径的定义，两点之间最短路径并不是直线，而是一条曲线。

这就意味特征的变化不能由z的的线性变化产生，对于不同的图像相对于它的特征变化方向是不同的，两者之间存在的是非线性的关系。
