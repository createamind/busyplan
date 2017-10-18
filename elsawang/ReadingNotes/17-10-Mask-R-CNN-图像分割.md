### **paper 阅读笔记**

***17-10-Mask-R-CNN-图像分割***

Speed 5 fps

#### Mask FCN

> The mask branch is a small FCN applied to each RoI, predicting a segmentation mask in a pixel-to- pixel manner.

像素点输出binary

L = Lcls + Lbox + Lmask

Lmask 像素级交叉熵

> we predict an m * m mask from each RoI using an FCN





#### RoIAlign

> To fix the misalignment, we pro- pose a simple, quantization-free layer, called RoIAlign, that faithfully preserves exact spatial locations.
> ...preserve the explicit per-pixel spatial correspondence
> bilinear interpolation to compute the exact values of the input features at four regularly sampled locations in each RoI bin, and aggregate the result(using max or average)


#### Class-Agnostic Masks

> predicting a single m*m output regardless of class


 

