
# SECOND: Sparsely Embedded Convolutional Detection

！3D目标检测、3D稀疏卷积

2018\.8\.20

参考文章：https://blog\.csdn\.net/slamer111/article/details/127545423

## 论文动机

1\.现有方法大多将点云转换为2D的BEV或前视图表示，丢失了大量的空间信息

2\.VoxelNet提出了基于voxel的3D检测网络，但是其3D卷积部分运算量过大，实时性不佳

## 论文方法

1\.引入稀疏卷积替代原有的3D卷积，并进行改进

2\.提出了一种新颖的角度回归方法

3\.为点云检测引入了一种新的数据增强方法

## 网络框架

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NWRmODhiYTU0NGVhNzZkYTVhZmYzNmQ4MjExZmE0NjRfZjFkNjk2MmZiMGM1M2EzZTY2MmYyNTZhYzMwNjQxMjNfSUQ6NzY1OTY5NDU2ODg2MjEwODY1Ml8xNzgzNDI5MjcyOjE3ODM1MTU2NzJfVjM)

1\.将3D点云转换为体素特征和坐标

2\.应用两个VFE \(voxel feature encoding\)层和一个线性层（Voxel Feature Extractor

3\.a sparse CNN

4\.RPN生成detection

## 具体细节

### Point Cloud Grouping

点云是稀疏、不规则的3D数据，无法直接输入CNN。体素化就是把3D空间切分成小立方体（体素），让每个体素包含若干点，从而得到类似2D图像的"规则网格"结构。

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MTUyODYyODQwYjExYTk4YzRjOGI3NTNkOThjMGNiYmNfZGU5MDY2ZGMwOWJjZGU3ODlmNGQ3MjQwZDE5YjI1NGVfSUQ6NzY1OTY5OTA5NDQ0OTcyMDUxN18xNzgzNDI5MjcyOjE3ODM1MTU2NzJfVjM)

> 与普通图像不同的是，大多数三维点云的体素是空的，这使得三维体素中的点云数据通常是稀疏信号。

1\.首先对原始点云进行裁剪，只保留感兴趣的区域

2\.对3D网格进行划分（use a voxel size of vD = 0\.4 × vH = 0\.2 × vW = 0\.2 m）

3\.每个 voxel 保存最多T个点\(35或45\)

### Voxelwise Feature Extractor

利用voxel feature encoding \(VFE\) layer来提取voxelwise特征：

VFE层将**同一体素**中的所有点作为输入，并使用由线性层、批归一化（BatchNorm）层和整流线性单元（ReLU）层组成的全连接网络（FCN）来提取**逐点特征**。

然后，它使用elementwise max pooling来获得**每个体素的局部聚合特征**。最后，它将获得的特征平铺并将这些平铺特征和逐点特征连接在一起。我们使用VFE（C\_out）表示将输入特征转换为C\_out维输出特征的VFE层。类似地，FCN（C\_out）表示将输入特征转换为C\_out维输出特征的Linear\-BatchNorm\-ReLU层。

总体而言，体素特征提取器由几个VFE层和一个FCN层组成。

### Sparse Convolutional Middle Extractor（本文重点）

稀疏卷积算法

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NjBjNjRlZGJkOTI0YThlOGVmN2Q5Njg1NDUxMTMzNTNfOTEwMGM2YzcxZjVmYWNmYTRmMzllNWUwOTkzMGZhOGFfSUQ6NzY1OTcwODg0MjU0MjE4OTgxNV8xNzgzNDI5MjcyOjE3ODM1MTU2NzJfVjM)

稀疏卷积中间抽取器

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NGE4MjM1NTNiMjFlMGYwNjk0OWZhNmZkY2RmMDFiYmFfNWE2NWQ1NTc3MzM5NmYzNzVmODg2MmE0MDMxYzRjMGJfSUQ6NzY1OTcwOTIxNTk1NTQzODc5OV8xNzgzNDI5MjcyOjE3ODM1MTU2NzJfVjM)

（感觉不是当前的研究方向，以后有需要再看）

详细讲解：https://zhuanlan\.zhihu\.com/p/382365889

## 损失函数

当同一个3D检测框的预测方向恰好与真实方向相反的时候，预测的前6个变量的回归损失较小，而最后一个方向的回归损失会很大，这其实并不利于模型训练。

为了解决这个问题，作者引入角度回归的正弦误差损失，当方向相反时，角度误差很小，但我们又加入了方向分类器softmax。

分类用**Focal loss**

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YWQxNjYwYmNiMGQ3NjI0Y2MzZWNjZjYxNjMzYjk0YzJfZGQ5NTdiYWY3ZTMzZjBjMzZlMzViYjQxMWQxYzI1YzZfSUQ6NzY1OTcxMTExOTExMjExMzM3NF8xNzgzNDI5MjcyOjE3ODM1MTU2NzJfVjM)

回归用**smoothL1**（新的角度损失回归）

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YjdkNDVkNTVlYWJiNzAxZWE4ZTI0OGE4NzQzMDk0MmVfYmM3MDZmZGNiMmU1Yjk3NWQxZTg4OWQxZThlYTE1NmZfSUQ6NzY1OTcxMDkzNTAxMTM4MDQ0Ml8xNzgzNDI5MjcyOjE3ODM1MTU2NzJfVjM)

**总的loss：**

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YTdlNzQ2OWU3MjA3MmQ3NWZmMjA0M2YxOGI1YjMzMTVfMzBiYmNlYjExMGNjNWJkMjcyOTlkZjlhMTBlYzhhMjFfSUQ6NzY1OTcxMTM1MTY2MTg3NDQyM18xNzgzNDI5MjcyOjE3ODM1MTU2NzJfVjM)

Lcls是分类损失，Lreg−other是位置和维度的回归损失，Lreg−θ是我们的新角度损失，Ldir是方向分类损失。

## 实验

KITTI dataset（包括car,pedestrain,cyclist）

对于每个类别，检测器都被评估为三个难度级别：easy, moderate and hard\.

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ZGRhODEzNDM3ZDQ1NzQyYTVlZTk1MThlMWE1MDFlMmJfNmFhMzA3YThhOGNhOWI2MzA4YzM1NTc4ZTc3ZGQ1NGZfSUQ6NzY1OTcxNDUyOTYwMTMwOTkxNl8xNzgzNDI5MjcyOjE3ODM1MTU2NzJfVjM)
