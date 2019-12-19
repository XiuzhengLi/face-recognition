## Face recognition

介绍：多姿态人脸识别研究

#### [相关概念](./doc/concept.md)

- 人脸识别的相关概念、专业术语以及评价指标。

#### [研究背景](./doc/background.md)

- 人脸识别的发展历程、应用领域、国内外研究现状。

- 几种在LFW数据集上表现较好的人脸识别方法分析。

#### [实现思路](doc/TP_GAN+Res2Net.md)

多姿态人脸识别研究的难点主要在于：

1. 通过不同角度的侧脸合成正脸图像
2. 正脸图像的特征训练及预测

这两个问题分别可以使用**生成模型**和**判别模型**来解决。

近年来，人脸识别中GAN和CNN占据这两类模型的主流。

生成模型：

- DR-GAN(CVPR 2017)
- FaceID-GAN(CVPR 2018)
- TP-GAN(ICCV 2018)

判别模型：

- ResNet(CVPR 2016)

  提出了「恒等快捷连接」(identity shortcut connection)，直接跳过一个或多个层来以应对梯度消失问题，使更深层的CNN成为可能。

- ResNeXt(CVPR 2017)

  引入了叫作「基数」(cardinality)的超参数，指独立路径的数量，这提供了一种调整模型容量的新思路。

- Res2Net(2019.9)

  使用分块再拼接的策略，使卷积更有效地处理特征。

暂定使用TP-GAN(ICCV含金量更高)或DR-GAN(有实现代码)和Res2Net(虽然变革性不如其他两个，但模型更新，今年9月分刚发布了预印版文献)相结合。

#### [数据集](./doc/dataset.md)

- LFW(Labeled Faces in the Wild)

  人脸识别领域很有影响力的数据集，但早在几年前，在LFW数据集上的平均识别准确率就已经达到了99%，很难有所提升，但其中有很多值得借鉴的方法。数据集可以用来做模型测试。

- PubFig(Public Figures Face Database)

  同样是用于非限制场景下的人脸识别，比LFW数据集更大，可用来做面部识别模型的训练。

- CFP(Celebrities in Frontal-Profile in the Wild)

  特别之处在于每个人包含10张正脸照与4张侧脸照，可用来做生成模型的训练。

#### [开发环境](doc/dev_setting.md)

- 开发所需软、硬件(主要是树莓派)

#### [环境搭建](doc/setting_up.md)

- 操作系统、python、opencv等安装运行。

#### [代码测试](doc/test.md)

- 人脸检测、人脸特征提取代码测试。