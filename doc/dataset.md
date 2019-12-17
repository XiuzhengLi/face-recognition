## 人脸识别数据集

### LFW

全称：Labeled Faces in the Wild

数据集链接：http://vis-www.cs.umass.edu/lfw/

数据集介绍：

​        LFW (Labeled Faces in the Wild) 人脸数据集是由美国马萨诸塞州立大学阿默斯特分校计算机视觉实验室整理完成的数据库，主要用来研究**非受限情况下的人脸识别**问题。其中提供的人脸图片均来源于生活中的自然场景，因此识别难度会增大，尤其由于多姿态、光照、表情、年龄、遮挡等因素影响导致即使同一人的照片差别也很大。并且有些照片中可能不止一个人脸出现，对这些多人脸图像仅选择中心坐标的人脸作为目标，其他区域的视为背景干扰。

​        LFW数据集共有13233张人脸图像，每张图像均给出对应的人名，共有5749人，且绝大部分人仅有一张图片。每张图片的尺寸为250X250，绝大部分为彩色图像，但也存在少许黑白人脸图片。

​        值得一提的是，在2015年，Google的FaceNet在LFW数据集上的平均分类精度就已经达到了99.63%，基本宣告了在LFW上性能竞赛的结束，所以现在LFW更多的是用来做人脸识别的验证和测试。

数据集信息：

- 13233张图像
- 5749人
- 1680人拥有两个或更多图像

[LFW数据集上的识别结果](http://vis-www.cs.umass.edu/lfw/results.html)



### PubFig

全称：Public Figures Face Database

数据集链接：http://www.cs.columbia.edu/CAVE/databases/pubfig/

数据集介绍：

​        PubFig数据集是一个大型的真实人脸数据集，这是哥伦比亚大学的公众人物脸部数据集，包含从互联网上收集的200人的58,797张图像，主要用于**非限制场景下的人脸识别**。与大多数其他现有的面部数据集不同，这些图像是在非合作对象的完全不受控制的情况下拍摄的。因此，姿势，照明，表情，场景，摄影机，成像条件和参数等都有很大差异。

数据集信息：

- 58797张图像
- 200人



### CFP

全称：Celebrities in Frontal-Profile in the Wild

数据集链接：http://www.cfpw.io/

论文：[Frontal to Profile Face Verification in the Wild](../literature/Frontal to Profile Face Verification in the Wild.pdf)

​        CFP数据集由500个identity的约共7000张图片组成，这个数据集的特别之处在于对于每个人，它有10张正面图像和4张侧面图像。

数据集信息：

- 7000张图像
- 500人



#### 参考链接：

https://zhuanlan.zhihu.com/p/54811743

https://zhuanlan.zhihu.com/p/31378836