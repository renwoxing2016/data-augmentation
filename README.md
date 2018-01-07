# data-augmentation
data augmentation on python
为了扩增数据集，采用了2种方式来进行数据的扩增。

1、使用keras的数据增强处理
  见x-augmentation.py
2、使用skimage的数据增强处理
  见common-augmentation.py

keras包括的处理，有featurewise视觉上图像会稍微变暗，samplewise视觉上图像会变成类x光图像形式，zca处理视觉上图像会变成灰白图像，rotation range 随机旋转图像，水平平移,垂直平移，错切变换，图像缩放，图片的整体的颜色变换，水平翻转操作，上下翻转操作， rescale。
存在的问题是图像的变换是随机的，即有的图像可能不会变换。

skimage的数据增强处理，有resize， gray，rescale，noise，flip，rotate， shift， zoom，gaussian zoom，shear，contrast，channelshift，PCA，polar。

每个文件后面都有各个函数的测试使用例子。
便于使用。
