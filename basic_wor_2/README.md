## 作业描述
请参考 MMDetection 文档及教程，基于自定义数据集 balloon 训练实例分割模型，基于训练的模型在样例视频上完成color splash的效果制作，即使用模型对图像进行逐帧实例分割，并将气球以外的图像转换为灰度图像。

## 实验设备
GTX2060

## [Balloon数据集](https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip)
balloon是带有mask的气球数据集，其中训练集包含61张图片，验证集包含13张图片。

## 使用mask rcnn进行实例分割
ballon2coco.py为数据集转换处理
color_splash.py为color_splash特效

- 模型地址

链接：https://pan.baidu.com/s/1kaF1v1riJt_EYS2FIUaCgg?pwd=qpnt 
提取码：qpnt 


|            Model             | bbox_map | bbox_map@50 | bbox_map@75 |                   Config                   |                             Download                              |
|:----------------------------:|:--------:|:-----------:|:-----------:|:------------------------------------------:|:-----------------------------------------------------------------:|
| mask_rcnn(r50_fpn_1x_ballon) |  0.688   |    0.808    |    0.808    | [config](./mask_rcnn_r50_fpn_1x_ballon.py) | [model](https://pan.baidu.com/s/1kaF1v1riJt_EYS2FIUaCgg?pwd=qpnt) |


- 效果图

![效果图](./color_splash_1.gif)