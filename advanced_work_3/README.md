## 作业描述
使用mmseg进行clothes数据集分割(效果很差 不太懂为啥)

## 实验设备
GTX3090

## [People Clothing Segmentation数据集](https://www.kaggle.com/datasets/rajkumarl/people-clothing-segmentation)


## 使用deeplabv3plus进行分割
- 模型地址
链接：https://pan.baidu.com/s/1o6W6MVzVQt6mNDrfAngSzw?pwd=yltg 
提取码：yltg 

- 模型指标

|       Model        | mIOU  | mACC  |                             Config                              |                              Download                              |
|:------------------:|:-----:|:-----:|:---------------------------------------------------------------:|:------------------------------------------------------------------:|
| deeplabv3plus(R50) | 0.133 | 0.197 | [config](./deeplabv3plus_r50-d8_4xb8-160k_clothes58-512x512.py) | [model](https://pan.baidu.com/s/1o6W6MVzVQt6mNDrfAngSzw?pwd=yltg ) |

|    Class    |  IoU  |  Acc  |
|:-----------:|:-----:|:-----:|
|    null     | 32.01 | 63.98 |
| accessories | 39.04 | 58.07 |
|     bag     | 8.42  | 10.21 |
|    belt     | 6.59  | 8.02  |
|   blazer    | 17.87 | 34.4  |
|   blouse    | 0.28  | 0.37  |
|  bodysuit   | 10.73 | 13.21 |
|    boots    |  0.0  |  0.0  |
|     bra     | 2.79  | 3.12  |
|  bracelet   | 2.63  | 2.92  |
|    cape     | 0.03  | 0.03  |
|  cardigan   | 0.01  | 0.01  |
|    clogs    | 34.58 | 73.06 |
|    coat     | 42.5  | 73.97 |
|    dress    | 0.04  | 0.04  |
|  earrings   |  0.0  |  0.0  |
|    flats    | 0.62  | 0.63  |
|   glasses   | 1.18  | 1.19  |
|   gloves    | 65.32 | 91.66 |
|    hair     | 30.63 | 36.95 |
|     hat     |  0.0  |  0.0  |
|    heels    |  0.0  |  0.0  |
|   hoodie    | 0.01  | 0.01  |
|  intimate   | 27.51 | 48.15 |
|   jacket    | 41.23 | 64.94 |
|    jeans    | 0.13  | 0.14  |
|   jumper    | 0.12  | 0.13  |
|  leggings   | 0.01  | 0.01  |
|   loafers   | 3.33  | 3.58  |
|  necklace   | 0.05  | 0.05  |
|   panties   | 55.66 | 77.66 |
|    pants    | 0.06  | 0.06  |
|    pumps    | 8.19  | 12.15 |
|    purse    |  0.0  |  0.0  |
|    ring     |  0.0  |  0.0  |
|   romper    | 0.91  | 1.21  |
|   sandals   | 17.91 | 21.64 |
|    scarf    | 23.67 | 29.1  |
|    shirt    | 47.53 | 73.78 |
|    shoes    | 12.08 | 14.01 |
|   shorts    | 70.64 | 93.82 |
|    skin     | 6.93  | 7.93  |
|    skirt    | 0.03  | 0.03  |
|  sneakers   |  7.6  | 8.48  |
|    socks    | 37.25 | 52.61 |
|  stockings  | 31.88 | 43.61 |
|    suit     | 32.68 | 42.95 |
| sunglasses  | 22.27 | 33.85 |
|   sweater   | 0.39  | 0.59  |
| sweatshirt  | 0.02  | 0.02  |
|  swimwear   | 30.63 | 45.93 |
|   t-shirt   | 7.46  | 7.76  |
|     tie     |  0.0  |  0.0  |
|   tights    | 0.43  | 0.55  |
|     top     | 3.75  | 4.13  |
|    vest     | 0.04  | 0.04  |
|   wallet    |  0.0  |  0.0  |
|    watch    |  0.0  |  0.0  |
|   wedges    |  0.0  |  0.0  |