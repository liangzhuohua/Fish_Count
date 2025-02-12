# 基于改进YOLOv5s的鱼苗个体检测与计数研究

## 下载数据集
数据集已经上传到Kaggle，在Kaggle中，搜索Fish_Count。 

![image](https://github.com/user-attachments/assets/521ed58c-797e-4f29-977d-0d4112f2d71a)

点进去后，下载即可 

![image](https://github.com/user-attachments/assets/a5068f98-e5a9-42af-a410-5634579ff8a0)

把下载下来的数据集进行解压，并且重命名为dataset_pro

## 项目目录结构与介绍

### dataset_pro 
数据集目录
从Kaggle上下载解压即可获得。

#### train 

训练集目录

#### val

验证集目录

### yolov5  yolov5项目  

里面是本研究的yolov5项目的代码

- SA 注意力机制的代码在`yolov5/models/Sa.py`中

- Bam注意力机制的代码在`yolov5/models/Bam.py`中

- MDPIoU的代码在 `yolov5/utils/metrics.py` 中的 `bbox_iou` 方法中

- 数据集增强代码在 `DataAugForObjectDetection` 中

- 改进模型的配置文件在 `yolov5/models/yolov5s.yaml` 中

- 数据集配置文件在 `yolov5/data/custom_data.yaml` 中

- 训练文件在 `yolov5/train.py` 中

- 验证文件在 `yolov5/val.py` 中

- 预训练模型在 `yolov5/weight/yolov5s.pt` 中
## 如何复现

### 1.环境搭建

新建python环境

`conda create -n yolov5 python=3.8`

进入该环境

`conda activate yolov5`

安装pytorch

`pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html`

安装yolov5的python依赖包

- 进入到yolov5的目录中

- `pip install -r requirements.txt`

> 如果后续运行时缺包，使用`pip install`下载完整即可

### 2.修改数据集配置文件 custom_data.yaml

```yaml

# Custom data for safety helmet
train: F:/源代码和数据集/Finsh/dataset/train/images
val: F:/源代码和数据集/Finsh/dataset/val/images

# Classes
nc: 1  # number of classes
names: ['fish'] # class names

```

- 把train中的路径修改成训练集图片的路径（最好使用绝对路径）

- 把val中的路径修改成验证集图片的路径（最好使用绝对路径）


### 3.修改模型配置文件 yolov5s.yaml

#### 3.1 当 yolov5s.yaml 为

```yaml
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

# Parameters
nc: 1  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv5 v6.0 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, C3, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 6, C3, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, C3, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 3, C3, [1024]],
   [-1, 1, SPPF, [1024, 5]],  # 9
  ]

# YOLOv5 v6.0 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, C3, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, C3, [256, False]],  # 17 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)

   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]

```

的时候，则表示不使用Sa和Bam模块

#### 3.2 当 yolov5s.yaml 为

```yaml
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

# Parameters
nc: 1  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv5 v6.0 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, C3, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 6, C3, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, C3, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 3, C3, [1024]],
   [-1, 1, SPPF, [1024, 5]],  # 9
   [-1,1,Sa,[]],
#    [-1,1,Bam,[16]],
  ]

# YOLOv5 v6.0 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, C3, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, C3, [256, False]],  # 17 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)

   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```

表示使用Sa模块，不使用Bam模块

#### 3.3 当 yolov5s.yaml 为

```yaml
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

# Parameters
nc: 1  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv5 v6.0 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, C3, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 6, C3, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, C3, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 3, C3, [1024]],
   [-1, 1, SPPF, [1024, 5]],  # 9
#    [-1,1,Sa,[]],
   [-1,1,Bam,[16]],
  ]

# YOLOv5 v6.0 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, C3, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, C3, [256, False]],  # 17 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)

   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```

表示使用Bam模块，不使用Sa模块

#### 3.4 当 yolov5s.yaml 为

```yaml
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

# Parameters
nc: 1  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv5 v6.0 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, C3, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 6, C3, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, C3, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 3, C3, [1024]],
   [-1, 1, SPPF, [1024, 5]],  # 9
   [-1,1,Sa,[]],
   [-1,1,Bam,[16]],
  ]

# YOLOv5 v6.0 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, C3, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, C3, [256, False]],  # 17 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)

   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```

表示使用Bam模块和Sa模块

### 4.是否使用MDPIoU

在 `yolov5/utils/loss.py`文件中，找到 `iou = bbox_iou(pbox, tbox[i], MDPIoU=True, hw=tobj.size()[2] * tobj.size()[3]).squeeze()` 并修改它。

- 如果不需要MDPIoU，则把 

`iou = bbox_iou(pbox, tbox[i], MDPIoU=True, hw=tobj.size()[2] * tobj.size()[3]).squeeze()` 

修改为

`iou = bbox_iou(pbox, tbox[i], CIoU=True, hw=tobj.size()[2] * tobj.size()[3]).squeeze()` 

- 如果需要，则修改为

`iou = bbox_iou(pbox, tbox[i], MDPIoU=True, hw=tobj.size()[2] * tobj.size()[3]).squeeze()` 




### 5.修改train.py

- `--weights` 修改为预训练模型`yolov5s.pt`的路径 （最好使用绝对路径）

- `--cfg` 修改为改进模型的配置文件`yolov5s.yaml`的路径 （最好使用绝对路径）

- `--data` 修改为数据集配置文件`custom_data.yaml`的路径 （最好使用绝对路径）

- `--batch-size` 修改为16，对应论文中的 `训练批大小为16` 的实验条件

- `-epochs` 修改为200，对应论文中的 `迭代次数200` 的实验条件

修改完成后，执行 python train.py即可开始训练。

### 6.修改val.py

- `--data` 修改为数据集配置文件`custom_data.yaml`的路径 （最好使用绝对路径）

- `--weights` 修改为训练好的模型`best.pt`的路径，一般是在`yolov5/run/train/exp/weight/best.pt`下 （最好使用绝对路径）

执行python val.py即可开始验证。

> 若最后训练出来的结果与论文有差异，可以使用val.py来验证我们已经训练好的模型。

## 注意
### 我们这边训练出来的最终模型为

`yolov5/weight/improve.pt`

可以使用val.py来进行验证

> 训练出来原始的模型为 `yolov5/weight/original.pt`

### 图形化界面

图形化界面使用的是pyqt5, 脚本在 `yolov5/pyqt2.py` 中，运行即可。
> 可修改此处，选用不同的模型 `self.model = attempt_load('weight/original.pt', map_location=self.device).fuse().eval()  # 修改模型文件路径`, 不修改则为本实验的最优模型。
