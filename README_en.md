# Fish fry individual detection and counting based on improved YOLOv5s.

## Download the dataset
The dataset has been uploaded to Kaggle. On Kaggle, search for "Fish_Count".

![image](https://github.com/user-attachments/assets/521ed58c-797e-4f29-977d-0d4112f2d71a)

After clicking on it, just download it.

Unzip the downloaded dataset and rename it as "dataset_pro".

## Project Directory Structure and Introduction

### dataset_pro 
Dataset directory
Just extract the dataset_pro.tar.gz file and you will get it.

#### train 

Training set directory

#### val

Validation set directory

### yolov5  yolov5 project

This directory contains the code of the YOLOv5 project for this research.

- The code for the SA attention mechanism is in `yolov5/models/Sa.py`.
- The code for the Bam attention mechanism is in `yolov5/models/Bam.py`.
- The code for MDPIoU is in the `bbox_iou` method in `yolov5/utils/metrics.py`.
- The dataset augmentation code is in `DataAugForObjectDetection`.
- The improved model configuration file is in `yolov5/models/yolov5s.yaml`.
- The dataset configuration file is in `yolov5/data/custom_data.yaml`.
- The training file is in `yolov5/train.py`.
- The validation file is in `yolov5/val.py`.
- The pre-trained model is in `yolov5/weight/yolov5s.pt`.


## Recreate the experiment

### 1.Environment

Create a new Python environment

`conda create -n yolov5 python=3.8`

Enter this environment

`conda activate yolov5`

Install PyTorch

`pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html`

stall the Python dependencies for yolov5

- Enter the directory of yolov5.

- `pip install -r requirements.txt`

> If there is a package missing during subsequent running, you can use `pip install` to download the complete package.

### 2.Modify the configuration file of the dataset, namely custom_data.yaml.

```yaml

# Custom data for safety helmet
train: The path of the images in the training set
val: The path of the validation set

# Classes
nc: 1  # number of classes
names: ['fish'] # class names

```

- Change the paths in train to the paths of the training set images (it would be better to use absolute paths)

- Change the path in val to the path of the validation set images (it is better to use an absolute path)


### 3.Modify the model configuration file yolov5s.yaml

#### 3.1 When the file yolov5s.yaml contains the following content, it indicates that the Sa and Bam modules are not to be used.

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



#### 3.2 When the file yolov5s.yaml contains the following content, it indicates the use of the Sa module and the non-use of the Bam module.

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



#### 3.3 When the file yolov5s.yaml contains the following content, it indicates that the Bam module is used and the Sa module is not employed.

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



#### 3.4 When the file yolov5s.yaml contains the following content, it indicates the use of the Bam module and the Sa module.

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



### 4.Whether to use MDPIoU

In the `yolov5/utils/loss.py` file, locate the line `iou = bbox_iou(pbox, tbox[i], MDPIoU=True, hw=tobj.size()[2] * tobj.size()[3]).squeeze()` and modify it.

If MDPIoU is not needed, then remove the following part. 

`iou = bbox_iou(pbox, tbox[i], MDPIoU=True, hw=tobj.size()[2] * tobj.size()[3]).squeeze()`


Please modify as follows:

`iou = bbox_iou(pbox, tbox[i], CIoU=True, hw=tobj.size()[2] * tobj.size()[3]).squeeze()`


If necessary, then modify it accordingly. 

`iou = bbox_iou(pbox, tbox[i], MDPIoU=True, hw=tobj.size()[2] * tobj.size()[3]).squeeze()`


5. Modify train.py 

- `--weights` should be modified to the path of the pre-trained model `yolov5s.pt` (it is preferable to use an absolute path) 

- `--cfg` Modify the path of the configuration file `yolov5s.yaml` for improving the model (it is recommended to use an absolute path) 

- `--data` should be modified to the path of the dataset configuration file `custom_data.yaml` (it is preferable to use an absolute path) 

- Change `--batch-size` to 16, which corresponds to the experimental condition in the paper where the "training batch size is 16". 

- Change `-epochs` to 200, which corresponds to the experimental condition of "200 iterations" in the paper. 

After the modification is completed, simply execute python train.py to start the training process.


6. Modify val.py 

- `--data` should be modified to the path of the dataset configuration file `custom_data.yaml` (it is preferable to use an absolute path) 

`--weights` should be modified to the path of the trained model `best.pt`, which is usually located in `yolov5/run/train/exp/weight/best.pt` (it is recommended to use an absolute path) 

Executing `python val.py` will start the verification process. 

> If there is a discrepancy between the final training outcome and the content of the paper, you can use val.py to verify the model that has been trained successfully.

## Note
### The final model trained by us is as follows 

`yolov5/weight/improve.pt`


val.py can be used for verification. 

The original model trained is `yolov5/weight/original.pt` 

## Graphical User Interface 

The graphical interface is implemented with PyQt5. The script is located in `yolov5/pyqt2.py`. Just run it.
> You can modify this part to select a different model. `self.model = attempt_load('weight/original.pt', map_location=self.device).fuse().eval()  # Modify the path of the model file`. If you don't modify it, it will be the optimal model for this experiment.
