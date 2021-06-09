## Model effect

YOLOv5 is divided into four models YOLOv5s, YOLOv5m, YOLOv5l, and YOLOv5x according to their size. The performance of these four models is shown in the figure below:

![yolo](https://figure.cruisetian.top/img/1622024144136-yolo.png)

The above picture shows the average end-to-end time of each image when inference is based on 5000 COCO val2017 images, batch size = 32, GPU: Tesla V100, this time includes image preprocessing, FP16 inference, post-processing and NMS (non-extreme Large value suppression). EfficientDet's data is obtained from the [google/automl](https://github.com/google/automl) (batch size = 8).

## Requirements

Python 3.8 or later with all [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) dependencies installed, including `torch>=1.7`. To install run:

```bash
$ pip install -r requirements.txt
```

You can also configure the mirror source if the pip installation is slow, the following is the mirror source of Tsinghua University.

```bash
$ pip install pip -U
$ pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

**Note:** The Windows version of using pip to install pycocotools requires a C++ compilation environment, or you can install `pycocotools-windows` using the flowing command:

```bash
$ pip install pycocotools-windows
```

and then change `pycocotools` to `pycocotools-windows` in the `requirements.txt`. Or just leave it if you don't use coco datasets.

## How To Run

### Training

#### 1. Prepare data

The yolo format label is a txt format file, and the file name is the same as the corresponding picture name, except that the suffix is changed to .txt. 

Images and labels should be stored separately. You need to create two new folders, one is called `images` to store pictures, the other is called `labels` to store label txt files, and the training, validation and testing folders should be created if needed.

#### 2. Prepare yaml file

You need to modify the .yaml file for your own training, one is the model file (optional), and the other is the data file.

- model file (optional): You can directly modify the `yolov5s.yaml` / `yolov5m.yaml` / `yolov5l.yaml` / `yolov5x.yaml` file in `./models` according to the model you choose to train. You only need to use `nc: 80` The 80 in is modified to the number of categories in your data set. Others are the model structure and do not need to be changed.
- Data file: Make your own data file based on the coco data file in the `./data` folder, define the training set, validation set, and test set paths in the data file; define the total number of categories; define the category name.

#### 3. Train

For training, run `train.py` directly, then add instruction parameters as needed, `--weights` to specify the weight, `--data` to specify the data file, `--batch-size` to specify the batch size, `-- epochs` specifies epochs, `--img-size` specifies the resolution of the picture, the default is 640, which can be abbreviated as `--img`. The training results will be saved in `./runs/train` by default. A simple training sentence:

```bash
# Use the yolov5s model to train the coco128 data set for 5 epochs, picture resolution is set to 640 and the batch size is set to 16
$ python train.py --batch 16 --img 640 --epochs 5 --data ./data/coco128.yaml --weights yolov5s.pt
```

### Detection

For detection, run `detect.py`directly, then add instruction parameters as needed, `--weights` to specify the weight, `--source` to specify the directory or file to be detected, `--conf` to specify detection confidence threshold, `--img` specifies the resolution of the detection picture. If the weight is not specified, the default COCO pre-training weight model will be automatically downloaded. The detection results will be saved in `./runs/detect` by default. A simple detection sentence:

```bash
# Use the weight yolov5s.pt to detect all media in the ./data/images folder, picture resolution is set to 640 and set the detection confidence threshold to 0.5
$ python detect.py --img 640 --source ./data/images/ --weights yolov5s.pt --conf 0.5
```

### Test

For test, run `test.py`directly, then add instruction parameters as needed, `--weights` to specify the weight, `--data` to specify the data file, `--img` specifies the resolution of the picture. The detection results will be saved in `./runs/test` by default. A simple detection sentence:

```bash
# Use the weight yolov5s.pt to test the test set defined in ./data/coco.yaml and picture resolution is set to 640
$ python test.py --weights yolov5s.pt --data ./data/coco.yaml --img 640
```

