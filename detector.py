import time
from pathlib import Path
import os
import torch
from numpy import random
from models.experimental import attempt_load
from models.resnet import resnet_model
from utils.datasets import  LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
     set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor,wait

# To anyone reading this code, feel free to contact rikosellic@github!!
# Introduction of this class can be found at
# https://github.com/SRT-Autonomous/SRT2021/blob/perception_wxy/src/perception/README.md
# To train the weights of yolov5 and resnet, use the exact code in
# YOLOv5: https://github.com/ultralytics/yolov5
# Keypoint regression: https://github.com/Srijan2001/Keypoint-regression/blob/master/Keypoint_regression.ipynb

IMG_SIZE=1280  # yolov5识别图片大小
CONF_THRES=0.7  # yolo置信度阈值
IOU_THRES=0.45
SOURCE='test' # yolo测试图片目录
DEVICE='' # torch设备
AUGMENT=False
CLASSES=None
AGNOSTIC_NMS=False

# This function is copied from yolov5 code
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

# Cone detection class
# To train yolov5 and resnet, use the exact code
# The resnet structure is simplified!
#  You may add X = identity_block(X, 64 * 4)  X = identity_block(X, 64 * 8)
#  in models.resnet.py for better performance (Do not forget the change when you train  resnet!)
class Detector(object):
    def __init__(self,yolo_weights='weights/yolo.pt',resnet_weights='weights/very_simple_resnet.hdf5'):
        # Initialize
        set_logging()
        self.device = select_device(DEVICE)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.imgsz = IMG_SIZE

        # Load model
        # yolov5
        self.model = attempt_load(yolo_weights, map_location=self.device)  # load FP32 model
        self.imgsz = check_img_size(self.imgsz, s=self.model.stride.max())  # check img_size
        if self.half:
            self.model.half()  # to FP16

        # resnet
        if resnet_weights != None:
            # This net structure is simplified!  Details  in models.resnet.py
            self.resnet = resnet_model((80, 80, 3))
            self.resnet.load_weights(resnet_weights)
        else:
            # If no resnet model, only detect_bounded_box works
            self.resnet=None

        # Second-stage classifier
        self.classify = False
        if self.classify:
            self.modelc = load_classifier(name='resnet101', n=2)  # initialize
            self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(self.device).eval()

    # This function shows the result of detect_bounded_box for images in the source folder,
    #  and has no return values! Only for debug
    def dataset_detect_bounded_box(self,source=SOURCE, view_img=False):
        with torch.no_grad():
            # Set Dataloader
            '''vid_path, vid_writer = None, None
            if webcam:
                view_img = True
                cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadStreams(source, img_size=imgsz)
            else:
                save_img = True'''

            dataset = LoadImages(source, img_size=self.imgsz)

            # Get names and colors
            names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

            # Run inference
            t0 = time.time()
            img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)  # init img
            _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once
            for path, img, im0s, vid_cap in dataset:
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                t1 = time_synchronized()
                pred = self.model(img, augment=AUGMENT)[0]

                # Apply NMS
                pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=CLASSES, agnostic=AGNOSTIC_NMS)
                t2 = time_synchronized()

                # Apply Classifier
                if self.classify:
                    pred = apply_classifier(pred, self.modelc, img, im0s)

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                    p = Path(p)  # to Path
                    s += '%gx%g ' % img.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            if  view_img:  # Add bbox to image
                                label = f'{names[int(cls)]} {conf:.2f}'
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                    # Print time (inference + NMS)
                    print(f'{s}Done. ({t2 - t1:.3f}s)')

                    # Stream results
                    if view_img:
                        cv2.imshow(str(p), im0)
                        cv2.waitKey(0)

            print(f'Done. ({time.time() - t0:.3f}s)')

    # detect bounded boxes of cones  in one image, return three lists for different colors
    # lists are like [((x1,y1),(x2,y2)),...], coordinates of  top-left and bottom-right corners of bounded boxes
    def detect_bounded_box(self,img0, view_img=False):
        with torch.no_grad():
            redcone_array=[]
            bluecone_array=[]
            yellowcone_array=[]
            # Get names and colors
            if view_img:
                names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
                colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

            # Run inference
            t0 = time.time()
            # Padded resize
            img = letterbox(img0, new_shape=self.imgsz)[0]

            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = self.model(img, augment=AUGMENT)[0]

            # Apply NMS
            pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=CLASSES, agnostic=AGNOSTIC_NMS)
            # Apply Classifier
            if self.classify:
                pred = apply_classifier(pred, self.modelc, img, img0)

            # Process detections
            for i, det in enumerate(pred):  # detections per imag
                gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if  view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)
                        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                        if int(cls)==0:
                            redcone_array.append((c1,c2))
                        elif int(cls)==1:
                            bluecone_array.append((c1,c2))
                        else:
                            yellowcone_array.append((c1,c2))
                # Print time (inference + NMS)
                t2 = time_synchronized()
                print(f'Done. ({t2 - t1:.3f}s)')
                # Stream results
                if view_img:
                    cv2.imshow('Result', img0)
                    cv2.waitKey(0)
            return redcone_array, bluecone_array, yellowcone_array

    # detect keypoints of cones  in one image, return three lists for different colors
    # lists are like [[(x1,y1),(x2,y2),...,(x7,y7)],...], coordinates of keypoints of cones
    # in the order  ['top','mid_L_top','mid_R_top','mid_L_bot','mid_R_bot','bot_L','bot_R']
    # First use yolov5 to detect bounded_boxes, then use resnet to detect keypoints in each bounded box in parallel
    def detect_keypoints(self, img0, view_img=False):
        if self.resnet==None:
            print('No resnet model!')
            return

        # detect keypoints for a single image
        def keypoints_per_image(img,x1,y1,label): # x1,y1 are coordinates of the bounded box's top-left corner
            shape = img.shape                                     # label is color of the cone red:0, blue:1, yellow:2
            img = cv2.resize(img, dsize=(80, 80), interpolation=cv2.INTER_CUBIC) # rescale to (80,80) for resnet
            v = img.astype('float') / 255
            v = np.array([v])
            prediction = self.resnet.predict(v)[0]
            points = prediction.reshape(7, 2)
            res=[]
            for point in points:
                point[0] = point[0] / 80 * shape[1]+x1 # rescale to original size, resnet process images with (80,80,3)
                point[1] = point[1] / 80 * shape[0]+y1
                res.append((int(point[0]),int(point[1])))
            if label==0:
                redkps.append(res)
            elif label==1:
                bluekps.append(res)
            else:
                yellowkps.append(res)

        # yolov5 to detect bounded boxes
        with torch.no_grad():
            cone_array = []
            redkps = []
            bluekps = []
            yellowkps = []
            # Run inference
            t0 = time.time()
            # Padded resize
            img = letterbox(img0, new_shape=self.imgsz)[0]

            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = self.model(img, augment=AUGMENT)[0]

            # Apply NMS
            pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=CLASSES, agnostic=AGNOSTIC_NMS)

            # Apply Classifier
            if self.classify:
                pred = apply_classifier(pred, self.modelc, img, img0)

            # Process detections
            for i, det in enumerate(pred):  # detections per imag
                gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                        cone_array.append((c1, c2,int(cls)))
            t2=time_synchronized()
            print(f'Bounded_box. ({t2 - t1:.3f}s)')  # yolov5 time

            # parallel execution to detect keypoints using resnet
            executor = ThreadPoolExecutor()
            futures=[]
            for ((x1,y1),(x2,y2),label) in cone_array:
                future=executor.submit(keypoints_per_image,img0[y1:y2,x1:x2],x1,y1,label)
                futures.append(future)
            wait(futures,return_when='ALL_COMPLETED')
            t3 = time_synchronized()
            print(f'Keypoint. ({t3 - t2:.3f}s)') # Keypoints detection time
            print(f'Total. ({t3 - t1:.3f}s)')
            # Stream results
            if view_img:
               for red in redkps:
                   for coordinate in red:
                       cv2.circle(img0, (coordinate[0], coordinate[1]), 1, (0, 0, 255), 4)
               for blue in bluekps:
                   for coordinate in blue:
                       cv2.circle(img0, (coordinate[0], coordinate[1]), 1, (255, 0, 0), 4)
               for yellow in yellowkps:
                   for coordinate in yellow:
                       cv2.circle(img0, (coordinate[0], coordinate[1]), 1, (0, 255, 255), 4)
               cv2.imshow('Result', img0)
               cv2.waitKey(0)
        return redkps,bluekps,yellowkps


if __name__ == '__main__':
    check_requirements()
    d = Detector(resnet_weights='weights/very_simple_resnet.hdf5')
    for root,dirnames,filenames in os.walk(SOURCE):
        for name in filenames:
            img=cv2.imread(SOURCE+'/'+name)
            d.detect_bounded_box(img,view_img=True)