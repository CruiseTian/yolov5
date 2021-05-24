import csv
import os
import json
import cv2

DIR='../dataset/mit_test/'
FILE='yolov3-training_train_mini_yolo.csv'

LABEL_DIR=DIR+'labels'
if not os.path.exists(LABEL_DIR):
    os.mkdir(LABEL_DIR)
for (root,dirs,files) in os.walk(LABEL_DIR):
    for file in files:
        os.remove(os.path.join(root,file))

with open(DIR+FILE) as csv_file:
    csv_reader = csv.reader(csv_file)
    for i, row in enumerate(csv_reader):
        print(row[0])
        if i < 2:
            continue
        img_boxes = []
        for img_box_str in row[5:]:
            if not img_box_str == "":
                img_boxes.append(json.loads(img_box_str))
        print(img_boxes)
        img=cv2.imread(DIR+'images/'+row[0])
        for img_box in img_boxes:
            x=img_box[0]
            y=img_box[1]
            w=img_box[3]
            h=img_box[2]
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow('image',img)
        cv2.waitKey(0)