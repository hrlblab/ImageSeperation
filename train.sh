#!/bin/bash

python train_cross_val.py --epochs 60 --batch-size 32  --data imageCLEF_cross_val.yaml --weights yolov5s.pt --single-cls --sideloss

#python train_cross_val.py --epochs 30 --batch-size 32  --data imageCLEF_cross_val.yaml --weights yolov5s.pt --single-cls