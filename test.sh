#!/bin/bash
#python test.py --batch 32 --data test.yaml --weight ./runs/train/exp4/fold5/weights/best5_1.pt  --single-cls
#python test.py --batch 32 --data test.yaml --weight ./runs/train/exp4/fold5/weights/best5_2.pt  --single-cls
#python test.py --batch 32 --data test.yaml --weight ./runs/train/exp4/fold5/weights/best5_3.pt  --single-cls
#python test.py --batch 32 --data test.yaml --weight ./runs/train/exp4/fold5/weights/best5_4.pt  --single-cls
#python test.py --batch 32 --data test.yaml --weight ./runs/train/exp4/fold5/weights/best5_5.pt  --single-cls
#python test.py --batch 32 --data test.yaml --weight ./runs/train/exp4/fold5/weights/best5_6.pt  --single-cls
#python test.py --batch 32 --data test.yaml --weight ./runs/train/exp4/fold5/weights/best5_7.pt  --single-cls
#python test.py --batch 32 --data test.yaml --weight ./runs/train/exp4/fold5/weights/best5_8.pt  --single-cls
#python test.py --batch 32 --data test.yaml --weight ./runs/train/exp4/fold5/weights/best5_9.pt  --single-cls
#python test.py --batch 32 --data test.yaml --weight ./runs/train/exp4/fold5/weights/best5_10.pt  --single-cls
#python test.py --batch 32 --data test.yaml --weight ./runs/train/exp4/fold5/weights/best5_11.pt  --single-cls
#python test.py --batch 32 --data test.yaml --weight ./runs/train/exp4/fold5/weights/best5_12.pt  --single-cls
#python test.py --batch 32 --data test.yaml --weight ./runs/train/exp4/fold5/weights/best5_13.pt  --single-cls
#python test.py --batch 32 --data test.yaml --weight ./runs/train/exp4/fold5/weights/best5_14.pt  --single-cls
#python test.py --batch 32 --data test.yaml --weight ./runs/train/exp4/fold5/weights/best5_15.pt  --single-cls
#python test.py --batch 32 --data test.yaml --weight ./runs/train/exp4/fold5/weights/best5_16.pt  --single-cls
#python test.py --batch 32 --data test.yaml --weight ./runs/train/exp4/fold5/weights/best5_17.pt  --single-cls
#python test.py --batch 32 --data test.yaml --weight ./runs/train/exp4/fold5/weights/best5_18.pt  --single-cls
#python test.py --batch 32 --data test.yaml --weight ./runs/train/exp4/fold5/weights/best5_19.pt  --single-cls
#python test.py --batch 32 --data test.yaml --weight ./runs/train/exp4/fold5/weights/best5_20.pt  --single-cls
#python test.py --batch 32 --data test.yaml --weight ./runs/train/exp4/fold5/weights/best5_21.pt  --single-cls
#python test.py --batch 32 --data test.yaml --weight ./runs/train/exp4/fold5/weights/best5_22.pt  --single-cls
#python test.py --batch 32 --data test.yaml --weight ./runs/train/exp4/fold5/weights/best5_23.pt  --single-cls
#python test.py --batch 32 --data test.yaml --weight ./runs/train/exp4/fold5/weights/best5_24.pt  --single-cls
#python test.py --batch 32 --data test.yaml --weight ./runs/train/exp4/fold5/weights/best5_25.pt  --single-cls
python test.py --batch 1 --data test.yaml --weight best1.pt  --single-cls --save-txt --save-conf
python test.py --batch 1 --data test.yaml --weight best2.pt  --single-cls --save-txt --save-conf
python test.py --batch 1 --data test.yaml --weight best3.pt  --single-cls --save-txt --save-conf
python test.py --batch 1 --data test.yaml --weight best4.pt  --single-cls --save-txt --save-conf
python test.py --batch 1 --data test.yaml --weight best5.pt  --single-cls --save-txt --save-conf






#python train_cross_val.py --epochs 30 --batch-size 32  --data imageCLEF_cross_val.yaml --weights yolov5s.pt --single-cls