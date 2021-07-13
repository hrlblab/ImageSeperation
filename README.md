## ImageSeperation

Repository contains Python implementation of ...

*
*
*

## Installation

'''$ pip install -r requirements.txt'''

## Usage examples

#### Train

'''$ python train_cross_val.py --epochs 100 --batch-size 32  --data imageCLEF_cross_val.yaml --weights yolov5s.pt --single-cls --sideloss'''

#### Test

'''python test.py --batch 32 --data test.yaml --weight best.pt  --single-cls --save-txt --save-conf'''

#### Model ensemble

test_merge.py

#### Subfigure crop



#### Train your own data
train.py
test.py



## Citation
If you find this repository useful in your research, please cite:
