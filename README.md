## ImageSeperation

Repository contains Python implementation of ...

* Training and testing code for sub-figure detector with sideloss
* cross validation on sub-figure detector, (results are essembled using weighted-box-fusion)

## Installation

``` pip install -r requirements.txt```

## Data access

You need to contact the organizers of the task (https://www.imageclef.org/2016/medical) and ask for licensing the dataset.



## Usage examples

#### Sub-figure detection

```$ python detect.py --weights ./detector.pt --source ./image_dir --hide-labels(optional) --hide-conf(optional)```

#### Subfigure crop

#### Train
```$ python train.py --epochs 100 --batch-size 32  --data imageCLEF.yaml --weights yolov5s.pt --single-cls --sideloss```

```$ python train_cross_val.py --epochs 100 --batch-size 32  --data imageCLEF_cross_val.yaml --weights yolov5s.pt --single-cls --sideloss```

#### Test

```python test.py --batch 32 --data test.yaml --weight best.pt  --single-cls --save-txt --save-conf```

#### Model ensemble

test_merge.py


## Citation
If you find this repository useful in your research, please cite:

Yao, T., Qu, C., Liu, Q., Deng, R., Tian, Y., Xu, J., Jha, A., Bao, S., Zhao, M., Fogo, A.B. and Landman, B.A., 2021. Compound Figure Separation of Biomedical Images with Side Loss. In Deep Generative Models, and Data Augmentation, Labelling, and Imperfections (pp. 173-183). Springer, Cham.
