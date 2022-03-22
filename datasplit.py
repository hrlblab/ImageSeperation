import os
import cv2
import glob
import random
#
train_txt_path = 'train'
val_txt_path = 'val'

path_imgs = '../imageCLEF/labels/train/*.txt'

image_list = glob.glob(path_imgs)

random.shuffle(image_list)

num = len(image_list)



for i in range(0,5):
    whole_list = image_list
    val_list = image_list[i * int(0.2 * num):(i + 1) * int(0.2 * num)]
    train_list = [i for i in whole_list if i not in val_list]

    print(len(train_list))
    print(len(val_list))

    with open(train_txt_path+str(i+1)+'.txt', 'w') as f:
        for line in train_list:
            jpg_name = line.replace('txt', 'jpg')
            jpg_name = jpg_name.replace('labels', 'images')
            f.write(jpg_name + '\n')


    with open(val_txt_path+str(i+1)+'.txt', 'w') as f:
        for line in val_list:
            jpg_name = line.replace('txt', 'jpg')
            jpg_name = jpg_name.replace('labels', 'images')
            f.write(jpg_name + '\n')


