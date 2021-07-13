import os
import cv2
import glob
import random
#
train_txt_path = 'train'
val_txt_path = 'val'
#全部的txt
path_imgs = '../imageCLEF/labels/train/*.txt'
#glob.glob返回所有匹配的文件路径列表。
image_list = glob.glob(path_imgs)
#打乱
random.shuffle(image_list)
#这里是划分，我设置的是0.85：0.15  可以根据自己情况划分
num = len(image_list)

# i = 0
# whole_list = image_list
# val_list = image_list[i*int(0.2*num):(i+1)*int(0.2*num)]
# train_list = [i for i in whole_list if i not in val_list]
#
# print(len(train_list))
# print(len(val_list))
#
# with open(train_txt_path,'w') as f:
#     for line in train_list:
#         jpg_name = line.replace('txt','jpg')
#         f.write(jpg_name + '\n')
# #写入验证集
# with open(val_txt_path,'w') as f:
#     for line in val_list:
#         jpg_name = line.replace('txt','jpg')
#         f.write(jpg_name + '\n')


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
    # 写入验证集
    with open(val_txt_path+str(i+1)+'.txt', 'w') as f:
        for line in val_list:
            jpg_name = line.replace('txt', 'jpg')
            jpg_name = jpg_name.replace('labels', 'images')
            f.write(jpg_name + '\n')



# train_list = image_list[:int(0.85*num)]
# val_list = image_list[int(0.85*num):]
# with open(train_txt_path,'w') as f:
#     for line in train_list:
#         jpg_name = line.replace('txt','jpg')
#         img = cv2.imread(jpg_name)
#         if img is not None:
#             f.write(jpg_name + '\n')
# #写入验证集
# with open(val_txt_path,'w') as f:
#     for line in val_list:
#         jpg_name = line.replace('txt','jpg')
#         img = cv2.imread(jpg_name)
#         if img is not None:
#             f.write(jpg_name + '\n')
