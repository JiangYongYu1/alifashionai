import os
import random

root_path = '/home/classify/alidata'

Image_root_path = os.path.join(root_path, 'Images')
for cloth in os.listdir(Image_root_path):
    cloth_path = os.path.join(Image_root_path, cloth)
    total_cloth_images_list = os.listdir(cloth_path)
    nums_images = len(total_cloth_images_list)
    list_nums = range(nums_images)
    train_percent = 0.85
    trian_nums_images = int(nums_images * train_percent)
    trainval = random.sample(list_nums, trian_nums_images)

    trainval_root_path = os.path.join(root_path, 'trainval')
    if not os.path.exists(trainval_root_path):
        os.mkdir(trainval_root_path)
    cloth_trainval_path = os.path.join(trainval_root_path, cloth)
    if not os.path.exists(cloth_trainval_path):
        os.mkdir(cloth_trainval_path)
    ftrain = open(cloth_trainval_path + 'train.txt', 'w')
    fval = open(cloth_trainval_path + 'val.txt', 'w')

    for i in list_nums:
        name = total_cloth_images_list[i] + '\n'
        if i in trainval:
            ftrain.write(name)
        else:
            fval.write(name)

    ftrain.close()

    fval.close()



