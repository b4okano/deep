import cv2
import numpy as np
import itertools
import random as rd
import os
import glob
import itertools

counter = 0
img_size = 40

dir_path = 'images'
listfile = []
for i in range(10):
    listfile.append(glob.glob(os.path.join(os.path.join(dir_path, str(i)),'*.png')))

x_scale = [0.9, 0.95, 1.0, 1.05] # x scale
y_scale = [0.9, 0.95, 1.0, 1.05] # y scale
angle_list = [-3, -1, 0, 1, 3]  # rotation
#angle_list = [-3, 0, 3]  # rotation
blur_list = [1, 3, 5]  # blur
noise_list = [0, 10, 20] # noise
orgfile_list = list(itertools.chain.from_iterable(listfile))

train_dir  = 'number_generated/train/'
verify_dir = 'number_generated/verify/'

def add_noise(img, mu=0, sigma=50):
    noise = np.random.normal(mu, sigma, img.shape)
    noisy_img = img.astype(np.float64) + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

    return noisy_img

def output_image(tr, vr, total, img, filename, rate=0.8):
    # 0.8 の割合で訓練データ 0.2の割合で検証データ作成
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if tr < total * rate and vr < total * (1.0 - rate):
        if rd.random() < rate:
            cv2.imwrite(train_dir+filename, img)
            tr+1
        else:
            cv2.imwrite(verify_dir+filename, img)
            vr+1
    elif tr >= total * rate:
        cv2.imwrite(verify_dir+filename, img)
    else:
        cv2.imwrite(train_dir+filename, img)
    return tr, vr


def figure_generate():
    counter_str = str(counter)
    counter_zerofill = counter_str.zfill(4)
    trans  = cv2.getRotationMatrix2D((20, 20), angle ,1)
    movetr = np.float32([[1, 0, 10],[0, 1, 10]])
    tr_counter = 0
    vr_counter = 0
    total = len(orgfile_list) * len(x_scale) * len(y_scale)\
             * len(angle_list) * len(blur_list) * len(noise_list)
    for i in range(10):
        for filepath in listfile[i]:
            number=str(i)
            filename = 'data'+counter_zerofill+'_ans'+number+'.png'
            img  = cv2.imread(filepath)
            img1 = cv2.resize(img, (int(img_size*0.7*xscale), int(img_size*0.7*yscale)))
            img2 = cv2.warpAffine(img1, movetr, (img_size, img_size))
            img3 = cv2.warpAffine(img2, trans, (img_size, img_size))
            img4 = img3[6:34, 6:34]
            img5 = cv2.blur(img4, (blur, blur))
            img6 = add_noise(img5, 0, noise)
            tr_counter, vr_counter = \
               output_image(tr_counter, vr_counter, total, img6, filename)
            print(filename)

os.makedirs(train_dir, exist_ok=True)
os.makedirs(verify_dir, exist_ok=True)
# apply combinations of x, y, rotation, blur, and noise
for orgfille, xscale, yscale, angle, blur, noise in itertools.product(orgfile_list,x_scale,y_scale,angle_list,blur_list,noise_list):
    counter+=1
    figure_generate()