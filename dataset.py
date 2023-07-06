import os, shutil
import torch
import cv2
import numpy as np
# from IPython.display import Image, display
from PIL import Image
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, rgb2gray, rgb2hsv


def main():
    os.makedirs('images_brief/train/class/', exist_ok=True)  # 40,000
    os.makedirs('images_brief/val/class', exist_ok=True)  # 1,000
    for i, file in enumerate(os.listdir('testSet_resize')):
        if i < 1000:
            os.rename('testSet_resize/' + file, 'images_brief/val/class/' + file)
        else:
            os.rename('testSet_resize/' + file, 'images_brief/train/class/' + file)


def move():
    for i in range(10):
        os.makedirs(f'places365_standard/test_256/part{i}', exist_ok=True)

    path_img='places365_standard/test_256'
    ls = os.listdir(path_img)
    print(len(ls))
    a = 0
    b = 1
    for i in ls:
        if i.find('Places365')!=-1:
            if a == 32850:
                b += 1
                a = 0
            shutil.move(path_img+'/'+i, f"places365_standard/test_256/part{b}/"+i)
            a += 1


def move2():
    path_img='places365_standard/test_256'
    ls = os.listdir(path_img)
    print(len(ls))

    for i in ls:
        if i.find('Places365')!=-1:

            shutil.move(path_img+'/'+i, "places365_standard/test_256/part0/"+i)

# def display():
#     display(Image(filename='images_brief/val/class/00a0b13cf8e233f72f047b7eb398b1a5.jpg'))


def test():
    file_pth = "places365_standard/val/airfield/Places365_val_00000435.jpg"
    image = Image.open(file_pth).convert('RGB')
    image = np.asarray(image)
    img_gray = rgb2gray(image)
    plt.clf()
    plt.imsave(fname='z1.jpg', arr=img_gray)
    img_gray = np.expand_dims(img_gray, 2)
    img = (np.repeat(img_gray, 3, 2)*255).astype(np.uint8) # 0-1 CxHxW
    print(img[:,:,0])
    print(img.shape)
    plt.clf()
    plt.imsave(fname='z.jpg', arr=img)
    # image2 = Image.open('z1.jpg').convert('RGB')
    # image2 = np.asarray(image2)/255.0
    # print(image2[:,:,0])
    # print(image2.shape)

    img_lab = rgb2lab(img)
    print(img_lab[:,:,1])


def test2():
    file_pth = "z.jpg"
    image = Image.open(file_pth).convert('RGB')
    image = np.asarray(image)
    img_gray = rgb2gray(image)
    plt.clf()
    plt.imsave(fname='zgray.jpg', arr=img_gray, cmap='gray')
    img_lab = rgb2lab(image)
    print(img_lab.astype(np.uint8))
    print(img_lab.shape)
    # img_ab = img_lab[:, :, 1:3]
    # img_ab = (img_ab + 128) / 255
    # img_ab = img_ab.transpose((2, 0, 1))
    # print()
    # print(img_ab)

def data_process():
    root = 'places365_standard/train'
    save_root = 'places365_standard/Unqualified'
    Unqualified_num = 0
    dir1_list = os.listdir(root)
    for dir1 in dir1_list: # 室外、海洋
        os.makedirs(os.path.join(save_root, dir1), exist_ok=True)
        dir2_list = os.listdir(root+'/'+dir1)
        for dir2 in dir2_list: # jpg
            file_name = os.path.join(root, dir1, dir2)
            save_pth = os.path.join(save_root, dir1)
            image = Image.open(file_name).convert('RGB')
            # 1.判断是否为灰度图像，删除
            flag = isGrayMap(image)
            if flag:
                Unqualified_num += 1
                shutil.move(file_name, save_pth)
                continue
            # 2.判断是否过暗/过曝
            flag = overExposeDetect(file_name)
            if flag:
                Unqualified_num += 1
                shutil.move(file_name, save_pth)
                continue
            # 3.判断是否模糊
            flag = isblurred(image)
            if flag:
                Unqualified_num += 1
                shutil.move(file_name, save_pth)
                continue
        print(Unqualified_num)


def pre_process():
    img_pth = "places365_standard/train/server_room/00004853.jpg"
    image = Image.open(img_pth).convert('RGB')
    # img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # img_gray1 = (rgb2gray(image)*255).astype(np.uint8)
    # 1.判断是否为灰度图像，删除
    isGrayMap(image)
    # 2.判断是否过暗/过曝
    overExposeDetect(img_pth)
    # 3.判断是否模糊
    isblurred(image)


# 1.判断是否是灰度图像
def isGrayMap(img, threshold = 15):
    """
    入参：
    img：PIL读入的图像
    threshold：判断阈值，图片3个通道间差的方差均值小于阈值则判断为灰度图。
    阈值设置的越小，容忍出现彩色面积越小；设置的越大，那么就可以容忍出现一定面积的彩色，例如微博截图。
    如果阈值设置的过小，某些灰度图片会被漏检，这是因为某些黑白照片存在偏色，例如发黄的黑白老照片、
    噪声干扰导致灰度图不同通道间值出现偏差（理论上真正的灰度图是RGB三个通道的值完全相等或者只有一个通道，
    然而实际上各通道间像素值略微有偏差看起来仍是灰度图）
    出参：
    bool值
    """
    if len(img.getbands()) == 1:
        return True
    img1 = np.asarray(img.getchannel(channel=0), dtype=np.int16)
    img2 = np.asarray(img.getchannel(channel=1), dtype=np.int16)
    img3 = np.asarray(img.getchannel(channel=2), dtype=np.int16)
    diff1 = (img1 - img2).var()
    diff2 = (img2 - img3).var()
    diff3 = (img3 - img1).var()
    diff_sum = (diff1 + diff2 + diff3) / 3.0
    if diff_sum <= threshold:
        # print("三通道差的方差平均值：", diff_sum, '<=', threshold, "为灰度图像")
        return True
    else:
        # print("三通道差的方差平均值：", diff_sum, '>', threshold, "为彩色图像")
        return False


# 2.判断曝光度
def overExposeDetect(img_path,size=[256, 256], thre_dark = 0.175, thre_bright = 0.27):
    """
    @illustrate: 曝光检测
    @param
    thre:过暗/过曝图像块占的比例
    """
    img = cv2.imread(img_path, 1)
    img = cv2.resize(img, size)
    status = "normal"
    flag = False
    if img.shape[2] != 1:
        hsvSpaceImage = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) # hsv转换
    else:
        hsvSpaceImage = img.clone()
    hsvImageVChannels = hsvSpaceImage[:, :, 2]
    step = 8   #以8*8小窗口遍历V通道图像
    imageOverExposeBlockNum = 0
    imageBlocks = 0
    imageDarkBlockNum = 0
    #遍历
    i = 0
    while i < hsvImageVChannels.shape[0]:
        j = 0
        while j < hsvImageVChannels.shape[1]:
            imageBlock = hsvImageVChannels[i:i+step, j:j+step]
            mea = np.mean(imageBlock)# 求小矩形的均值
            if mea > 233.0:
                imageOverExposeBlockNum += 1
            elif mea < 38.0:
                imageDarkBlockNum += 1
            imageBlocks += 1
            j += step
        i += step
    dark_ratio = imageDarkBlockNum/imageBlocks
    bright_ratio = imageOverExposeBlockNum/imageBlocks
    # print("暗区域正常区间:(0," + str(thre_dark*100) + "]", dark_ratio*100, "%")
    # print("亮区域正常区间:(0," + str(thre_bright*100) + "]", bright_ratio*100, "%")
    if dark_ratio > thre_dark:
        status = "dark"
        flag = True
        # print(str(img_path) + " 过暗patch比例:" + str(dark_ratio * 100) +"% "+ "status:" + status)
    if bright_ratio > thre_bright:
        status = "overexposure"
        flag = True
        # print(str(img_path) + " 过曝patch比例:" + str(bright_ratio * 100) +"% "+ "status:" + status)
    return flag


# 3.判断是否模糊
def isblurred(img, threshold = 0.0045):
    '''
    laplacian算子作卷积，判断方差
    '''
    img = np.asarray(img)
    img_gray = rgb2gray(img)
    # plt.imsave(fname='zzz.jpg', arr=img_gray, cmap='gray')
    grayVar = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    if grayVar <= threshold:
        # print("灰度图像laplacian变换的方差：", grayVar, '<=', threshold, "图像模糊")
        return True
    else:
        # print("灰度图像laplacian变换的方差：", grayVar, '>', threshold, "图像清晰")
        return False


# 4.判断颜色是否多样(颜色方差小)
def iscolorful(img, threshold = 0.0016):
    img = np.asarray(img)
    img_hsv = rgb2hsv(img)
    colorVar = np.var(img_hsv[:,:,0])
    print(colorVar)
    # print(img_hsv[:,:,0], '\n', img_hsv[:,:,0].max())
    # print(img_hsv[:,:,1], '\n', img_hsv[:,:,1].max())
    # print(img_hsv[:,:,2], '\n', img_hsv[:,:,2].max())
    # img_lab = rgb2lab(img)
    # colorVar1 = np.var(img_lab[:,:,1])
    # colorVar2 = np.var(img_lab[:,:,2])
    # colorVar = (colorVar1+colorVar2)/2.0
    # print("颜色方差", colorVar1, colorVar2, colorVar)


def getImageVar(imgPath):
    image = cv2.imread(imgPath)
    img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageVar = cv2.Laplacian(img2gray, cv2.CV_64F).var()
    return imageVar


if __name__ == '__main__':
    # display(Image(filename='images_brief/val/class/00a0b13cf8e233f72f047b7eb398b1a5.jpg'))
    # move2()
    # test2()
    # pre_process()
    data_process()
    # if_color("places365_standard/val/airplane_cabin/Places365_val_00001553.jpg")
