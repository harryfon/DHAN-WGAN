# encoding:gbk
import torch, os, sys, lpips, cv2, time
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray, rgb2lab, lab2rgb
from torchvision import transforms
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from Quality_assess.PCQI import PCQI
from Quality_assess.UCIQE import UCIQE

# 回归
from model.Compare_Unet import Unet
# from model.model_Colorization import ColorizationHAN

# 分类
# from model.Compare_Lizuka import ColNet as LizukaNet
# from model.model_Colorization import ColorizationHAN as ColorizationHanClass
# from model.model_Colorization import ColorizationHAN as ColorizationHanSeg

# 生成
# from model.Compare_Vitoria import Generator as ChromaGAN_GAN 
# from model.Compare_Vitoria import Generator as ChromaGAN_WGAN
from model.model_GAN import Generator as ColorizationHan_GAN
# from model.model_GAN import Generator as ColorizationHan_WGAN


# model
# *************改************
# model_name = 'UNet'
# model_name = 'CHAN'
# model_name = 'LizukaNet'
# model_name = 'CHAN-class'
# model_name = 'CHAN-seg'
# model_name = 'ChromaGAN'
# model_name = 'ChromaWGAN'
# model_name = 'CHAN_GAN'
# model_name = 'CHAN_WGAN'
model_name = 'DHAN-class'
# *************改************
# model_path = 'checkpoints/DHAN/model-epoch-1-iter-180346-losses-0.02935-3.04221.pth'
# model_path = 'checkpoints/WGAN-GP/netG-epoch-1-iters-138000-lossG-4.82910-0.02392.pth'
model_path = 'checkpoints/WGAN-GP/netG-epoch-1-iters-86000-lossG-3.23338-0.03343.pth'

# 参考图像
# *************改************
dir_name = 'ocean'
# *************改************
graf_name = 'Places365_test_00237302' # jpg
open_name = f'generator/source/{dir_name}/{graf_name}.jpg'

# 生成图像
save_name = f'generator/out/{dir_name}/{graf_name}-{model_name}.jpg'
# graf_path = f'generator/source/{graf_name}.jpg'

# save_path = f'generator/out/{graf_name}-contrast.jpg'

# *************改************
gpu = 0

def main():
    f = open('实验结果.out', 'a')
    sys.stdout = f
    device  = torch.device(f'cuda:{gpu}')
    """生成灰度图像以及改变大小后的灰度图像"""
    #img_trans('generator/source/bridge.jfif')
    #image = mpimg.imread('zhuoer.png')
    image = Image.open(open_name).convert('RGB')
    image = transforms.Resize([224, 224])(image)
    img_resized = np.asarray(image)
    image = np.asarray(image)
    img_gray = rgb2gray(image)
    img_lab = rgb2lab(image)
    img_ab = img_lab[:, :, 1:3]
    img_ab = (img_ab + 128)/255
    img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float().unsqueeze(0).to(device)
    img_gray = torch.from_numpy(img_gray).unsqueeze(0).unsqueeze(0).float().to(device)

    # *************改************
    # 1.UNet
    # model = Unet(1, 2).to(device)
    model = ColorizationHan_GAN(use_seg_feature=False).to(device)

    # model = ColorizationHAN().to(device)
    # model.change_seg_model(21, device)
    # # model = nn.DataParallel(model)

    pretrained = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(pretrained)
    model.eval()
    # criterion = nn.L1Loss().to(device)
    with torch.no_grad():
        
        # *************改************
        # out_ab= model(img_gray)
        out_ab, _ = model(img_gray)

        # print(out_ab.shape)
        # print(img_ab.shape)
        # loss = criterion(out_ab, img_ab)
        plt.clf()
        color_image = torch.cat((img_gray, out_ab), 1).squeeze(0).cpu().numpy()
        color_image = color_image.transpose(1, 2, 0)
        color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
        color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128
        color_image = lab2rgb(color_image.astype(np.float64))
        color_image = (color_image*255).astype(np.uint8)
        plt.imsave(fname=save_name, arr=color_image)

        psnr = compare_psnr(img_resized, color_image)
        mssim = compare_ssim(img_resized, color_image, channel_axis=-1)
        print(model_name, '\t', graf_name+'.jpg')
        print('PSNR:', psnr,'\t','MSSIM:', mssim)

        # print(np.mean(img_resized))
        # print(np.mean(color_image))

        # plt.figure("测试") # , dpi=300, figsize=(4, 3)
        # plt.subplot(1,2,1)
        # plt.imshow(img_resized)
        # plt.axis('off')
        # plt.subplot(1,2,2)
        # plt.imshow(color_image)
        # plt.axis('off')
        # # plt.tight_layout()
        # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0.01)
        # # f = plt.gcf()
        # # f.savefig('{}.jpg'.format(f'{name}-contrast'))
        # # f.clear()
        # plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        # plt.show()
        plt.clf()
        plt.close()

        # print(loss)


def main2():
    os.makedirs('generator/out/多人造物体',exist_ok=True)
    os.makedirs('generator/out/海洋',exist_ok=True)
    os.makedirs('generator/out/陆地',exist_ok=True)
    os.makedirs('generator/out/事物多种类',exist_ok=True)
    os.makedirs('generator/out/室内',exist_ok=True)
    os.makedirs('generator/out/室外',exist_ok=True)


def main3():
    
    f = open('实验结果.out', 'a')
    sys.stdout = f
    # sys.stdout = sys.__stdout__
    # print('std out')


    def Test1(rootDir): 
        list_dirs = os.walk(rootDir) 
        for root, dirs, files in list_dirs: 
            if files is not []:
                for f in files: 
                    print(os.path.join(root, f))  
            

    path = 'generator/source'
    # dir1 = os.listdir(path)
    # for dir in dir1:
    #     dir2 = os.listdir(path+'/'+dir)
    #     print(dir2)

    Test1(path)


def img_trans(src_path):
    """将任何位置的图片移动至工程"""
    os.makedirs('generator/source',exist_ok=True)
    os.makedirs('generator/out',exist_ok=True)
    os.rename(src_path, 'generator/source/Places365_test_00000028.png')
    #shutil.move(src_path, 'E:/pycharm-project/pytorch_colorization/generator/source/bridge.jfif')


def main4():
    os.makedirs('generator/source_224/多人造物体',exist_ok=True)
    os.makedirs('generator/source_224/海洋',exist_ok=True)
    os.makedirs('generator/source_224/陆地',exist_ok=True)
    os.makedirs('generator/source_224/事物多种类',exist_ok=True)
    os.makedirs('generator/source_224/室内',exist_ok=True)
    os.makedirs('generator/source_224/室外',exist_ok=True)
    root = 'generator/source'
    dir1_list = os.listdir(root)
    for dir1 in dir1_list: # 室外、海洋
        dir2_list = os.listdir(root+'/'+dir1) # jpg
        for dir2 in dir2_list:
            file_name = os.path.join(root, dir1, dir2)
            """生成灰度图像以及改变大小后的灰度图像"""
            image = Image.open(file_name).convert('RGB')
            image = transforms.Resize([224, 224])(image)
            image = np.asarray(image)
            save_root = 'generator/source_224'
            save_name = os.path.join(save_root, dir1, dir2)
            plt.imsave(fname=save_name, arr=image)


def main_auto():
    f = open('result.out', 'a')
    sys.stdout = f
    print()
    print(f'**************{model_name}****************')
    device  = torch.device(f'cuda:{gpu}')
    # *************改************
    # 1.UNet
    # model = Unet(1, 2).to(device)
    # model = LizukaNet(365).to(device)
    model = ColorizationHan_GAN().to(device)
    
    # model = ColorizationHanSeg().to(device)
    model.change_seg_model(21, device)
    # # model = nn.DataParallel(model)
    # criterion = nn.L1Loss().to(device)
    pretrained = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(pretrained)
    model.eval()

    root = 'generator/source'
    save_root = 'generator/out'
    dir1_list = os.listdir(root)
    for dir1 in dir1_list: # 室外、海洋
        dir2_list = os.listdir(root+'/'+dir1) # jpg
        for dir2 in dir2_list:
            file_name = os.path.join(root, dir1, dir2)
            """生成灰度图像以及改变大小后的灰度图像"""
            image = Image.open(file_name).convert('RGB')
            image = transforms.Resize([224, 224])(image)
            img_resized = np.asarray(image)
            image = np.asarray(image)
            img_gray = rgb2gray(image)
            img_lab = rgb2lab(image)
            img_ab = img_lab[:, :, 1:3]
            img_ab = (img_ab + 128)/255
            img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float().unsqueeze(0).to(device)
            img_gray = torch.from_numpy(img_gray).unsqueeze(0).unsqueeze(0).float().to(device)
            
            with torch.no_grad():
                
                # *************改************
                # out_ab= model(img_gray)
                out_ab, _ = model(img_gray)
                # loss = criterion(out_ab, img_ab)
                plt.clf()
                color_image = torch.cat((img_gray, out_ab), 1).squeeze(0).cpu().numpy()
                color_image = color_image.transpose(1, 2, 0)
                color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
                color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128
                color_image = lab2rgb(color_image.astype(np.float64))
                color_image = (color_image*255).astype(np.uint8)
                save_name = os.path.join(save_root, dir1, f'{model_name}'+'-'+dir2)
                plt.imsave(fname=save_name, arr=color_image)

                psnr = compare_psnr(img_resized, color_image)
                mssim = compare_ssim(img_resized, color_image, channel_axis=-1)
                print(dir1+'/'+dir2)
                print('PSNR:', psnr,'\t','MSSIM:', mssim)

                plt.clf()
                plt.close()
    print(f'**************{model_name}****************')


def main_auto2():
    f = open('result.out', 'a')
    sys.stdout = f
    print()
    print(f'************** LPIPS/PCQI/UCIQE ****************')
    root = 'generator/Ground-truth'
    dir_list = os.listdir(root)
    for dir in dir_list: # jpg
        for dir1 in os.listdir('generator/out'):
            for dir2 in os.listdir('generator/out'+'/'+dir1):
                if dir in dir2:
                    ref_img_pth = os.path.join(root, dir)
                    gen_img_pth = os.path.join('generator/out', dir1, dir2)
                    # LPIPS
                    loss_fn = lpips.LPIPS(net='vgg', verbose=False)
                    img0 = lpips.im2tensor(lpips.load_image(ref_img_pth))
                    img1 = lpips.im2tensor(lpips.load_image(gen_img_pth))
                    lpips_distance = loss_fn.forward(img0, img1)
                    # PCQI
                    img0 = cv2.imread(ref_img_pth)
                    img1 = cv2.imread(gen_img_pth)
                    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
                    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                    PCQI_score, p_map = PCQI(img0, img1)
                    # UCIQE
                    uciqe, (delta, conl, mu) = UCIQE(gen_img_pth)
                    print(dir2, '\n', 
                            'LPIPS:', lpips_distance.item(), '\t',
                            'PCQI:', PCQI_score, '\t',
                            'UCIQE:', uciqe, '\t')
    print(f'************** LPIPS/PCQI/UCIQE ****************')


def history_gen():
    device  = torch.device(f'cuda:{gpu}')

    sour_img = '北京1957.jpg'
    img_gray1 = cv2.imread('历史照片/'+sour_img, cv2.IMREAD_GRAYSCALE)
    img_gray1 = cv2.resize(img_gray1, [224, 224])

    img_gray = torch.from_numpy(img_gray1[None]/255.0).unsqueeze(0).float().to(device)
    # img_gray = transforms.Resize([224, 224])(img_gray)
    # model = ColorizationHan_GAN().to(device)
    # model.change_seg_model(21, device)
    model = ColorizationHan_GAN(use_seg_feature=False).to(device)
    pretrained = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(pretrained)
    model.eval()

    with torch.no_grad():
                
        out_ab, _ = model(img_gray)
        color_image = torch.cat((img_gray, out_ab), 1).squeeze(0)
        color_image = color_image.permute(1, 2, 0)
        color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
        color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128
        color_image = lab2rgb(color_image.cpu().numpy().astype(np.float64))
        color_image = (color_image*255).astype(np.uint8)

        plt.clf()
        plt.imsave(fname='历史照片/result/'+'原'+sour_img, arr=img_gray1, cmap='gray')
        plt.imsave(fname='历史照片/result/'+'11'+sour_img, arr=color_image)
        plt.clf()
        plt.close()

    # print(img.shape, '', type(img), '', img.max())
    # print(img)



if __name__ == '__main__':
    time.sleep(2)
    history_gen()
    # main_auto2()
    # main()
    # img_trans('places365_standard/test_256/part0/Places365_test_00000028.jpg')
