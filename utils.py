import torch
import os
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.datasets as dset
import matplotlib.pyplot as plt
from d2l import torch as d2l
from skimage.color import rgb2lab, rgb2gray, lab2rgb
import skimage.io as io
from visdom import Visdom


# 数据加载器，重写，加载、转换、返回 生成标签
class MyImageFolder(dset.ImageFolder):
    def __getitem__(self, index):
        path, target = self.imgs[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        img_gray, img_ab = rgb2gray_ab(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img_gray, img_ab, target


class MyVOCSegDataset(torch.utils.data.Dataset):
    def __init__(self, is_train, crop_size, voc_dir):
        self.is_train = is_train
        self.crop_size = crop_size
        self.voc_dir = voc_dir
        self.path_features, self.path_labels = self.read_voc_images()
        self.VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                        [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                        [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                        [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                        [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                        [0, 64, 128]]
        self.VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                       'diningtable', 'dog', 'horse', 'motorbike', 'person',
                       'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
        self.colormap2label = self.voc_colormap2label()
        print('read ' + str(len(self.path_features)) + ' examples')

    def read_voc_images(self):
        """读取所有符合的VOC图像路径"""
        txt_fname = os.path.join(self.voc_dir, 'ImageSets', 'Segmentation',
                                 'train.txt' if self.is_train else 'val.txt')
        with open(txt_fname, 'r') as f:
            file_names = f.read().split()
        file_names = self.filter(file_names)
        path_features, path_labels = [], []
        for _, f_name in enumerate(file_names):
            path_features.append(os.path.join(
                self.voc_dir, 'JPEGImages', f'{f_name}.jpg'))
            path_labels.append(os.path.join(
                self.voc_dir, 'SegmentationClass', f'{f_name}.png'))
        return path_features, path_labels

    def voc_colormap2label(self):
        """构建从RGB到VOC类别索引的映射"""
        colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
        for i, colormap in enumerate(self.VOC_COLORMAP):
            colormap2label[
                (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
        return colormap2label

    @staticmethod
    def voc_label_indices(colormap, colormap2label):
        """将VOC标签中的RGB值映射到它们的类别索引"""
        colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
        idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
               + colormap[:, :, 2])
        return colormap2label[idx]

    @staticmethod
    def voc_rand_crop(feature, label, height, width):
        """随机裁剪特征和标签图像"""
        rect = torchvision.transforms.RandomCrop.get_params(
            feature, (height, width))
        feature = torchvision.transforms.functional.crop(feature, *rect)
        label = torchvision.transforms.functional.crop(label, *rect)
        return feature, label

    def normalize_image(self, img):
        return self.transform(img.float() / 255)

    def filter(self, file_names):
        file_name_end = [f_name for f_name in file_names if(
            torchvision.io.read_image(os.path.join(
                self.voc_dir, 'JPEGImages', f'{f_name}.jpg')).shape[1] >= self.crop_size[0] and
            torchvision.io.read_image(os.path.join(
                self.voc_dir, 'JPEGImages', f'{f_name}.jpg')).shape[2] >= self.crop_size[1]
        )]
        return file_name_end

    def __getitem__(self, idx):
        # 读取
        mode = torchvision.io.image.ImageReadMode.RGB
        feature = torchvision.io.read_image(self.path_features[idx])
        label = torchvision.io.read_image(self.path_labels[idx], mode)
        label = self.voc_label_indices(label, self.colormap2label)
        # 裁剪
        feature, label = self.voc_rand_crop(feature, label, *self.crop_size)
        # 色域转换 2lab, 归一化 [0, 1]
        feature = feature.permute(1, 2, 0)
        img_gray, img_ab = rgb2gray_ab(feature)
        # 返回 img_gray, img_ab, label
        return  img_gray, img_ab, label

    def __len__(self):
        return len(self.path_features)


class Meter(object):
    def __init__(self):
        self.val, self.count, self.sum, self.avg = 0, 0, 0, 0

    def update(self, val, num=1):
        self.val = val
        self.count += num
        self.sum += val * num
        self.avg = self.sum/self.count


# 可视化
class MyVisdom():
    def __init__(self, args):
        try:
            self.vis = Visdom(env=args.work_name, raise_exceptions=True)
            self.vis.line([None], [None], win='train_loss_col',
                        opts=dict(title='train_loss_col', xlabel='iters', ylabel='loss_col'))
            if args.use_global:
                self.vis.line([None], [None], win='train_loss_class',
                            opts=dict(title='train_loss_class', xlabel='iters', ylabel='loss_class'))
            if args.use_seg:
                self.vis.line([None], [None], win='train_loss_seg',
                            opts=dict(title='train_loss_seg', xlabel='iters', ylabel='loss_seg'))
        except OSError as e:
            print("visdom 连接错误")
            self.vis = None
        except Exception as e:
            print("visdom 其它错误")
            self.vis = None
        
    def update(self, args, batch_loss_col, batch_loss_class, batch_loss_seg, iter):
        try:
            self.vis.line([batch_loss_col.cpu().detach().numpy()], [iter], win='train_loss_col', update='append',
                        opts=dict(title='train_loss_col', xlabel='iters', ylabel='loss_col'))
            if args.use_global:
                self.vis.line([batch_loss_class.cpu().detach().numpy()], [iter], win='train_loss_class',
                            update='append',
                            opts=dict(title='train_loss_class', xlabel='iters', ylabel='loss_class'))
            if args.use_seg:
                self.vis.line([batch_loss_seg.cpu().detach().numpy()], [iter], win='train_loss_seg',
                            update='append',
                            opts=dict(title='train_loss_seg', xlabel='iters', ylabel='loss_seg'))
        except OSError as e:
            print("visdom 连接错误")
        except Exception as e:
            print("visdom 其它错误", e)

    def image(self, img_gray, out_ab):
        img_list = []
        for j in range(4):
            img_list.append(gray_ab2rgb(img_gray[j].detach().cpu(), out_ab[j].detach().cpu()).transpose(2, 0, 1))
        img_list = np.asarray(img_list)
        try:
            self.vis.images(img_list, win='Visualization', nrow=2)
        except Exception as e:
            print("visdom 其它错误")


def rgb2gray_ab(img_org):
    img_org = np.asarray(img_org)
    img_lab = rgb2lab(img_org)
    img_ab = img_lab[:, :, 1:3]
    img_ab = (img_ab + 128) / 255
    img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()
    img_gray = rgb2gray(img_org)
    img_gray = torch.from_numpy(img_gray).unsqueeze(0).float()
    return img_gray, img_ab


def gray_ab2rgb(img_gray, img_ab, save_path=None, save_name=None):
    plt.clf()
    img_rgb = torch.cat((img_gray, img_ab), dim=0).numpy()
    img_rgb = img_rgb.transpose((1, 2, 0))
    img_rgb[:, :, 0] = img_rgb[:, :, 0] * 100
    img_rgb[:, :, 1:3] = img_rgb[:, :, 1:3] * 255 - 128
    img_rgb = lab2rgb(img_rgb.astype(np.float64))
    img_gray = img_gray.squeeze().numpy()
    if save_name is not None and save_path is not None:
        plt.imsave(fname='{}{}'.format(save_path['colorized'], save_name), arr=img_rgb)
        plt.imsave(fname='{}{}'.format(save_path['gray'], save_name), arr=img_gray, cmap='gray')
    return img_rgb


def my_makedirs(args):
    os.makedirs(f'checkpoints/{args.work_name}/best', exist_ok=True)
    os.makedirs(f'checkpoints/{args.work_name}/color', exist_ok=True)
    os.makedirs(f'checkpoints/{args.work_name}/gray', exist_ok=True)
    os.makedirs(f'plot/{args.work_name}', exist_ok=True)


def my_loss_plot(x_label="iterations", y_label="loss", name="loss", **kwargs):
    plt.figure(figsize=(10, 5))
    plt.title("Loss During Training")
    for key, val in kwargs.items():
        plt.plot(val, label=key)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='best')
    plt.savefig('{}.jpg'.format(name))
    # plt.show()
    plt.close()


def my_save_plot(args, model, epoch, i, batch_loss_col, loss_col_iter,
                batch_loss_class, loss_class_iter, batch_loss_seg, loss_seg_iter):
    if not args.use_global and not args.use_seg:
        torch.save(model.state_dict(),'checkpoints/{}/model-epoch-{}-iter-{}-losses-{:.5f}.pth'.format(
                    args.work_name, epoch + 1, i + 1, batch_loss_col.item()))
        my_loss_plot(name="plot/{}/epoch-{}-loss_plot_col".format(args.work_name, epoch+1),
                    train_loss=loss_col_iter)
    elif args.use_global and not args.use_seg:
        torch.save(model.state_dict(), 'checkpoints/{}/model-epoch-{}-iter-{}-losses-{:.5f}-{:.5f}.pth'.format(
                    args.work_name, epoch + 1, i + 1, batch_loss_col.item(), batch_loss_class.item()))
        my_loss_plot(name="plot/{}/epoch-{}-loss_plot_col".format(args.work_name, epoch+1),
                    train_loss=loss_col_iter)
        my_loss_plot(name="plot/{}/epoch-{}-loss_plot_class".format(args.work_name, epoch+1),
                    train_loss=loss_class_iter)
    elif args.use_seg and not args.use_global:
        torch.save(model.state_dict(), 'checkpoints/{}/model-epoch-{}-iter-{}-losses-{:.5f}-{:.5f}.pth'.format(
                    args.work_name, epoch + 1, i + 1, batch_loss_col.item(), batch_loss_seg.item()))
        my_loss_plot(name="plot/{}/epoch-{}-loss_plot_col".format(args.work_name, epoch+1),
                    train_loss=loss_col_iter)
        my_loss_plot(name="plot/{}/epoch-{}-loss_plot_seg".format(args.work_name, epoch+1),
                    train_loss=loss_seg_iter)
    else:
        torch.save(model.state_dict(), 'checkpoints/{}/model-epoch-{}-iter-{}-losses-{:.5f}-{:.5f}-{:.5f}.pth'.format(
                    args.work_name, epoch + 1, i + 1, batch_loss_col.item(), batch_loss_class.item(), batch_loss_seg.item()))
        my_loss_plot(name="plot/{}/epoch-{}-loss_plot_col".format(args.work_name, epoch+1),
                    train_loss=loss_col_iter)
        my_loss_plot(name="plot/{}/epoch-{}-loss_plot_class".format(args.work_name, epoch+1),
                    train_loss=loss_class_iter)
        my_loss_plot(name="plot/{}/epoch-{}-loss_plot_seg".format(args.work_name, epoch+1),
                    train_loss=loss_seg_iter)


def my_save_best(args, model, epoch, i, batch_loss_col, best_col,
                batch_loss_class, best_class, batch_loss_seg, best_seg):
    if not args.use_global and not args.use_seg:
        if batch_loss_col.item() < best_col:
            best_col = batch_loss_col.item()
            torch.save(model.state_dict(), 'checkpoints/{}/best/model-epoch-{}-iter-{}-losses-{:.5f}.pth'.format(
                        args.work_name, epoch+1, i+1, best_col))
    elif args.use_global and not args.use_seg:
        if (batch_loss_col.item() < best_col) and (batch_loss_class.item() < best_class):
            best_col = batch_loss_col.item()
            best_class = batch_loss_class.item()
            torch.save(model.state_dict(), 'checkpoints/{}/best/model-epoch-{}-iter-{}-losses-{:.5f}-{:.5f}.pth'.format(
                        args.work_name, epoch+1, i+1, batch_loss_col.item(), batch_loss_class.item()))
    elif args.use_seg and not args.use_global:
        if (batch_loss_col.item() < best_col) and (batch_loss_seg.item() < best_seg):
            best_col = batch_loss_col.item()
            best_seg = batch_loss_seg.item()
            torch.save(model.state_dict(), 'checkpoints/{}/best/model-epoch-{}-iter-{}-losses-{:.5f}-{:.5f}.pth'.format(
                        args.work_name, epoch+1, i+1, batch_loss_col.item(), batch_loss_seg.item()))
    else:
        if (batch_loss_col.item() < best_col) and (batch_loss_class.item() < best_class) and (batch_loss_seg.item() < best_seg):
            best_col = batch_loss_col.item()
            best_class = batch_loss_class.item()
            best_seg = batch_loss_seg.item()
            torch.save(model.state_dict(), 'checkpoints/{}/best/model-epoch-{}-iter-{}-losses-{:.5f}-{:.5f}-{:.5f}.pth'.format(
                        args.work_name, epoch+1, i+1, batch_loss_col.item(), batch_loss_class.item(), batch_loss_seg.item()))
    return best_col, best_class, best_seg


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def main():
    img = io.imread('checkpoints/CHAN/color/img-490-epoch-1.jpg')
    img_gray, img_ab = rgb2gray_ab(img)
    a = torch.cat((img_gray, img_ab), dim=0)
    print(a.shape)
    img2 = gray_ab2rgb(img_gray, img_ab)
    io.imshow(img2)
    print(img2.shape)
    # io.show()


if __name__ == '__main__':
    # main()
    img = torch.from_numpy(io.imread('checkpoints/CHAN/color/img-490-epoch-1.jpg'))
    img_lab = rgb2lab(img)
    img_gray = rgb2gray(img)
    print(img_lab.dtype)
    print(img_gray.shape)
    print(type(img.type()))
    print(type(img_lab))
    