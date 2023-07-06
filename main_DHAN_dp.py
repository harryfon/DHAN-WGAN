import torch
import time
import argparse
import torch.nn as nn
import numpy as np
from torch.utils import data
from torchvision import transforms
from model.model_Colorization import ColorizationHAN
from utils import MyImageFolder, MyVOCSegDataset, Meter, MyVisdom, gray_ab2rgb, my_makedirs, my_save_plot, my_save_best
from d2l import torch as d2l
from tqdm import tqdm


# Parse arguments and prepare program
# work
parser = argparse.ArgumentParser(description='Training and Using ColorizationHAN')
parser.add_argument('--work-name', type=str, default='DHAN-class', help='Which net structure to use, CHAN, CHAN-GAN, CHAN-WGAN-GP')
# gpu
parser.add_argument('--try-gpu', type=int, default=2, help='Which gpu to use')
parser.add_argument('--data-parallel', type=list, default=[1, 2], help='Which gpu to parallel')
# train
parser.add_argument('--num-epochs', default=1, type=int, metavar='N', help='Number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N', help='Size of mini-batch (default: 12)')
parser.add_argument('--lr', '--learning-rate', default=2e-5, type=float, metavar='LR', help='Learning rate at start of training')
parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float, metavar='W', help='Weight decay (default: 1e-6)')
# model
parser.add_argument('--data-sets', type=str, default='./places365_standard', metavar='DIR', help='Path to dataset')
parser.add_argument('--save-images', type=bool, default=True, help='Val save images')
parser.add_argument('--use-global', type=bool, default=True, help='Use global feature')
parser.add_argument('--use-seg', type=bool, default=False, help='Use segmentation feature')
parser.add_argument('--use-pretrain', type=bool, default=True, help='Use pretrain model')
parser.add_argument('--pre-net-path', type=str, default='checkpoints/DHAN-class/model-epoch-7-iter-72000-losses-0.03269-2.44003.pth', metavar='DIR', help='Path to pre net')
parser.add_argument('--change-model', type=bool, default=False)
parser.add_argument('--require-grad', type=bool, default=True)


def main(args, use_gpu):
    torch.backends.cudnn.benchmark = True
    print('Arguments: {}'.format(args))
    # define
    model = ColorizationHAN(use_seg_feature=False)
    # model.apply(init_weights)
    # criterion = nn.MSELoss()
    # criterion = nn.HuberLoss(delta=0.05)
    criterion_col = nn.L1Loss()
    criterion_class = nn.CrossEntropyLoss() if args.use_global else None
    criterion_seg = nn.CrossEntropyLoss() if args.use_seg else None

    if not args.use_seg:
        # training data
        train_transforms = transforms.Compose([transforms.RandomCrop([224, 224]), transforms.RandomHorizontalFlip()])
        train_data = MyImageFolder(f'{args.data_sets}/train', train_transforms)
        train_loader = data.DataLoader(train_data, args.batch_size, shuffle=True, drop_last=True)

        # validation data
        val_transforms = transforms.Compose([transforms.RandomCrop([224, 224])])
        val_data = MyImageFolder(f'{args.data_sets}/val', val_transforms)
        val_loader = data.DataLoader(val_data, args.batch_size, shuffle=False, drop_last=True)
    else:
        train_data = MyVOCSegDataset(True, (224, 224), './VOCdevkit/VOC2012')
        train_loader = data.DataLoader(train_data, args.batch_size, shuffle=True, drop_last=True)
        val_data = MyVOCSegDataset(False, (224, 224), './VOCdevkit/VOC2012')
        val_loader = data.DataLoader(val_data, args.batch_size, shuffle=False, drop_last=True)

    print(f"Total number of train samples: {len(train_data)}")
    print(f"Total number of val samples: {len(val_data)}")
    print(f"Total number of (train) batches: {len(train_loader)}")
    print(f"Total number of (val) batches: {len(val_loader)}")

    # to GPU
    device = d2l.try_gpu(args.try_gpu)
    if use_gpu:
        criterion_col = criterion_col.to(device)
        criterion_class = criterion_class.to(device) if args.use_global else None
        criterion_seg = criterion_seg.to(device) if args.use_seg else None
        model = model.to(device)
        # model = nn.DataParallel(model, args.data_parallel)
    if args.use_pretrain:
        # model.change_seg_model(21, device if use_gpu else torch.device('cpu'))
        pretrained_net = torch.load(args.pre_net_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(pretrained_net)
    # if args.change_model:
    #     model.change_seg_model(21, device if use_gpu else torch.device('cpu'))
    if args.require_grad:
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.wd)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    else:
        model.set_freeze(args.require_grad)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=args.lr, betas=(0.9, 0.99), weight_decay=args.wd)

    my_makedirs(args)

    # loss_plt_epoch = []
    # loss_plt_iter = []

    # vis try
    vis = MyVisdom(args)

    # train model
    for epoch in range(args.num_epochs-1, args.num_epochs):
        train_loss, loss_iter = train(epoch, model, train_loader, criterion_col, criterion_class, criterion_seg,
                                    optimizer, use_gpu, device, vis, args)
        with torch.no_grad():
            val_loss = validate(epoch, model, val_loader, criterion_col, use_gpu, device, args)
        # loss_plt_epoch.append((train_loss, val_loss))
        # loss_plt_iter.extend(loss_iter)
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     torch.save(model.state_dict(),
        #             'checkpoints/{}/best/model-epoch-{}-val-losses-{:.5f}.pth'.format(args.work_name, epoch + 1, val_loss))

        # torch.save(model.state_dict(),
        #         'checkpoints/{}/model-epoch-{}-val-losses-{:.5f}.pth'.format(args.work_name, epoch + 1, val_loss))
        # 绘图
        # 损失-迭代
        # my_loss_plot(name="plot/loss_plot_iter", train_loss=loss_plt_iter)
        # 损失-回合
        # loss_arr_epoch = np.array(loss_plt_epoch)
        # dict_loss = {"train_loss": loss_arr_epoch.transpose()[0], "val_loss": loss_arr_epoch.transpose()[1]}
        # my_loss_plot(x_label="epoch", name="plot/loss_plot_epoch", **dict_loss)

    return val_loss


# modi target
def train(epoch, model, train_loader, criterion_col, criterion_class, criterion_seg, 
        optimizer, use_gpu, device, vis, args):
    print('--------- Start training epoch {} ---------'.format(epoch+1))
    model.train()
    loss_col_iter, loss_class_iter,  loss_seg_iter= [], [], []
    best_col, best_class, best_seg= 1e2, 1e2, 1e2
    times, losses = Meter(), Meter()
    for i, (img_gray, img_ab, target) in enumerate(train_loader):
        start_time = time.time()
        if use_gpu:
            img_gray, img_ab, target = img_gray.to(device), img_ab.to(device), target.to(device)
        # ab_out, class_out, seg_out
        out_ab, else_info = model(img_gray)
        batch_loss_col = criterion_col(out_ab, img_ab)
        batch_loss_class, batch_loss_seg = None, None
        batch_loss = batch_loss_col
        loss_col_iter.append(batch_loss_col.item())
        # exist class_info
        if args.use_global:
            batch_loss_class = criterion_class(else_info[0], target)
            batch_loss = batch_loss + batch_loss_class/300.0
            loss_class_iter.append(batch_loss_class.item())
        # exist seg_info
        if args.use_seg:
            batch_loss_seg = criterion_seg(else_info[1], target)
            batch_loss = batch_loss + batch_loss_seg/50.0
            loss_seg_iter.append(batch_loss_seg.item())
        # update
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        # bath time
        batch_time = time.time() - start_time
        times.update(batch_time, 1)
        # avg loss
        losses.update(batch_loss_col.item(), img_gray.size(0))
        # update visdom
        if vis.vis is not None:
            vis.update(args, batch_loss_col, batch_loss_class, batch_loss_seg, i+1+epoch*len(train_loader))
        # print
        # if (i+1) % 10 == 0:
        #     with torch.no_grad():
        #         vis.image(img_gray, out_ab)
        if (i+1) % 100 == 0:
            with torch.no_grad():
                vis.image(img_gray, out_ab)
            print('Epoch:[{0}] Batch:[{1}/{2}]\t'
                  'Batch_time:{batch_time:.3f}\t'
                  'Loss:{loss:.5f}\t'.format(epoch+1, i+1, len(train_loader), batch_time=times.val, loss=losses.val))
        # save, plot and visual
        if ((i+1) % 3000 == 0) or (i == len(train_loader) - 1):
            # Visualization
            with torch.no_grad():
                vis.image(img_gray, out_ab)
            my_save_plot(args, model, epoch, i, batch_loss_col, loss_col_iter,
                        batch_loss_class, loss_class_iter, batch_loss_seg, loss_seg_iter)
        # best
        if i > 5000:
            best_col, best_class, best_seg = my_save_best(args, model, epoch, i, batch_loss_col, best_col,
                                                        batch_loss_class, best_class, batch_loss_seg, best_seg)
    # finish
    print('--------- Finishing training epoch {}\t'
          'Time:{batch_time_sum:.3f}\t'
          'Loss:{loss_avg:.5f} ---------'.format(epoch+1, batch_time_sum=times.sum, loss_avg=losses.avg))
    return losses.avg, loss_col_iter


def validate(epoch, model, val_loader, criterion_col, use_gpu, device, args):
    print('--------- Start validation epoch {} ---------'.format(epoch+1))
    model.eval()
    times, losses = Meter(), Meter()
    time.sleep(0.5)
    with tqdm(total=len(val_loader)) as pbar:
        for i, (img_gray, img_ab, target) in enumerate(val_loader):
            start_time = time.time()
            if use_gpu:
                img_gray, img_ab, target = img_gray.to(device), img_ab.to(device), target.to(device)
            out_ab, _ = model(img_gray)
            batch_loss = criterion_col(out_ab, img_ab)
            batch_time = time.time() - start_time
            losses.update(batch_loss.item(), img_gray.size(0))
            times.update(batch_time, 1)
            pbar.set_description(f'batches[{i + 1}]')
            pbar.set_postfix(batch_loss=batch_loss.item(), batch_time=batch_time)
            pbar.update()
            if args.save_images and (i+1) % 50 == 0:  # 每几个batch存一次
                for j in range(min(len(img_ab), 1)):  # 每个batch存前几张
                    save_path = {'colorized': f'checkpoints/{args.work_name}/color/',
                                 'gray': f'checkpoints/{args.work_name}/gray/'}
                    save_name = 'img-{}-epoch-{}.jpg'.format(i*val_loader.batch_size+j, epoch+1)
                    gray_ab2rgb(img_gray[j].detach().cpu(), out_ab[j].detach().cpu(),
                                save_path=save_path, save_name=save_name)
    time.sleep(0.5)
    print('--------- Finishing validation epoch {}\t'
          'Time:{batch_time_sum:.3f}\t'
          'Loss:{loss_avg:.5f} ---------'.format(epoch+1, batch_time_sum=times.sum, loss_avg=losses.avg))
    return losses.avg


def main_val(args):
    epoch = 0
    net_path = 'checkpoints/HAN/model-epoch-1-iter-150288-losses-0.00127.pth'
    device = d2l.try_gpu(args.try_gpu)
    model = ColorizationHAN().to(device)
    pretrained = torch.load(net_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(pretrained)
    model = nn.DataParallel(model, args.data_parallel)
    val_transforms = transforms.Compose([transforms.Resize([256, 256])])
    val_data = MyImageFolder(f'{args.data_sets}/val', val_transforms)
    val_loader = data.DataLoader(val_data, args.batch_size, shuffle=False, drop_last=True)
    # criterion = nn.HuberLoss(delta=0.05).to(device)
    criterion = nn.L1Loss().to(device)
    validate(epoch, model, val_loader, criterion, device)


if __name__ == '__main__':
    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()
    # CHAN
    main(args, use_gpu)
