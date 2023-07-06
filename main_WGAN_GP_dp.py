from tkinter.messagebox import NO
import torch
import time
import argparse
import os
import torch.nn as nn
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from torch.utils import data
from torchvision import transforms
import torchvision.utils as vutils
from utils import MyImageFolder, Meter, gray_ab2rgb, my_loss_plot, my_makedirs
from model.model_WGAN_GP import Generator, Discriminator
# from d2l import torch as d2l
from visdom import Visdom
import matplotlib.animation as animation
from IPython.display import HTML
# import matplotlib
# matplotlib.use('tkagg')

'''图片显示、损失绘制、马尔可夫判别器、反向传播、计算图 GPU不被打断, 划定lab颜色空间范围'''
# Parse arguments and prepare program
# Use GPU
parser = argparse.ArgumentParser(description='Training and Using ColorizationGAN')
parser.add_argument('--save-images', type=bool, default=True, help='Val save images')
parser.add_argument('--use-global', type=bool, default=True, help='Use global feature')
parser.add_argument('--num-epochs', default=1, type=int, metavar='N', help='Number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=2e-5, type=float, metavar='LR', help='Learning rate at start of training')
parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float, metavar='W', help='Weight decay (default: 1e-5)')
parser.add_argument('-b', '--batch-size', default=8, type=int, metavar='N', help='Size of mini-batch (default: 10)')
parser.add_argument('--data-sets', type=str, default='./places365_standard', metavar='DIR', help='Path to dataset')
parser.add_argument('--work-name', type=str, default='WGAN-GP', help='Which network to use')
parser.add_argument('--try-gpu', type=int, default=2, help='Which gpu to use')
parser.add_argument('--data-parallel', type=list, default=[1, 2], help='Which gpu to parallel')


if_pretrain = False
netG_path = 'save/4/netG-epoch-1-iters-225432-lossG-0.00093-0.03938-0.00608.pth'
netD_path = 'save/4/netD-epoch-1-iters-225432-lossD-0.00090.pth'


def main():
    args = parser.parse_args()
    print('Arguments: {}'.format(args))

    # dir
    my_makedirs(args)

    # dataset
    # training data
    train_transforms = transforms.Compose([transforms.Resize([224, 224]), transforms.RandomHorizontalFlip()])
    train_data = MyImageFolder(f'{args.data_sets}/train', train_transforms)
    train_loader = data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # validation data
    val_transforms = transforms.Compose([transforms.Resize([224, 224])])
    val_data = MyImageFolder(f'{args.data_sets}/val', val_transforms)
    val_loader = data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, drop_last=True)

    print(f"Total number of train samples: {len(train_data)}")
    print(f"Total number of val samples: {len(val_data)}")
    print(f"Total number of (train) batches: {len(train_loader)}")
    print(f"Total number of (val) batches: {len(val_loader)}")

    # device = d2l.try_gpu(args.try_gpu)
    device = torch.device(f'cuda:{args.try_gpu}')

    # model define
    torch.backends.cudnn.benchmark = True
    netG = Generator().to(device)
    # change model & load prenet
    netG.change_seg_model(21, device)
    pretrained_net = torch.load('checkpoints/CHAN/model-epoch-80-iter-145-losses-0.02578-1.21032.pth', map_location=lambda storage, loc: storage)
    netG.load_state_dict(pretrained_net)
    # freeze
    netG.set_freeze(False)
    # netG = nn.DataParallel(netG, args.data_parallel)
    netD = Discriminator(3).to(device)
    # netD = nn.DataParallel(netD, args.data_parallel)
    # if if_pretrain:
    #     pretrained_netG = torch.load(netG_path, map_location=lambda storage, loc: storage)
    #     netG.load_state_dict(pretrained_netG)
    #     pretrained_netD = torch.load(netD_path, map_location=lambda storage, loc: storage)
    #     netD.load_state_dict(pretrained_netD)

    criterion_col = nn.L1Loss().to(device)
    criterion_Class = None
    # criterion_col = nn.HuberLoss(delta=0.05).to(device) if args.use_global else None
    # criterion_col = nn.L1Loss().to(device) if args.use_global else None
    # criterion_Class = nn.CrossEntropyLoss().to(device) if args.use_global else None

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=args.wd)
    # optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=args.wd)
    optimizerG = torch.optim.Adam(filter(lambda p: p.requires_grad, netG.parameters()),
                                lr=args.lr, betas=(0.5, 0.999), weight_decay=args.wd)
    # optimizerD = optim.RMSprop(netD.parameters(), lr=args.lr, weight_decay=args.wd)
    # optimizerG = optim.RMSprop(netG.parameters(), lr=args.lr, weight_decay=args.wd)

    # vis try
    try:
        vis = Visdom(env=args.work_name, raise_exceptions=True)
        # GAN 损失
        vis.line([[None, None]], [0.], win='train_loss',
                 opts=dict(title='D&G_train_loss', legend = ['G_loss','D_loss'], xlabel='iters', ylabel='loss'))
        # GAN 概率
        vis.line([[None, None]], [0.], win='discriminate',
                 opts=dict(title='D_discri', legend = ['D_x','D_z'], xlabel='iters', ylabel='prob'))
        # 颜色损失
        vis.line([None], [0.], win='train_loss_col', opts=dict(title='train_loss_col', xlabel='iters', ylabel='loss_col'))
        # if args.use_global:
        #     # 颜色损失
        #     vis.line([None], [0.], win='train_loss_col', opts=dict(title='train_loss_col', xlabel='iters', ylabel='loss_col'))
        #     # 类别损失
        #     vis.line([None], [0.], win='train_loss_class', opts=dict(title='train_loss_class', xlabel='iters', ylabel='loss_class'))
    except OSError as e:
        print("visdom 连接错误")
        vis = None
    except Exception as e:
        print("visdom 其它错误")
        vis = None

    # train
    # vis = None
    train(train_loader, val_loader, netG, netD, device, optimizerD, optimizerG, args.num_epochs, args.save_images,
        args.work_name, vis, args.use_global, criterion_col, criterion_Class)


def train(train_loader, val_loader, netG, netD, device, optimizerD, optimizerG, num_epochs, save_images,
    work_name, vis, use_global_feature, criterion_col=None, criterion_class=None):
    # Training and val Loop

    # Lists to keep track of progress
    G_losses = [] # every batch
    D_losses = []
    col_loss = []
    # class_loss = []
    val_loss = [] # every epoch
    best_col = 1e2
    # best_class = 1e2
    weight_GAN = 0.005
    # weight_class = 0.003
    weight_gp = 10
    weight_pix = 1

    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the train data_loader
        # data: img_gray, img_ab, target

        # ****************** train ******************* #
        print("--------- Starting training Loop... ---------")
        print('--------- Start training epoch {} ---------'.format(epoch+1))
        netG.train()
        netD.train()
        iters_D = 5
        iter_nums = int(len(train_loader)/iters_D)
        for i, data in enumerate(train_loader, 0):

            ############################
            # 1 Update D network: maximize D(x) - D(G(z))
            ###########################
            # 1.1 Train with all-real batch
            # Format batch 5次
            optimizerD.zero_grad()

            img_gray = data[0].to(device)
            real_ab = data[1].to(device)
            real = torch.cat((img_gray, real_ab), dim=1)
            b_size = real_ab.size(0)
            # Forward pass real batch through D
            D_real = netD(real).view(-1)
            # backward
            (-D_real.mean()*weight_GAN).backward()
            # 鉴别器对真判别为真的平均分数
            D_x = D_real.mean().item()

            # Generate fake image batch with G
            out_ab, _ = netG(img_gray)
            # if use_global_feature:
            #     out_ab, class_out= netG(img_gray)
            #     target = data[2].to(device)
            # else:
            #     out_ab, class_out, target= netG(img_gray), None, None

            # 1.2 Train with all-fake batch
            # Generate batch of latent vectors
            fake = torch.cat((img_gray, out_ab), dim=1)
            D_fake = netD(fake.detach()).view(-1)
            # backward
            (D_fake.mean()*weight_GAN).backward()
            # 鉴别器对假判别为真的平均分数
            D_G_z1 = D_fake.mean().item()
            
            # 1.3 计算损失值，最大化EM距离
            D_loss = -(D_real.mean() - D_fake.mean())
            gradient_penalty = cal_gradient_penalty(netD, real, fake.detach(), device)
            (weight_gp*gradient_penalty*weight_GAN).backward()
            # errD_all = (D_loss + weight_gp * gradient_penalty)*weight_GAN

            # 1.4 反向传播，优化
            # optimizerD.zero_grad()
            # errD_all.backward()
            optimizerD.step()

            ############################
            # 2 Update G network: maximize D(G(z))
            ###########################
            # 2.1 generate fake D
            if (i+1)%5 == 0:
                optimizerG.zero_grad()
                D_fake = netD(fake).view(-1)
                # 鉴别器对假判别为真的平均分数
                D_G_z2 = D_fake.mean().item()
                # 2.2 Calculate G's loss based on this output
                G_loss = -D_fake.mean()
                # 2.3 Calculate gradients for G and Update G
                batch_loss_col = criterion_col(out_ab, real_ab)
                errG_all = G_loss*weight_GAN + batch_loss_col*weight_pix
                errG_all.backward()
                # if use_global_feature:
                #     batch_loss_col = criterion_col(out_ab, real_ab)
                #     batch_loss_class = criterion_class(class_out, target)
                #     errG_all = G_loss*weight_GAN + batch_loss_col + batch_loss_class*weight_class
                #     errG_all.backward()
                # else:
                #     G_loss.backward()
                
                optimizerG.step()

                if vis is not None:
                    try:
                        vis.line([[(G_loss*weight_GAN).cpu().detach().numpy(), (D_loss*weight_GAN).cpu().detach().numpy()]],
                                 [i+1+epoch*len(train_loader)], win='train_loss', update='append',
                                 opts=dict(title='D&G_train_loss', legend = ['G_loss','D_loss'], xlabel='iters', ylabel='loss'))
                        vis.line([[D_x, D_G_z2]], [i+1+epoch*len(train_loader)], win='discriminate', update='append',
                                 opts=dict(title='D_discri', legend = ['D_x','D_z'], xlabel='iters', ylabel='prob'))
                        vis.line([batch_loss_col.cpu().detach().numpy()], [i+1+epoch*len(train_loader)], win='train_loss_col', update='append',
                                opts=dict(title='train_loss_col', xlabel='iters', ylabel='loss_col'))
                        # if use_global_feature:
                        #     vis.line([batch_loss_col.cpu().detach().numpy()], [i+1+epoch*len(train_loader)], win='train_loss_col', update='append',
                        #             opts=dict(title='train_loss_col', xlabel='iters', ylabel='loss_col'))
                        #     vis.line([(batch_loss_class*weight_class).cpu().detach().numpy()], [i+1+epoch*len(train_loader)], win='train_loss_class', update='append',
                        #             opts=dict(title='train_loss_class', xlabel='iters', ylabel='loss_class'))
                    except OSError as e:
                        print("visdom 连接错误")
                    except Exception as e:
                        print("visdom 其它错误", e)

                # Output training stats x真值 z假值 D概率
                if (i+1) % 20 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.5f %.5f\tLoss_G: %.5f %.5f\tD(x): %.5f\tD(G(z)): %.5f / %.5f'
                        % (epoch+1, num_epochs, i+1, iter_nums,
                            D_loss.item(), gradient_penalty.item(),
                            G_loss.item(), batch_loss_col.item(), D_x, D_G_z1, D_G_z2))
                
                # Save Losses for plotting later
                G_losses.append(G_loss.item())
                D_losses.append(D_loss.item())
                col_loss.append(batch_loss_col.item())
                # class_loss.append(batch_loss_class.item())

                # ****************** checkpoint ******************** #
                if (batch_loss_col.item() < best_col) and (i >= 1000):
                    best_col = batch_loss_col.item()
                    # best_class = batch_loss_class.item()
                    torch.save(netG.state_dict(),
                            'checkpoints/{}/best/netG-epoch-{}-iter-{}-lossG-{:.5f}-{:.5f}.pth'.format(
                                work_name, epoch+1, i+1, G_loss.item(), batch_loss_col.item()))
                    torch.save(netD.state_dict(),
                            'checkpoints/{}/best/netD-epoch-{}-iter-{}-lossD-{:.5f}.pth'.format(
                                work_name, epoch+1, i+1, D_loss.item()))
                
                # Check how the generator is doing by saving G's output on fixed_noise
                if ((i+1) % 2000 == 0) or (i == len(train_loader) - 1):
                    dict1 = {"G_loss": G_losses, "D_loss": D_losses}
                    my_loss_plot(name=f'plot/{work_name}loss_plot_G&D', **dict1)
                    my_loss_plot(name=f'plot/{work_name}/loss_plot_col', color_loss=col_loss)
                    # loss_plot(name='plot/loss_plot_class', class_loss=class_loss)
                    # Visualization
                    img_list = []
                    with torch.no_grad():
                        for j in range(4):
                            img_list.append(gray_ab2rgb(img_gray[j].detach().cpu(), out_ab[j].detach().cpu()).transpose(2, 0, 1))
                        img_list = np.asarray(img_list)
                        try:
                            vis.images(img_list, win='Visualization', nrow=2)
                        except Exception as e:
                            print("visdom 其它错误")
                    # save
                    torch.save(netG.state_dict(),
                            'checkpoints/{}/netG-epoch-{}-iters-{}-lossG-{:.5f}-{:.5f}.pth'.format(
                                work_name, epoch+1, i+1, G_loss.item(), batch_loss_col.item()))
                    torch.save(netD.state_dict(),
                            'checkpoints/{}/netD-epoch-{}-iters-{}-lossD-{:.5f}.pth'.format(work_name, epoch+1, i+1, D_loss.item()))

        print('--------- Finishing training epoch {} ---------'.format(epoch + 1))

        # ****************** validation ******************* #
        # plot col_loss & picture val
        # print score
        print("--------- Starting validation Loop... ---------")
        print('Start validation epoch {}'.format(epoch+1))
        with torch.no_grad():
            netG.eval()
            netD.eval()
            batch_time, losses = Meter(), Meter()
            time.sleep(0.5)
            with tqdm(total=len(val_loader)) as pbar:
                for i, (img_gray, img_ab, target) in enumerate(val_loader, 0):
                    start_time = time.time()
                    pbar.set_description(f'batch[{i + 1}]')
                    img_gray, img_ab, target = img_gray.to(device), img_ab.to(device), target.to(device)
                    b_size = img_gray.size(0)
                    out_ab, _= netG(img_gray)
                    # if use_global_feature:
                    #     out_ab, _= netG(img_gray)
                    # else:
                    #     out_ab = netG(img_gray)
                    D_fake = netD(torch.cat((img_gray, out_ab), dim=1)).view(-1)
                    # 鉴别器对假判别为真的平均分数
                    D_G_z1 = D_fake.mean().item()
                    batch_loss_col = criterion_col(out_ab, img_ab)
                    losses.update(batch_loss_col.item(), b_size)
                    batch_time.update(time.time() - start_time, 1)
                    # 每个epoch,每8个批次存1张图片
                    if save_images and (i + 1) % 50 == 0:
                        for j in range(min(len(img_ab), 1)):
                            save_path = {'colorized': f'checkpoints/{work_name}/color/',
                                         'gray': f'checkpoints/{work_name}/gray/'}
                            save_name = 'img-{}-epoch-{}.jpg'.format(i * val_loader.batch_size + j, epoch + 1)
                            gray_ab2rgb(img_gray[j].detach().cpu(), out_ab[j].detach().cpu(),
                                        save_path=save_path, save_name=save_name)
                    # pbar.set_description(f't[{i+1}]: loss:{loss:2.5f}. acc:{acc:.5f}')
                    pbar.set_postfix(loss=batch_loss_col.item(), D_G_z1=D_G_z1)
                    pbar.update()
            time.sleep(0.5)
            val_loss.append(losses.avg)
        print('---------Finishing validation epoch {}\t'
              'Time:{batch_time.sum:.3f}\t'
              'Loss:{loss.avg:.5f} ---------'.format(epoch+1, batch_time=batch_time, loss=losses))

    # ****************** plot *******************************#
    # 绘制损失图
    dict_train = {"G_loss": G_losses, "D_loss": D_losses}
    dict_val = {"G_loss": val_loss}
    my_loss_plot(name=f'plot/{work_name}/loss_plot_G&D', **dict_train)
    my_loss_plot(x_label="epoch", name=f'plot/{work_name}/val_loss_epoch', **dict_val)
    my_loss_plot(name=f'plot/{work_name}/loss_plot_col', color_loss=col_loss)
    # loss_plot(name='plot/loss_plot_class', class_loss=class_loss)
    time.sleep(1)


def cal_gradient_penalty(netD, real, fake, device):
    # 每一个样本对应一个sigma。样本个数为real.size(0)，特征数...
    sigma = torch.rand((real.size(0), 1, 1, 1), device=device)
    sigma = sigma.expand(real.size())
    # 按公式计算x_hat
    x_hat = sigma * real + (torch.tensor(1., device=device) - sigma) * fake
    x_hat.requires_grad = True
    # 为得到梯度先计算y
    d_x_hat = netD(x_hat)

    # 计算梯度,autograd.grad返回的是一个元组(梯度值，)
    gradients = torch.autograd.grad(outputs=d_x_hat, inputs=x_hat,
                                    grad_outputs=torch.ones(d_x_hat.size(), device=device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    # 利用梯度计算出gradient penalty
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


if __name__ == '__main__':
    main()
