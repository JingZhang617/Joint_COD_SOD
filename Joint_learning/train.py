import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from datetime import datetime
import cv2
from model.ResNet_models import Generator
from data import get_loader, get_loader_cod
from utils import adjust_lr
import torchvision.transforms as transforms
from save_to_temp import visualize_cod_gt,visualize_cod_prediction,\
    visualize_sal_original_img,visualize_cod_original_img,visualize_sal_gt,visualize_sal_prediction,\
    visualize_cod_edge,visualize_sal_edge
import torch.optim.lr_scheduler as lr_scheduler




parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=30, help='epoch number')
parser.add_argument('--train_iters', type=int, default=20000, help='epoch number')
parser.add_argument('--lr_gen', type=float, default=2.5e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=28, help='training batch size')
parser.add_argument('--trainsize', type=int, default=480, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=20, help='every n epochs decay learning rate')
parser.add_argument('-beta1_gen', type=float, default=0.5,help='beta of Adam for generator')

opt = parser.parse_args()
print('Generator Learning Rate: {}'.format(opt.lr_gen))
# build models

generator=Generator()
generator.cuda()
generator_optimizer=torch.optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), opt.lr_gen, betas=[opt.beta1_gen, 0.999])
scheduler = lr_scheduler.StepLR(generator_optimizer,step_size=2000,gamma = 0.95)


image_root = '/home/jingzhang/jing_files/TPAMI/joint_cod_sod/DUTS_COD10553/img/'
gt_root = '/home/jingzhang/jing_files/TPAMI/joint_cod_sod/DUTS_COD10553/gt/'

cod_image_root = '/home/jingzhang/jing_files/RGBD_COD/dataset/train/Imgs/'
cod_gt_root = '/home/jingzhang/jing_files/RGBD_COD/dataset/train/GT/'

train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
cod_train_loader = get_loader_cod(cod_image_root, cod_gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)

sal_train_iter = iter(train_loader)
sal_it = 0
cod_train_iter = iter(cod_train_loader)
cod_it = 0

CE = torch.nn.BCEWithLogitsLoss()
MSE = torch.nn.MSELoss(size_average=True, reduce=True)
l1_loss = torch.nn.L1Loss(size_average = True, reduce = True)

fx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)
fy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).astype(np.float32)
fx = np.reshape(fx, (1, 1, 3, 3))
fy = np.reshape(fy, (1, 1, 3, 3))
fx = Variable(torch.from_numpy(fx)).cuda()
fy = Variable(torch.from_numpy(fy)).cuda()
contour_th = 1.5


def label_edge_prediction(label):
    # convert label to edge
    label = label.gt(0.5).float()
    label = F.pad(label, (1, 1, 1, 1), mode='replicate')
    label_fx = F.conv2d(label, fx)
    label_fy = F.conv2d(label, fy)
    label_grad = torch.sqrt(torch.mul(label_fx, label_fx) + torch.mul(label_fy, label_fy))
    label_grad = torch.gt(label_grad, contour_th).float()

    return label_grad

def label_edge_prediction_pre(label):
    # convert label to edge
    label = label.gt(0.2).float()
    label = F.pad(label, (1, 1, 1, 1), mode='replicate')
    label_fx = F.conv2d(label, fx)
    label_fy = F.conv2d(label, fy)
    label_grad = torch.sqrt(torch.mul(label_fx, label_fx) + torch.mul(label_fy, label_fy))

    return label_grad


def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()


def focal_loss(input,target,alpha=0.25,gamma=2):
    L=-target*alpha*((1-input)**gamma)*torch.log(input+0.00001)-\
      (1-target)*(1-alpha)*(input**gamma)*torch.log(1-input+0.00001)
    l=L.mean()
    # print(l.sum())
    return l.sum()

def laplacian_edge(img):
    laplacian_filter = torch.Tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    filter = torch.reshape(laplacian_filter, [1, 1, 3, 3])
    filter = filter.cuda()
    lap_edge = F.conv2d(img, filter, stride=1, padding=1)
    return lap_edge

def edgesm_loss_jing(pred, gt):

    edge = label_edge_prediction_pre(torch.sigmoid(pred))

    edge_loss = focal_loss(edge, gt)

    return edge_loss


def edgesm_loss(pred, gt):

    edge = torch.sigmoid(pred)

    edge_loss = CE(edge, gt)
    return edge_loss

def edgesm_loss_focal(pred, gt):

    edge = torch.sigmoid(pred)

    edge_loss = focal_loss(edge, gt)
    return edge_loss

print("go go go!!!!!!!!!!!")
for i in range(1,opt.train_iters+1):
    scheduler.step()

    # shared_optimizer.step() twice
    if sal_it >= len(train_loader):
        sal_train_iter = iter(train_loader)
        sal_it = 0
    sal_pack = sal_train_iter.next()
    sal_imgs, sal_gts  = sal_pack
    sal_imgs = Variable(sal_imgs)
    sal_gts = Variable(sal_gts)
    sal_imgs = sal_imgs.cuda()
    sal_gts = sal_gts.cuda()
    sal_it += 1

    generator_optimizer.zero_grad()
    sal_init,sal_ref = generator.forward(sal_imgs,True)
    # init loss
    sal_structure_loss1 = structure_loss(sal_init,sal_gts)
    sal_structure_loss2 = structure_loss(sal_ref, sal_gts)

    # total loss
    sal_loss=sal_structure_loss1+sal_structure_loss2

    sal_loss.backward()
    generator_optimizer.step()


    #######################################################################
    # shared_optimizer.step()
    if cod_it >= len(cod_train_loader):
        cod_train_iter = iter(cod_train_loader)
        voc_it = 0
    cod_pack = cod_train_iter.next()
    cod_imgs, cod_gts = cod_pack
    cod_imgs = Variable(cod_imgs)
    cod_gts = Variable(cod_gts)
    cod_imgs = cod_imgs.cuda()
    cod_gts = cod_gts.cuda()
    cod_it += 1


    generator_optimizer.zero_grad()
    cod_init, cod_ref = generator.forward(cod_imgs, False)
    # init loss
    cod_ce_loss1 = structure_loss(cod_init, cod_gts)
    cod_ce_loss2 = structure_loss(cod_ref, cod_gts)

    # total loss
    cod_loss = cod_ce_loss1 + cod_ce_loss2

    cod_loss.backward()
    generator_optimizer.step()


    if i % 10 == 0 or i == len(train_loader) or i == len(cod_train_loader):
        print('{} Step [{:04d}/{:04d}], sal Loss: {:.4f}, cod Loss: {:.4f}'.
              format(datetime.now(), i, opt.train_iters, sal_loss.data, cod_loss.data))
        visualize_sal_prediction(torch.sigmoid(sal_ref))
        visualize_cod_prediction(torch.sigmoid(cod_ref))
        visualize_sal_gt(sal_gts)
        visualize_cod_gt(cod_gts)

    save_path = 'models/joint/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if i % 2000 == 0:
        torch.save(generator.state_dict(), save_path + 'Model' + '_%d' % i + '_gen.pth')
