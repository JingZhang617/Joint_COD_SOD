import os
import time
import os.path as osp
import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn
import argparse
from dataloader import EvalDataset

# os.environ["CUDA_VISIBLE_DEVICES"] = '3'

def Eval_mae(loader,cuda=True):
        #print('eval[MAE]:{} dataset with {} method.'.format(self.dataset, self.method))
        avg_mae, img_num, total = 0.0, 0.0, 0.0
        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            for pred, gt in loader:
                if cuda:
                    pred = trans(pred).cuda()
                    gt = trans(gt).cuda()
                else:
                    pred = trans(pred)
                    gt = trans(gt)
                mea = torch.abs(pred - gt).mean()
                #print(mea)
                #total = total+mea
                if mea == mea: # for Nan
                    avg_mae += mea
                    img_num += 1.0
            avg_mae /= img_num
            #avg = total / img_num
        return avg_mae

pred_dir0 =  '/home4/user_from_home1/liaixuan/joint_cod_sod_results/Resnet_joint_share_decoder2_1(focal)/'



test_datasets = ['ECSSD','DUT','DUTS','THUR','HKU-IS', 'SOC']
gt_dir = '/home1/liaixuan/datasets/GT/'
pred_dir=pred_dir0 + 'sod/'

for dataset in test_datasets:
    #print(dataset)
    loader = EvalDataset(osp.join(pred_dir, dataset), osp.join(gt_dir, dataset))
    mae = Eval_mae(loader=loader,cuda=True)
    print('dataset:{}  MAE:{:.4f} \n'.format(dataset,mae))
    #print('eval[MAE]:{:.4f} \n'.format(mae))


test_datasets = ['CAMO','CHAMELEON','COD10K']
gt_dir = '/home1/liaixuan/data/camouflage/COD_test/GT/'
pred_dir=pred_dir0 + 'cod/'

for dataset in test_datasets:
    #print(dataset)
    loader = EvalDataset(osp.join(pred_dir, dataset), osp.join(gt_dir, dataset))
    mae = Eval_mae(loader=loader,cuda=True)
    print('dataset:{}  MAE:{:.4f} \n'.format(dataset,mae))
    #print('eval[MAE]:{:.4f} \n'.format(mae))


