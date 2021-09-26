import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from scipy import misc
from model.ResNet_models import Generator
from data import test_dataset
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=480, help='testing size')
opt = parser.parse_args()



save_root = './results/'
model_root= './models/joint/'


generator=Generator()
generator.load_state_dict(torch.load(model_root+'Model_20000_gen.pth'))


generator.cuda()
generator.eval()


sod_dataset_path = './RGB_Dataset/test/img/'
sod_test_datasets = ['DUTS_Test', 'DUT', 'ECSSD', 'HKU-IS', 'SOD', 'PASCAL']

cod_dataset_path = './RGB_COD/dataset/test/'
cod_test_datasets = ['CAMO','CHAMELEON','COD10K', 'NC4K']

for dataset in sod_test_datasets:
    save_path = save_root + 'sod/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_root = sod_dataset_path + dataset + '/'
    test_loader = test_dataset(image_root, opt.testsize)
    for i in range(test_loader.size):
        print(i)
        image, HH, WW, name = test_loader.load_data()
        image = image.cuda()
        _,sal_pred = generator.forward(image,is_sal=True)
        res = sal_pred
        res = F.upsample(res, size=[WW,HH], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = 255*(res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path+name, res)

for dataset in cod_test_datasets:
    save_path = save_root + 'cod/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_root = cod_dataset_path + dataset + '/Imgs/'
    test_loader = test_dataset(image_root, opt.testsize)
    for i in range(test_loader.size):
        print(i)
        image, HH, WW, name = test_loader.load_data()
        image = image.cuda()
        _,cod_pred = generator.forward(image,is_sal=False)
        res = cod_pred
        res = F.upsample(res, size=[WW,HH], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = 255*(res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path+name, res)

