from torch.utils import data
import torch
import os
from PIL import Image
import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import torch as t

class EvalDataset(data.Dataset):
    def __init__(self, img_root, label_root):
        lst_label = sorted(os.listdir(label_root))
        # print(label_root)
        lst_pred = sorted(os.listdir(img_root))
        # print(img_root)
        lst = []
        for name in lst_label:
            if name in lst_pred:
                lst.append(name)

        self.image_path = list(map(lambda x: os.path.join(img_root, x), lst))
        # print(self.image_path)
        self.label_path = list(map(lambda x: os.path.join(label_root, x), lst))
        # print(self.label_path)
        # print(self.image_path)
        # print(self.image_path.sort())

    def __getitem__(self, item):
        pred = Image.open(self.image_path[item]).convert('L')
        gt = Image.open(self.label_path[item]).convert('L')
        # print(self.image_path[item], self.label_path[item])
        if pred.size != gt.size:
            pred = pred.resize(gt.size, Image.BILINEAR)
        return pred, gt

    def __len__(self):
        return len(self.image_path)



class EvalDataset2(data.Dataset):
    def __init__(self, img_root, gt_root,mask_root):
        lst_gt = sorted(os.listdir(gt_root))
        lst_mask = sorted(os.listdir(mask_root))
        # print(label_root)
        lst_pred = sorted(os.listdir(img_root))
        # print(img_root)
        lst = []
        for name in lst_gt:
            if name in lst_pred:
                lst.append(name)

        self.image_path = list(map(lambda x: os.path.join(img_root, x), lst))
        # print(self.image_path)
        self.label_path = list(map(lambda x: os.path.join(gt_root, x), lst))
        self.mask_path = list(map(lambda x: os.path.join(mask_root, x), lst))
        # print(self.label_path)
        # print(self.image_path)
        # print(self.image_path.sort())

    def __getitem__(self, item):
        pred = Image.open(self.image_path[item]).convert('L')
        gt = Image.open(self.label_path[item]).convert('L')
        mask = Image.open(self.mask_path[item]).convert('L')
        # print(self.image_path[item], self.label_path[item])
        if pred.size != gt.size:
            pred = pred.resize(gt.size, Image.BILINEAR)

        return pred, gt, mask

    def __len__(self):
        return len(self.image_path)




class feature_dataset2(data.Dataset):
    def __init__(self, image_root, gt_root):
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.png')]
        self.gt = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        # self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gt = sorted(self.gt)

        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.ToTensor()])
        self.index = 0

    def load_data(self):
        # print(self.index)
        # print(self.images[self.index])
        image = self.binary_loader(self.images[self.index])
        gt = self.binary_loader(self.gt[self.index])

        image = self.img_transform(image).unsqueeze(0)
        gt = self.img_transform(gt).unsqueeze(0)


        name = self.images[self.index].split('/')[-1]

        self.index += 1
        return image,gt, name

    def filter_files(self):
        images = []
        for img_path in (self.images):
            # print(img_path)
            img = Image.open(img_path)

            images.append(img_path)

        self.images = images

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def __len__(self):
        return self.size


