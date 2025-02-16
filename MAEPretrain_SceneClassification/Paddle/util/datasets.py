# https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/beginner/data_preprocessing_cn.html
# https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/beginner/data_load_cn.html
# --------------------------------------------------------

import os
import PIL
# from torch import NoneType

# from torchvision import datasets#, transforms
from paddle.io import Dataset
from paddle.vision import transforms

# from timm.data import create_transform
# from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


from PIL import Image,ImageFile
# from torch.utils import data
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import paddle

class NWPURESISCDataset(Dataset):
    def __init__(self, root, train=True, transform=None, split=None, tag=None):

        with open(os.path.join(root, 'train_labels_{}_{}.txt'.format(split,tag)), mode='r') as f:
            train_infos = f.readlines()
        f.close()

        trn_files = []
        trn_targets = []

        for item in train_infos:
            fname, fold, idx = item.strip().split()
            trn_files.append(os.path.join(root + '/',fold+'/', fname))
            trn_targets.append(int(idx))

        with open(os.path.join(root, 'valid_labels_{}_{}.txt'.format(split,tag)), mode='r') as f:
            valid_infos = f.readlines()
        f.close()

        val_files = []
        val_targets = []

        for item in valid_infos:
            fname, fold, idx = item.strip().split()
            val_files.append(os.path.join(root + '/',fold+'/', fname))
            val_targets.append(int(idx))

        if train:
            self.files = trn_files
            self.targets = trn_targets
        else:
            self.files = val_files
            self.targets = val_targets

        self.transform = transform

        print('Creating NWPU_RESISC45 dataset with {} examples'.format(len(self.targets)))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        img_path = self.files[i]

        img = Image.open(img_path)

        if self.transform != None:

            img = self.transform(img)

        return img, self.targets[i]

def build_dataset(is_train, args):
    transform = build_transform(args)

    print('Loading NWPU-RESISC45 dataset!')
    data_path = args.data_path
    args.nb_classes = 45
    dataset = NWPURESISCDataset(data_path, train=is_train, transform=transform, split=args.split, tag=args.tag)

    return dataset


def build_transform(args):
    mean = [0.485, 0.456, 0.406]#IMAGENET_DEFAULT_MEAN
    std = [0.229, 0.224, 0.225]#IMAGENET_DEFAULT_STD
    # train transform
    # if is_train:
        # this should always dispatch to transforms_imagenet_train
        # transform = create_transform(
        #     input_size=args.input_size,
        #     is_training=True,
        #     color_jitter=args.color_jitter,
        #     auto_augment=args.aa,
        #     interpolation='bicubic',
        #     re_prob=args.reprob,
        #     re_mode=args.remode,
        #     re_count=args.recount,
        #     mean=mean,
        #     std=std,
        # )
        # return transform

    # eval transform
    t = []
    # if args.input_size <= 224:
    crop_pct = 224 / 256
    # else:
    #     crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset_train=NWPURESISCDataset('/data/xiaolei.qin/Dataset/NWPU/',split='28',tag='100',train=False)
    print(len(dataset_train))
    print(dataset_train[0][0],dataset_train[0][1])
    plt.imshow(dataset_train[0][0])
    plt.show()
    n_pix = 0
    true_pix = 0
