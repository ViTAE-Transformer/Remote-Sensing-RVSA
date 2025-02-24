import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class NWPURESISCDataset:
    def __init__(self, root, train=True, transform=None, split=None, tag=None):

        with open(os.path.join(root, 'train_labels_{}_{}.txt'.format(split, tag)), mode='r') as f:
            train_infos = f.readlines()
        f.close()

        trn_files = []
        trn_targets = []

        for item in train_infos:
            fname, fold, idx = item.strip().split()
            trn_files.append(os.path.join(root, fold, fname))
            trn_targets.append(int(idx))

        with open(os.path.join(root, 'valid_labels_{}_{}.txt'.format(split, tag)), mode='r') as f:
            valid_infos = f.readlines()
        f.close()

        val_files = []
        val_targets = []

        for item in valid_infos:
            fname, fold, idx = item.strip().split()
            val_files.append(os.path.join(root, fold, fname))
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
        img = self.transform(img)
        return img, self.targets[i]


def build_dataset(is_train, args):
    transform = build_transform(args)
    print('Loading NWPU-RESISC45 dataset!')
    data_path = args.data_path
    args.nb_classes = 45
    dataset = NWPURESISCDataset(data_path, train=is_train, transform=transform, split=args.split, tag=args.tag)

    return dataset


from mindspore.dataset.vision import Inter
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
def build_transform(args):
    t = []
    crop_pct = 224 / 256
    size = int(args.input_size / crop_pct)
    
    transform = transforms.Compose([
    vision.Resize(size, interpolation=Inter.BICUBIC),
    vision.CenterCrop(args.input_size),
    vision.ToTensor(),
    vision.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), is_hwc=False),
    ])
    
    return transform


if __name__ == '__main__':
    dataset_train=NWPURESISCDataset('./NWPU-RESISC45/',split='28',tag='100',train=False)
    print(len(dataset_train))
    print(dataset_train[0][0],dataset_train[0][1])
    n_pix = 0
    true_pix = 0

