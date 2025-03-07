# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Adapted for AutoFocusFormer by Ziwen 2023

import os
import torch
import numpy as np
import utils
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
from timm.data.transforms import _str_to_pil_interpolation
import pandas as pd
from .samplers import SubsetRandomSampler

from torch.utils.data import dataset
import h5py,csv

class Loadh5Data(dataset.Dataset):
    def __init__(self,data_dir=None,prefix="traing",fold="1"):
        super(Loadh5Data, self).__init__()
        self.data_dir=data_dir
        for _ in range(3):
            data_dir = os.path.dirname(data_dir)
        # train_csv_path = os.path.join(data_dir,'label/tcga/fold_{}/train_data.csv'.format(str(fold)))
        # val_csv_path = os.path.join(data_dir,'label/tcga/fold_{}/val_data.csv'.format(str(fold)))
        csv_path = os.path.join(data_dir,'label/tcga/fold{}.csv'.format(str(fold-1)))
        self.slide_data = pd.read_csv(csv_path, index_col=0)
        self.label_dict = {}
        self.h5file=[]
        if prefix=='training':
            self.data = self.slide_data['train_slide_id'].dropna()
            self.survival_months = self.slide_data['train_survival_months'].dropna()
            self.censorship = self.slide_data['train_censorship'].dropna()
            self.case_id = self.slide_data['train_case_id'].dropna()
            self.label = self.slide_data['train_disc_label'].dropna()
        else:
            self.data = self.slide_data['test_slide_id'].dropna()
            self.survival_months = self.slide_data['test_survival_months'].dropna()
            self.censorship = self.slide_data['test_censorship'].dropna()
            self.case_id = self.slide_data['test_case_id'].dropna()
            self.label = self.slide_data['test_disc_label'].dropna()

    def __getitem__(self,index):
        hfile=h5py.File(os.path.join(self.data_dir,self.data[index]+'.h5'),'r')
        fe=np.array(hfile['features'])
        pos=np.array(hfile['coords'])/ 256
        # pos=torch.tensor([matrix])
        pos[:, 0]=pos[:, 0]-np.min(pos[:, 0])
        pos[:, 1]=pos[:, 1]-np.min(pos[:, 1])
        pos=np.ceil(pos)
        label= [self.label[index],self.survival_months[index],self.censorship[index]]
        # print("-".join((self.h5file[index]).split("-")[:3]))
        # print(self.h5file[index])
        return torch.Tensor(fe),torch.Tensor(pos),label
    
    def __len__(self):
        return len(self.data)

def build_loader(config,data_path):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config, data_path=data_path)
    config.freeze()
    print(f"local rank {config.LOCAL_RANK} / global rank {utils.get_rank()} successfully build train dataset")
    dataset_val, _ = build_dataset(is_train=False, config=config, data_path=data_path)
    print(f"local rank {config.LOCAL_RANK} / global rank {utils.get_rank()} successfully build val dataset")

    num_tasks = 1
    global_rank = utils.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    indices = np.arange(utils.get_rank(), len(dataset_val), 1)
    sampler_val = SubsetRandomSampler(indices)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return data_loader_train, data_loader_val, mixup_fn


def build_dataset(is_train, config,data_path):
    transform = build_transform(is_train, config)
    if config.DATA.DATASET == 'hipt':
        prefix = 'training' if is_train else 'validation'
        dataset = Loadh5Data(data_path,prefix,config.fold)
    else:
        prefix = 'training' if is_train else 'validation'
        dataset = Loadh5Data(data_path,prefix,config.fold)

    return dataset, 2


def build_transform_imagenet(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_str_to_pil_interpolation[config.DATA.INTERPOLATION]),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_str_to_pil_interpolation[config.DATA.INTERPOLATION])
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def build_transform(is_train, config):
    if config.DATA.DATASET == 'imagenet':
        return build_transform_imagenet(is_train, config)
    else:
        raise NotImplementedError("We only support ImageNet now.")
