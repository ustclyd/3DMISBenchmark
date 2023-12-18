# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import pandas as pd
import numpy as np
import torch
import pickle

from monai import data, transforms
from monai.data import load_decathlon_datalist
from torch.utils.data import Dataset

class DataLoaderArgs:
    pass


class AlignLabelShape(object):
    '''
    Align label shape to (n, c, d, h, w) from (n, 1, d, h, w)
    '''
    def __init__(self, num_classes=2):
        self.num_classes = num_classes

    def __call__(self,sample):

        # print('align image shape:', sample['image'].shape)
        label = sample['label']
        # print('pre label dtype:', torch.float32)
        # expand dims
        new_label = torch.zeros((self.num_classes,) + label.shape[1:], dtype= torch.float32)
        for z in range(1, self.num_classes):
            temp = (label==z).astype(torch.float32)
            new_label[z,...] = temp
        new_label[0,...] = torch.amax(new_label[1:,...],axis=0) == 0
        # print('new label dtype:', new_label.dtype)
        # convert to Tensor
        sample['label'] = torch.tensor(new_label.tolist())
        sample['image'] = torch.tensor(sample['image'].tolist())
        # print('aligned image shape:', sample['image'].shape)
        return sample


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_loader(args):
    data_dir = args.data_dir
    datalist_json = args.json_list
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.SpatialPadd(
                keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z)
            ),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
            transforms.ToTensord(keys=["image", "label"]),
            AlignLabelShape(num_classes=args.num_classes),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    if args.test_mode:
        test_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        test_ds = data.Dataset(data=test_files, transform=val_transform)
        test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=test_sampler,
            pin_memory=False,
            persistent_workers=False,
        )
        loader = test_loader
    else:
        datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
        # print("3d datalist", datalist)
        if args.use_normal_dataset:
            train_ds = data.Dataset(data=datalist, transform=train_transform)
        else:
            train_ds = data.CacheDataset(
                data=datalist, transform=train_transform, cache_num=24, cache_rate=1.0, num_workers=args.workers
            )
        train_sampler = Sampler(train_ds) if args.distributed else None
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=False,
            persistent_workers=False,
        )
        val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        val_ds = data.Dataset(data=val_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=val_sampler,
            pin_memory=False,
            persistent_workers=False,
        )
        loader = [train_loader, val_loader]

    return loader


class KBR_DataGenerator(Dataset):
    '''
    Custom Dataset class for data loader.
    Args：
    - path_list: list of file path
    - roi_number: integer or None, to extract the corresponding label
    - num_class: the number of classes of the label
    - transform: the data augmentation methods
    '''
    def __init__(self, data_dict_list, roi_number=None, num_class=2, transform=None):

        self.data_dict_list = data_dict_list
        self.roi_number = roi_number
        self.num_class = num_class
        self.transform = transform


    def __len__(self):
        return len(self.data_dict_list)


    def __getitem__(self,index):
        # Get image and label
        # image: (D,H,W) or (H,W) 
        # label: same shape with image, integer, [0,1,...,num_class]
        f = open(self.data_dict_list[index]['image'],'rb')
        image = pickle.load(f)
        f.close()

        f = open(self.data_dict_list[index]['label'],'rb')
        label = pickle.load(f)
        f.close()
        # image = hdf5_reader(self.path_list[index],'image')
        # label = hdf5_reader(self.path_list[index],'label')
        if self.roi_number is not None:
            if isinstance(self.roi_number,list):
                tmp_mask = np.zeros_like(label,dtype=np.float32)
                assert self.num_class == len(self.roi_number) + 1
                for i, roi in enumerate(self.roi_number):
                    tmp_mask[label == roi] = i+1
                label = tmp_mask
            else:
                assert self.num_class == 2
                label = (label==self.roi_number).astype(np.float32) 

        image = 1*image.reshape(1,256,256,256)
        label = 1*label.reshape(1,256,256,256)
        # print(type(image))
        # print(image.shape)
        sample = {'image':image, 'label':label}
        # Transform
        if self.transform is not None:
            sample = self.transform(sample)

        # print(type(sample['image']))
        # print(sample['image'].shape)
        return sample


def load_kbr_datalist(csv_path, data_list_key):

    df = pd.read_csv(csv_path)
    if data_list_key == 'training':
        df = df[df['split'] == 'training']
        
        image_list = df['train_filepath'].tolist()
        label = df['gt_filepath'].tolist()
        data_dict =  [{'image': image, 'label': label} for image, label in zip(image_list, label)]
    else:
        df = df[df['split'] == 'validation']
        
        image_list = df['train_filepath'].tolist()
        label = df['gt_filepath'].tolist()
        data_dict =  [{'image': image, 'label': label} for image, label in zip(image_list, label)]

    return data_dict





def get_kbr_loader(args):
    csv_path = "/staff/ydli/projects/3DMISBenchmark/UNet_data.csv"

    if args.test_mode:
        test_files = load_kbr_datalist(csv_path, "validation")
        test_ds = KBR_DataGenerator(test_files, roi_number=None, num_class=2, transform=None)
        test_loader = data.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=None,
            pin_memory=False,
            persistent_workers=False,
        )
        loader = test_loader
    else:
        datalist = load_kbr_datalist(csv_path, "training")
        # print("3d datalist", datalist)
       
        train_ds = KBR_DataGenerator(datalist, roi_number=None, num_class=2, transform=None)
       
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            sampler=None,
            pin_memory=False,
            persistent_workers=False,
        )
        val_files = load_kbr_datalist(csv_path, "validation")
        val_ds = KBR_DataGenerator(val_files, roi_number=None, num_class=2, transform=None)
        val_loader = data.DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=None,
            pin_memory=False,
            persistent_workers=False,
        )
        loader = [train_loader, val_loader]

    return loader
