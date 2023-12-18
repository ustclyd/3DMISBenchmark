# -*- encoding: utf-8 -*-
'''
@File    :   trainer.py
@Time    :   2021/11/29 16:50:24
@Author  :   Jun Shi 
@Version :   1.0
@Contact :   shijun18@mail.ustc.edu.cn
@License :   (C)Copyright 2019-2025, USTC-ACSA
'''

# here put the import lib

import os
import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import transforms
import numpy as np
import math
import shutil
from thop import profile

from torch.nn import functional as F

from data_utils.transformer_2d import Get_ROI,RandomFlip2D,RandomRotate2D,RandomErase2D,RandomAdjust2D,RandomDistort2D,RandomZoom2D,RandomNoise2D
from data_utils.transformer_3d import RandomTranslationRotationZoom3D,RandomFlip3D
from data_utils.data_loader import DataGenerator,CropResize,To_Tensor,Trunc_and_Normalize
import torch.distributed as dist
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
import warnings
warnings.filterwarnings('ignore')

from utils import dfs_remove_weight
from setproctitle import setproctitle
from split_train_val import get_cross_validation_by_sample
from data_utils.data_loader_3d import get_loader as get_loader_3d
from data_utils.data_loader_3d import get_kbr_loader, DataLoaderArgs
import glob

from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.transforms import AsDiscrete
from monai.utils.enums import MetricReduction
from monai.metrics import DiceMetric
from functools import partial
import time

# GPU version.

class SemanticSeg(object):
    '''
    Control the training, evaluation, and inference process.
    Args:
    - net_name: string
    - lr: float, learning rate.
    - n_epoch: integer, the epoch number
    - channels: integer, the channel number of the input
    - num_classes: integer, the number of class
    - input_shape: tuple of integer, input dim
    - crop: integer, cropping size
    - batch_size: integer
    - num_workers: integer, how many subprocesses to use for data loading.
    - device: string, use the specified device
    - pre_trained: True or False, default False
    - weight_path: weight path of pre-trained model
    '''
    def __init__(self,net_name=None,encoder_name=None,lr=1e-3,n_epoch=1,channels=1,num_classes=2,roi_number=1,scale=None, input_shape=None,crop=48,
                  batch_size=6,num_workers=0,device=None,pre_trained=False,ex_pre_trained=False,ckpt_point=True,weight_path=None,use_moco=None,
                  weight_decay=0.,momentum=0.95,gamma=0.1,milestones=[40,80],T_max=5,mean=None,std=None,topk=50,use_fp16=True, mode='2d'):
        super(SemanticSeg,self).__init__()

        self.net_name = net_name
        self.encoder_name = encoder_name
        self.lr = lr
        self.n_epoch = n_epoch
        self.channels = channels
        self.num_classes = num_classes
        self.roi_number = roi_number
        self.scale = scale
        self.input_shape = input_shape
        self.crop = crop
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.mode = mode
        
        self.pre_trained = pre_trained
        self.ex_pre_trained = ex_pre_trained
        self.ckpt_point = ckpt_point
        self.weight_path = weight_path
        self.use_moco = use_moco

        self.start_epoch = 0
        self.global_step = 0
        self.loss_threshold = 1.0
        self.metrics_threshold = 0.

        self.weight_decay = weight_decay
        self.momentum = momentum
        self.gamma = gamma
        self.milestones = milestones
        self.T_max = T_max
        self.mean = mean
        self.std = std
        self.topk = topk
        self.use_fp16 = use_fp16

        os.environ['CUDA_VISIBLE_DEVICES'] = self.device

        self.net = self._get_net(self.net_name)

        if self.pre_trained:
            self._get_pre_trained(self.weight_path,ckpt_point)


        if self.roi_number is not None:
            assert self.num_classes == 2, "num_classes must be set to 2 for binary segmentation"

        self.get_roi = False
        self.acc_func = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True)
        
        if self.mode == "3d":
            self.infer_net = partial(
                sliding_window_inference,
                roi_size=self.input_shape,
                sw_batch_size=1,
                predictor=self.net,
                overlap=0.5,
            )
            self.post_label = AsDiscrete(to_onehot=self.num_classes, n_classes=self.num_classes)
            self.post_pred = AsDiscrete(argmax=True, to_onehot=self.num_classes, n_classes=self.num_classes)
        else:
            self.infer_net = None
            self.post_label = None
            self.post_pred = None
        
    
    
    def stat_net(self):
        test_input_shape = (1, self.channels, ) + self.input_shape
        print('test shape:', test_input_shape)
        input_test = torch.randn(test_input_shape)
        flops, params = profile(self.net, inputs=(input_test,))
        print(flops, params)


    def trainer(self, dataset_info,cur_fold, fold_num, output_dir=None,log_dir=None,optimizer='Adam',
                  loss_fun='Cross_Entropy',class_weight=None,lr_scheduler=None,freeze_encoder=False,get_roi=False):

        torch.manual_seed(0)
        np.random.seed(0)
        torch.cuda.manual_seed_all(0)
        print('Device:{}'.format(self.device))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        output_dir = os.path.join(output_dir, "fold"+str(cur_fold))
        log_dir = os.path.join(log_dir, "fold"+str(cur_fold))

        if os.path.exists(log_dir):
            if not self.pre_trained:
                shutil.rmtree(log_dir)
                os.makedirs(log_dir)
        else:
            os.makedirs(log_dir)
        
        os.system(f"cp {os.path.join(os.path.dirname(__file__),'config.py')} {log_dir}")

        if os.path.exists(output_dir):
            if not self.pre_trained:
                shutil.rmtree(output_dir)
                os.makedirs(output_dir)
        else:
            os.makedirs(output_dir)
            
        
        self.writer = SummaryWriter(log_dir)
        

        net = self.net

        if freeze_encoder:
            for param in net.encoder.parameters():
                param.requires_grad = False
        
        self.get_roi = get_roi

        lr = self.lr
        loss = self._get_loss(loss_fun,class_weight)

        if len(self.device.split(',')) > 1:
            net = DataParallel(net)



        # dataloader setting
        [train_loader, val_loader] = self._prepare_data_loader(dataset_info, cur_fold=cur_fold, fold_num=fold_num)
            
        self.step_per_epoch = len(train_loader)
        print(f'length : train loader {len(train_loader)} val loader {len(val_loader)}')
        self.global_step = self.start_epoch * math.ceil(len(train_loader)/self.batch_size)
        # copy to gpu
        net = net.cuda()
        loss = loss.cuda()

        # optimizer setting
        optimizer = self._get_optimizer(optimizer,net,lr)

        scaler = GradScaler()

        # if self.pre_trained and self.ckpt_point:
        #     checkpoint = torch.load(self.weight_path)
        #     optimizer.load_state_dict(checkpoint['optimizer'])

        if lr_scheduler is not None:
            lr_scheduler = self._get_lr_scheduler(lr_scheduler,optimizer)


        # loss_threshold = 1.0
        early_stopping = EarlyStopping(patience=100,verbose=True,monitor='val_run_dice',op_type='max')
        for epoch in range(self.start_epoch,self.n_epoch):
            setproctitle(f'{self.net_name}:[{epoch+1}/{self.n_epoch}]')
            train_loss,train_dice,train_run_dice = self._train_on_epoch(epoch,net,loss,optimizer,train_loader,scaler)

            val_loss,val_dice,val_run_dice = self._val_on_epoch(epoch,net,loss,val_loader)

            if lr_scheduler is not None:
                lr_scheduler.step()

            # torch.cuda.empty_cache()

            print('epoch:{},train_loss:{:.5f},val_loss:{:.5f}'
              .format(epoch,train_loss,val_loss))

            print('epoch:{},train_dice:{:.5f},train_run_dice:{:.5f},val_dice:{:.5f},val_run_dice:{:.5f}'
              .format(epoch,train_dice,train_run_dice,val_dice,val_run_dice))


            self.writer.add_scalars(
              'data/loss',{'train':train_loss,'val':val_loss},epoch
            )
            self.writer.add_scalars(
              'data/dice',{'train':train_dice,'val':val_dice},epoch
            )
            self.writer.add_scalars(
                'data/run_dice', {'train': train_run_dice,'val': val_run_dice},epoch)
                
            self.writer.add_scalar(
              'data/lr',optimizer.param_groups[0]['lr'],epoch
            )

            '''
            if val_loss < self.loss_threshold:
                self.loss_threshold = val_loss
            '''
            early_stopping(val_run_dice)

            #save
            if val_dice > self.metrics_threshold:
                self.metrics_threshold = val_dice

                if len(self.device.split(',')) > 1:
                    state_dict = net.module.state_dict()
                else:
                    state_dict = net.state_dict()

                saver = {
                  'epoch':epoch,
                  'save_dir':output_dir,
                  'state_dict':state_dict,
                #   'optimizer':optimizer.state_dict()
                }

                file_name = 'epoch={}-train_loss={:.5f}-train_dice={:.5f}-train_run_dice={:.5f}-val_loss={:.5f}-val_dice={:.5f}-val_run_dice={:.5f}.pth'.format(
                    epoch,train_loss,train_dice,train_run_dice,val_loss,val_dice,val_run_dice)
                save_path = os.path.join(output_dir,file_name)
                print("Save as: %s" % file_name)

                torch.save(saver,save_path)
            
            #early stopping
            if early_stopping.early_stop:
                print("Early stopping")
                break

        self.writer.close()
        dfs_remove_weight(output_dir,3)

    def _train_on_epoch(self,epoch,net,criterion,optimizer,train_loader,scaler):

        net.train()

        train_loss = AverageMeter()
        train_dice = AverageMeter()

        from metrics import RunningDice
        run_dice = RunningDice(labels=range(self.num_classes),ignore_label=-1)
        s_time = time.time()
        for step,sample in enumerate(train_loader):

            data = sample['image']
            target = sample['label']

            data = data.cuda()
            target = target.cuda()
            data = data.float()
            # target = target.float()
            data = data.to(torch.float16)
            # target = target.to(torch.float16)
            with autocast(self.use_fp16):
                output = net(data)
                
                if isinstance(output,tuple):
                    output = output[0]
                print('shape check: data.shape, target.shape, output.shape, target.dtype, output.dtype/n')
                print('shape check', data.shape, target.shape, output.shape, target.dtype, output.dtype)
                print('output:')
                print(output)
                print('target:')
                print(target)
                # print('pause')
                # time.sleep(10)
                loss = criterion(output,target)

            optimizer.zero_grad()
            if self.use_fp16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            output = output.float()
            loss = loss.float()

            # measure dice and record loss
            dice = compute_dice(output.detach(),target)
            train_loss.update(loss.item(),data.size(0))
            train_dice.update(dice.item(),data.size(0))
            
            # measure run dice  
            print('before argmax:', output.shape, target.shape)
            output = torch.argmax(torch.softmax(output,dim=1),1).detach().cpu().numpy()  #N*H*W 
            target = torch.argmax(target,1).detach().cpu().numpy()
            print('after argmax:', output.shape, target.shape)
            run_dice.update_matrix(target,output)

            # torch.cuda.empty_cache()

            if self.global_step%10==0:
                rundice, dice_list = run_dice.compute_dice() 
                print("Category Dice: ", dice_list)
                print('epoch:{},step:{},train_loss:{:.5f},train_dice:{:.5f},run_dice:{:.5f},lr:{:.5f}'.format(epoch, step, loss.item(), dice.item(), rundice, optimizer.param_groups[0]['lr']))
                # run_dice.init_op()
                self.writer.add_scalars(
                  'data/train_loss_dice',{'train_loss':loss.item(),'train_dice':dice.item()},self.global_step
                )
            print(time.time()-s_time)
            s_time = time.time()
            self.global_step += 1

        # return train_loss.avg,run_dice.compute_dice()[0]
        return train_loss.avg,train_dice.avg,run_dice.compute_dice()[0]


    def _val_on_epoch(self,epoch,net,criterion,val_loader,val_transformer=None):

        net.eval()
        val_loss = AverageMeter()
        val_dice = AverageMeter()
        if self.mode == '3d':
            monai_dice = AverageMeter()

        from metrics import RunningDice
        run_dice = RunningDice(labels=range(self.num_classes),ignore_label=-1)

        with torch.no_grad():
            for step,sample in enumerate(val_loader):
                data = sample['image']
                target = sample['label']

                data = data.cuda()
                target = target.cuda()
                data = data.float()
                target = target.float()
                data = data.to(torch.float16)
                target = target.to(torch.float16)
                # print('val shape check', data.shape, target.shape, data.dtype, target.dtype)
                with autocast(self.use_fp16):
                    if self.infer_net is not None:
                        output = self.infer_net(data)
                        val_labels_list = decollate_batch(target)
                        val_labels_convert = [self.post_label(val_label_tensor) for val_label_tensor in val_labels_list]
                        val_outputs_list = decollate_batch(output)
                        val_output_convert = [self.post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
                        # print('val length', len(val_output_convert), val_output_convert[0].dtype)
                        output, target = torch.stack(val_output_convert), torch.stack(val_labels_convert)
                        # print('val output:', val_output_convert[0].shape,  output.shape, output.dtype)
                        monai_class_dice_score = self.acc_func(y_pred=val_output_convert, y=val_labels_convert).cpu().numpy()
                        monai_dice_score = np.nanmean(monai_class_dice_score[0,1:])
                        print("monai dice:",monai_class_dice_score,  monai_dice_score)
                        monai_dice.update(monai_dice_score, data.size(0))
                    else:
                        output = net(data)
                        if isinstance(output,tuple):
                            output = output[0]
                    # print('val shape check', data.shape, target.shape, output.shape, target.dtype, output.dtype)
                
                loss = criterion(output,target)
                output = output.float()
                loss = loss.float()

                # measure dice and record loss
                dice = compute_dice(output.detach(),target)
                
                val_loss.update(loss.item(),data.size(0))
                val_dice.update(dice.item(),data.size(0))
                

                # measure run dice  
                output = torch.argmax(torch.softmax(output,dim=1),1).detach().cpu().numpy()  #N*H*W 
                target = torch.argmax(target,1).detach().cpu().numpy()
                run_dice.update_matrix(target,output)

                # torch.cuda.empty_cache()

                if step % 10 == 0:
                    rundice, dice_list = run_dice.compute_dice() 
                    print("Category Dice: ", dice_list)
                    print('epoch:{},step:{},val_loss:{:.5f},val_dice:{:.5f},run_dice:{:.5f}'.format(epoch, step, loss.item(), dice.item(), rundice))
                    if self.mode == '3d':
                        print('average monai dice:{:.5f}'.format(monai_dice.avg))
                    # run_dice.init_op()

        # return val_loss.avg,run_dice.compute_dice()[0]
        return val_loss.avg,val_dice.avg,run_dice.compute_dice()[0]


    def _prepare_data_loader(self, dataset_info, cur_fold=1 , fold_num = 1, test_mode=False, test_path=None):
        
        if self.mode == '3d':
            # use monai-style 3d data loader.
            data_loader_args = DataLoaderArgs()
            data_loader_args.batch_size = self.batch_size 
            data_loader_args.workers = self.num_workers
            data_loader_args.data_dir = dataset_info['nii_path']
            data_loader_args.json_list = dataset_info['dataset_json_path']
            # data_loader_args.csc_dir = dataset_info['csv_path']
            data_loader_args.space_x = 1.0
            data_loader_args.space_y = 1.0
            data_loader_args.space_z = 1.0
            data_loader_args.a_min = self.scale[0]
            data_loader_args.a_max = self.scale[1]
            data_loader_args.b_min = 0.0
            data_loader_args.b_max = 1.0
            data_loader_args.roi_x = self.input_shape[0]
            data_loader_args.roi_y = self.input_shape[1]
            data_loader_args.roi_z = self.input_shape[2]
            data_loader_args.RandFlipd_prob = 0.2
            data_loader_args.RandRotate90d_prob = 0.2 
            data_loader_args.RandScaleIntensityd_prob = 0.1
            data_loader_args.RandShiftIntensityd_prob = 0.1
            data_loader_args.test_mode = test_mode
            data_loader_args.use_normal_dataset = False
            data_loader_args.distributed = False
            data_loader_args.num_classes = self.num_classes
            return get_kbr_loader(data_loader_args)# get_loader_3d(data_loader_args)
            
        else:
            if self.mode == '2d_clean':
                    assert self.roi_number is not None, "roi number must not be None in 2d clean"
                    if isinstance(self.roi_number,list):
                        roi_name = 'Part_{}'.format(str(len(self.roi_number)))
                    else:
                        roi_name = dataset_info['annotation_list'][self.roi_number - 1]
                    path_list = get_path_with_annotation(dataset_info['2d_data']['csv_path'],'path',roi_name)
            elif self.mode == '2d':
                path_list = glob.glob(os.path.join(dataset_info['2d_data']['save_path'],'*.hdf5'))
                dataset_split_path = dataset_info["dataset_split_path"]
                dataset_split = get_cross_validation_by_sample(path_list, from_file=dataset_split_path, fold_num=fold_num, mode=self.mode)
                train_path, val_path = dataset_split[cur_fold-1]
            else:
                raise ValueError("Fail to create dataloader, only 2d 2d_clean and 3d mode are available.")
            
            if test_mode:
                
                test_transformer = transforms.Compose([
                    Trunc_and_Normalize(self.scale),
                    Get_ROI(pad_flag=False) if self.get_roi else transforms.Lambda(lambda x:x),
                    CropResize(dim=self.input_shape,
                            num_class=self.num_classes,
                            crop=self.crop),
                    To_Tensor(num_class=self.num_classes)
                ])

                test_dataset = DataGenerator(test_path,
                                            roi_number=self.roi_number,
                                            num_class=self.num_classes,
                                            transform=test_transformer)

                test_loader = DataLoader(test_dataset,
                                        batch_size=16,
                                        shuffle=False,
                                        num_workers=2,
                                        pin_memory=True)

                return test_loader
            
            else:

                train_transformer = transforms.Compose([
                    Trunc_and_Normalize(self.scale),
                    Get_ROI(pad_flag=False) if self.get_roi else transforms.Lambda(lambda x:x),
                    CropResize(dim=self.input_shape,num_class=self.num_classes,crop=self.crop),
                    RandomErase2D(scale_flag=False),
                    RandomZoom2D(),
                    RandomDistort2D(),
                    RandomRotate2D(),
                    RandomFlip2D(mode='v'),
                    # RandomAdjust2D(),
                    RandomNoise2D(),
                    To_Tensor(num_class=self.num_classes)
                ])

                train_dataset = DataGenerator(train_path,roi_number=self.roi_number,num_class=self.num_classes,transform=train_transformer)

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    drop_last=True
                )
                
                val_transformer = transforms.Compose([
                    Trunc_and_Normalize(self.scale),
                    Get_ROI(pad_flag=False) if self.get_roi else transforms.Lambda(lambda x:x),
                    CropResize(dim=self.input_shape,num_class=self.num_classes,crop=self.crop),
                    To_Tensor(num_class=self.num_classes)
                ])

                val_dataset = DataGenerator(val_path,roi_number=self.roi_number,num_class=self.num_classes,transform=val_transformer)

                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    drop_last=True
                )
        
                return [train_loader, val_loader]

    def inference(self, dataset_info, test_path,net=None):

        if net is None:
            net = self.net

        net = net.cuda()
        net.eval()

        

        test_loader = self._prepare_data_loader(dataset_info, test_mode=True, test_path=test_path)
        test_dice = AverageMeter()
        from metrics import RunningDice
        run_dice = RunningDice(labels=range(self.num_classes),ignore_label=-1)

        with torch.no_grad():
            for step,sample in enumerate(test_loader):
                data = sample['image']
                target = sample['label']

                data = data.cuda()
                target = target.cuda() #N
                data = data.float()
                target = target.float()
                data = data.to(torch.float16)
                target = target.to(torch.float16)
                with autocast(self.use_fp16):
                    # print(type(data))
                    output = net(data)
                    if isinstance(output,tuple):
                        output = output[0]
                output = output.float() #N*C

                dice = compute_dice(output.detach(),target)
                test_dice.update(dice.item(),data.size(0))

                # measure run dice  
                output = torch.argmax(torch.softmax(output,dim=1),1).detach().cpu().numpy()  #N*H*W 
                target = torch.argmax(target,1).detach().cpu().numpy()
                run_dice.update_matrix(target,output)

                if step % 10 == 0:
                    rundice, dice_list = run_dice.compute_dice() 
                    print("Category Dice: ", dice_list)
                    print('step:{},test_dice:{:.5f},run_dice:{:.5f}'.format(step,dice.item(),rundice))
                    # run_dice.init_op()

                # torch.cuda.empty_cache()

        print('average test_dice:{:.5f}'.format(test_dice.avg))

        # return run_dice.compute_dice()[0]
        return test_dice.avg


    def _get_net(self,net_name):

        if net_name == 'unet_3d':
            from model.old_unet import unet_3d
            net = unet_3d(n_channels=self.channels,n_classes=self.num_classes)

        elif net_name == 'unet_3d_kbr':
            from model.old_unet import unet_3d_kbr
            net = unet_3d_kbr(n_channels=self.channels,n_classes=self.num_classes)

        elif net_name == 'da_unet':
            from model.da_unet import da_unet
            # print(self.input_shape[0])
            net = da_unet(init_depth=self.input_shape[0],n_channels=self.channels,n_classes=self.num_classes)
        
        elif net_name == 'vnet_lite':
            from model.vnet import vnet_lite
            net = vnet_lite(init_depth=self.input_shape[0],in_channels=self.channels,classes=self.num_classes)
        
        elif net_name == 'vnet':
            from model.vnet import vnet
            net = vnet(init_depth=self.input_shape[0],in_channels=self.channels,classes=self.num_classes)
        
        elif net_name == 'unet':
            if self.encoder_name in ['simplenet','swin_transformer','swinplusr18']:
                from model.unet import unet
                net = unet(
                    model_name=net_name,
                    encoder_name=self.encoder_name,
                    encoder_weights=self.use_moco,
                    in_channels=self.channels,
                    classes=self.num_classes,
                    aux_classifier=True
                )
            else:
                import segmentation_models_pytorch as smp
                net = smp.Unet(
                    encoder_name=self.encoder_name,
                    encoder_weights=None if not self.ex_pre_trained else 'imagenet',
                    in_channels=self.channels,
                    classes=self.num_classes,                     
                    aux_params={"classes":self.num_classes-1} 
                )
        elif net_name == 'unet++':
            if self.encoder_name is None:
                raise ValueError(
                    "encoder name must not be 'None'!"
                )
            else:
                import segmentation_models_pytorch as smp
                net = smp.UnetPlusPlus(
                    encoder_name=self.encoder_name,
                    encoder_weights=None if not self.ex_pre_trained else 'imagenet',
                    in_channels=self.channels,
                    classes=self.num_classes,                     
                    aux_params={"classes":self.num_classes-1} 
                )

        elif net_name == 'FPN':
            if self.encoder_name is None:
                raise ValueError(
                    "encoder name must not be 'None'!"
                )
            else:
                import segmentation_models_pytorch as smp
                net = smp.FPN(
                    encoder_name=self.encoder_name,
                    encoder_weights=None if not self.ex_pre_trained else 'imagenet',
                    in_channels=self.channels,
                    classes=self.num_classes,                     
                    aux_params={"classes":self.num_classes-1} 
                )
        
        elif net_name == 'deeplabv3+':
            if self.encoder_name in ['swinplusr18']:
                from model.deeplabv3plus import deeplabv3plus
                net = deeplabv3plus(
                    model_name=net_name,
                    encoder_name=self.encoder_name,
                    encoder_weights=self.use_moco,
                    in_channels=self.channels,
                    classes=self.num_classes
                )
            else:
                import segmentation_models_pytorch as smp
                net = smp.DeepLabV3Plus(
                    encoder_name=self.encoder_name,
                    encoder_weights=None if not self.ex_pre_trained else 'imagenet',
                    in_channels=self.channels,
                    classes=self.num_classes,                     
                    aux_params={"classes":self.num_classes-1} 
                )
        elif net_name == 'res_unet':
            from model.res_unet import res_unet
            net = res_unet(
                model_name=net_name,
                encoder_name=self.encoder_name,
                encoder_weights=self.use_moco,
                in_channels=self.channels,
                classes=self.num_classes
            )
        
        elif net_name == 'att_unet':
            from model.att_unet import att_unet
            net = att_unet(
                model_name=net_name,
                encoder_name=self.encoder_name,
                encoder_weights=self.use_moco,
                in_channels=self.channels,
                classes=self.num_classes
            )

        elif net_name == 'sfnet':
            from model.sfnet import sfnet
            net = sfnet(net_name,
                encoder_name=self.encoder_name,
                encoder_weights=self.use_moco,
                in_channels=self.channels,
                classes=self.num_classes
            )

        ## external transformer + Unet
        elif net_name == 'UTNet':
            from model.trans_model.utnet import UTNet
            net = UTNet(self.channels, base_chan=32, num_classes=self.num_classes, reduce_size=8, block_list='1234', num_blocks=[1,1,1,1], num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=False, maxpool=True)
        elif net_name == 'UTNet_encoder':
            from model.trans_model.utnet import UTNet_Encoderonly
            # Apply transformer blocks only in the encoder
            net = UTNet_Encoderonly(self.channels, base_chan=32, num_classes=self.num_classes, reduce_size=8, block_list='1234', num_blocks=[1,1,1,1], num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=False, maxpool=True)
        elif net_name =='TransUNet':
            from model.trans_model.transunet import VisionTransformer as ViT_seg
            from model.trans_model.transunet import CONFIGS as CONFIGS_ViT_seg
            config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
            config_vit.n_classes = self.num_classes 
            config_vit.n_skip = 3 
            config_vit.patches.grid = (int(self.input_shape[0]/16), int(self.input_shape[1]/16))
            net = ViT_seg(config_vit, img_size=self.input_shape[0], num_classes=self.num_classes)
            #net.load_from(weights=np.load('./initmodel/R50+ViT-B_16.npz')) # uncomment this to use pretrain model download from TransUnet git repo

        elif net_name == 'ResNet_UTNet':
            from model.trans_model.resnet_utnet import ResNet_UTNet
            net = ResNet_UTNet(self.channels, self.num_classes, reduce_size=8, block_list='1234', num_blocks=[1,1,1,1], num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True)
        
        elif net_name == 'SwinUNet':
            from model.trans_model.swin_unet import SwinUnet, SwinUnet_config
            config = SwinUnet_config()
            config.num_classes = self.num_classes
            config.in_chans = self.channels
            config.window_size = 8 if self.input_shape[0] in [128,256,512] else 7
            net = SwinUnet(config, img_size=self.input_shape[0], num_classes=self.num_classes)
            # net.load_from('./initmodel/swin_tiny_patch4_window7_224.pth')

        elif net_name == 'UNETR':
            from model.trans_model.unetr import UNETR
            net =  UNETR(
                    in_channels=self.channels,
                    out_channels = self.num_classes,
                    img_size=self.input_shape,
                    feature_size=16,
                    hidden_size=768,
                    mlp_dim=3072,
                    num_heads=12,
                    pos_embed='perceptron',
                    norm_name='instance',
                    conv_block=True,
                    res_block=True,
                    dropout_rate=0.0)
            # (
            #     in_channels: int,
            #     out_channels: int,
            #     img_size: Tuple[int, int, int],
            #     feature_size: int = 16,
            #     hidden_size: int = 768,
            #     mlp_dim: int = 3072,
            #     num_heads: int = 12,
            #     pos_embed: str = "perceptron",
            #     norm_name: Union[Tuple, str] = "instance",
            #     conv_block: bool = False,
            #     res_block: bool = True,
            #     dropout_rate: float = 0.0,
            # )
           

        return net


    def _get_loss(self,loss_fun,class_weight=None):
        if class_weight is not None:
            class_weight = torch.tensor(class_weight)

        if loss_fun == 'Cross_Entropy':
            from loss.cross_entropy import CrossentropyLoss
            loss = CrossentropyLoss(weight=class_weight)

        elif loss_fun == 'TopKLoss':
            from loss.cross_entropy import TopKLoss
            loss = TopKLoss(weight=class_weight,k=self.topk)
        
        elif loss_fun == 'CELabelSmoothingPlusDice':
            from loss.combine_loss import CELabelSmoothingPlusDice
            loss = CELabelSmoothingPlusDice(smoothing=0.1, weight=class_weight, ignore_index=0)

        elif loss_fun == 'OHEM':
            from loss.cross_entropy import OhemCELoss
            loss = OhemCELoss(thresh=0.7)

        elif loss_fun == 'DiceLoss':
            from loss.dice_loss import DiceLoss
            loss = DiceLoss(weight=class_weight, ignore_index=0, p=1)

        if loss_fun == 'DynamicTopKLoss':
            from loss.cross_entropy import DynamicTopKLoss
            loss = DynamicTopKLoss(weight=class_weight,step_threshold=self.step_per_epoch)
        
        elif loss_fun == 'PowDiceLoss':
            from loss.dice_loss import DiceLoss
            loss = DiceLoss(weight=class_weight, ignore_index=0, p=2)

        elif loss_fun == 'CEPlusDice':
            from loss.combine_loss import CEPlusDice
            loss = CEPlusDice(weight=class_weight, ignore_index=0)

        elif loss_fun == 'TopkCEPlusDice':
            from loss.combine_loss import TopkCEPlusDice
            loss = TopkCEPlusDice(weight=class_weight, ignore_index=0, k=self.topk)
        

        return loss


    # def _get_optimizer(self,optimizer,net,lr):
    #     if optimizer == 'Adam':
    #         optimizer = torch.optim.Adam(net.parameters(),lr=lr,weight_decay=self.weight_decay)

    #     elif optimizer == 'SGD':
    #         optimizer = torch.optim.SGD(net.parameters(),lr=lr,weight_decay=self.weight_decay,momentum=self.momentum)
        
    #     elif optimizer == 'AdamW':
    #         optimizer = torch.optim.AdamW(net.parameters(), lr=lr,weight_decay=self.weight_decay, amsgrad=False)
    #     return optimizer

    def _get_optimizer(self, optimizer, net, lr):
        """
        Build optimizer, set weight decay of normalization to 0 by default.
        """
        def check_keywords_in_name(name, keywords=()):
            isin = False
            for keyword in keywords:
                if keyword in name:
                    isin = True
            return isin

        def set_weight_decay(model, skip_list=(), skip_keywords=()):
            has_decay = []
            no_decay = []

            for name, param in model.named_parameters():
                # check what will happen if we do not set no_weight_decay
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                        check_keywords_in_name(name, skip_keywords):
                    no_decay.append(param)
                    # print(f"{name} has no weight decay")
                else:
                    has_decay.append(param)
            return [{'params': has_decay},
                    {'params': no_decay, 'weight_decay': 0.}]

        skip = {}
        skip_keywords = {}
        if hasattr(net, 'no_weight_decay'):
            skip = net.no_weight_decay()
        if hasattr(net, 'no_weight_decay_keywords'):
            skip_keywords = net.no_weight_decay_keywords()
        parameters = set_weight_decay(net, skip, skip_keywords)

        opt_lower = optimizer.lower()
        optimizer = None
        if opt_lower == 'sgd':
            optimizer = torch.optim.SGD(parameters, momentum=self.momentum, nesterov=True,
                                lr=lr, weight_decay=self.weight_decay)
        elif opt_lower == 'adamw':
            optimizer = torch.optim.AdamW(parameters, eps=1e-8, betas=(0.9, 0.999),
                                    lr=lr, weight_decay=self.weight_decay)
        elif opt_lower == 'adam':
            optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=self.weight_decay)

        return optimizer


    def _get_lr_scheduler(self,lr_scheduler,optimizer):
        if lr_scheduler == 'ReduceLROnPlateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                             mode='min',patience=5,verbose=True)
        elif lr_scheduler == 'MultiStepLR':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                             optimizer, self.milestones, gamma=self.gamma)
        elif lr_scheduler == 'CosineAnnealingLR':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                             optimizer, T_max=self.T_max)
        elif lr_scheduler == 'CosineAnnealingWarmRestarts':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer, 20, T_mult=2)
        return lr_scheduler


    def _get_pre_trained(self,weight_path, ckpt_point=True):
        checkpoint = torch.load(weight_path)
        self.net.load_state_dict(checkpoint['state_dict'])
        if ckpt_point:
            self.start_epoch = checkpoint['epoch'] + 1
            # self.metrics_threshold = eval(os.path.splitext(self.weight_path.split(':')[-1])[0])
            self.metrics_threshold = eval(self.weight_path.split('=')[-2].split('-')[0])




# computing tools

class AverageMeter(object):
    '''
  Computes and stores the average and current value
  '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def binary_dice(predict, target, smooth=1e-5):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1e-5
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
    # print('binary shape', predict.shape)
    predict = predict.contiguous().view(predict.shape[0], -1) #N，H*W
    target = target.contiguous().view(target.shape[0], -1) #N，H*W

    inter = torch.sum(torch.mul(predict, target), dim=1) #N
    union = torch.sum(predict + target, dim=1) #N

    dice = (2*inter + smooth) / (union + smooth ) #N
    
    # nan mean
    dice_index = dice != 1.0
    # print('dice ', dice, dice.dtype, dice.shape)
    # print('dice index: ', dice_index, dice_index.dtype, dice_index.shape)
    dice = dice[dice_index]

    return dice.mean()


def compute_dice(predict,target,ignore_index=0):
    """
    Compute dice
    Args:
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        ignore_index: class index to ignore
    Return:
        mean dice over the batch
    """
    assert predict.shape == target.shape, 'predict & target shape do not match'
    predict = F.softmax(predict, dim=1)
    # print('cd, pshape', predict.shape)
    onehot_predict = torch.argmax(predict,dim=1)#N*H*W
    onehot_target = torch.argmax(target,dim=1) #N*H*W
    # print('cd, onehot pshape', onehot_predict.shape)
    dice_list = -1.0 * np.ones((target.shape[1]),dtype=np.float32)
    for i in range(target.shape[1]):
        if i != ignore_index:
            if i not in onehot_predict and i not in onehot_target:
                continue
            bool_predict = (onehot_predict==i).float()
            # print('cd, bool pshape', bool_predict.shape)
            bool_target =  (onehot_target==i).float()
            dice = binary_dice(bool_predict, bool_target)
            dice_list[i] = round(dice.item(),4)
    dice_list = np.where(dice_list == -1.0, np.nan, dice_list)
    # print("cd dice list:", dice_list)
    dice_score = np.nanmean(dice_list)
    # print('compute dice:', dice_score)
    return dice_score




class EarlyStopping(object):
    """Early stops the training if performance doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=True, delta=0, monitor='val_loss',op_type='min'):
        """
        Args:
            patience (int): How long to wait after last time performance improved.
                            Default: 10
            verbose (bool): If True, prints a message for each performance improvement. 
                            Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            monitor (str): Monitored variable.
                            Default: 'val_loss'
            op_type (str): 'min' or 'max'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.monitor = monitor
        self.op_type = op_type

        if self.op_type == 'min':
            self.val_score_min = np.Inf
        else:
            self.val_score_min = 0

    def __call__(self, val_score):

        score = -val_score if self.op_type == 'min' else val_score

        if self.best_score is None:
            self.best_score = score
            self.print_and_update(val_score)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.print_and_update(val_score)
            self.counter = 0

    def print_and_update(self, val_score):
        '''print_message when validation score decrease.'''
        if self.verbose:
           print(self.monitor, f'optimized ({self.val_score_min:.6f} --> {val_score:.6f}).  Saving model ...')
        self.val_score_min = val_score