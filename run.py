import os
import argparse
from trainer import SemanticSeg
import pandas as pd
import random

from config import INIT_TRAINER, SETUP_TRAINER, VERSION, CURRENT_FOLD, FOLD_NUM, MODE, TEST_PATH, DATASET_INFO
from split_train_val import get_cross_validation_by_sample
import time
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import warnings 
warnings.filterwarnings("ignore")

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--mode',
                        default='train-cross',
                        choices=["train", 'train-cross', "inf", "stat"],
                        help='choose the mode',
                        type=str)
    args = parser.parse_args()

    # Set data path & segnetwork
    if args.mode != 'train-cross':
        segnetwork = SemanticSeg(**INIT_TRAINER)
        # print(get_parameter_number(segnetwork.net))
        
    # Training
    ###############################################
    if args.mode == 'train-cross':
        for current_fold in range(1, FOLD_NUM + 1):
            print("=== Training Fold ", current_fold, " ===")
            segnetwork = SemanticSeg(**INIT_TRAINER)
            print(get_parameter_number(segnetwork.net))
            train_path, val_path = dataset_split[current_fold-1]
            SETUP_TRAINER['dataset_info'] = DATASET_INFO
            SETUP_TRAINER['cur_fold'] = current_fold
            SETUP_TRAINER['fold_num'] = FOLD_NUM
            start_time = time.time()
            segnetwork.trainer(**SETUP_TRAINER)

            print('run time:%.4f' % (time.time() - start_time))


    if args.mode == 'train':
        print("=== Training Fold ", CURRENT_FOLD, " ===")
        SETUP_TRAINER['dataset_info'] = DATASET_INFO
        SETUP_TRAINER['cur_fold'] = CURRENT_FOLD
        SETUP_TRAINER['fold_num'] = FOLD_NUM
        start_time = time.time()
        segnetwork.trainer(**SETUP_TRAINER)

        print('run time:%.4f' % (time.time() - start_time))
    ###############################################

    # Inference
    ###############################################
    elif args.mode == 'inf':
        test_path = TEST_PATH
        print('test len: %d'%len(test_path))
        start_time = time.time()
        result = segnetwork.inference(DATASET_INFO, test_path)
        print('run time:%.4f' % (time.time() - start_time))
        print('ave dice:%.4f' % (result))
    ###############################################

    else:
        print(get_parameter_number(segnetwork.net))
        segnetwork.stat_net()
        # from model.encoder.resnet import resnet18
        # import torch
        # from thop import profile
        # resnet_18 = resnet18(num_classes=2, n_channels=1, classification=True)
        # test_input_shape = (1, 1, 512,512 ) 
        # print('test shape:', test_input_shape)
        # input_test = torch.randn(test_input_shape)
        # flops, params = profile(resnet_18, inputs=(input_test,))
        # get_parameter_number(resnet18)
        # print(flops, params )
    print(f"output_dir:{SETUP_TRAINER['output_dir']}")
