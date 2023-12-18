
import os
import json
import glob

from utils import get_path_with_annotation,get_path_with_annotation_ratio,get_path_with_column
from utils import get_weight_path


__disease__ = ['Cervical','Nasopharynx','Structseg_HaN','Structseg_THOR','SegTHOR','Lung']
# __2d_net__ = ['unet_2d']  old ./ckpt
__2d_net__ = ['unet','unet++','FPN','deeplabv3+','att_unet','res_unet'] # 1,2,3,4,5,6
__trans_net__ = ['UTNet','UTNet_encoder','TransUNet','ResNet_UTNet','SwinUNet'] # 7,8,9,10,11
__new_net__ = ['sfnet'] # 12
__encoder_name__ = ['simplenet','resnet18','resnet34','resnet50','se_resnet50', \
                   'resnext50_32x4d','timm-resnest14d','timm-resnest26d','timm-resnest50d', \
                    'efficientnet-b4', 'efficientnet-b5'] # 0 ~ 10

__new_encoder_name__ = ['swin_transformer','swinplusr18'] # 11, 12
__3d_net__ = ['unet_3d','da_unet','da_se_unet','res_da_se_unet', 'vnet_lite', 'UNETR', 'vnet'] # 13, 14, 15, 16, 17, 18
__mode__ = ['2d','2d_clean','3d']

json_path = {
    'Cervical':'/staff/shijun/torch_projects/Med_Seg/converter/dcm_converter/static_files/Cervical_Oar.json',
    'Nasopharynx':'/staff/shijun/torch_projects/Med_Seg/converter/dcm_converter/static_files/Nasopharynx_Oar.json',
    'Liver':'/staff/shijun/torch_projects/Med_Seg/converter/dcm_converter/static_files/Liver_Oar.json',
    'Stomach':'/staff/shijun/torch_projects/Med_Seg/converter/dcm_converter/static_files/Stomach_Oar.json',
    'Structseg_HaN':'/staff/zzq/code/3DMISBenchmark/dataset/StructSeg-HaN/Structseg_HaN.json',
    'Structseg_THOR':'/staff/shijun/torch_projects/Med_Seg/converter/nii_converter/static_files/Structseg_THOR.json',
    'HaN_GTV':'/staff/shijun/torch_projects/Med_Seg/converter/nii_converter/static_files/HaN_GTV.json',
    'THOR_GTV':'/staff/shijun/torch_projects/Med_Seg/converter/nii_converter/static_files/THOR_GTV.json',
    'SegTHOR':'/staff/zzq/code/3DMISBenchmark/dataset/SegTHOR/SegTHOR.json',
    'Covid-Seg':'/staff/shijun/torch_projects/Med_Seg/converter/nii_converter/static_files/Covid-Seg.json', # competition
    'Lung':'/staff/shijun/torch_projects/Med_Seg/converter/dcm_converter/static_files/Lung_Oar.json',
    'Lung_Tumor':'/staff/shijun/torch_projects/Med_Seg/converter/dcm_converter/static_files/Lung_Tumor.json',
    'Nasopharynx_Tumor':'/staff/shijun/torch_projects/Med_Seg/converter/dcm_converter/static_files/Nasopharynx_Tumor.json',
    'Cervical_Tumor':'/staff/shijun/torch_projects/Med_Seg/converter/dcm_converter/static_files/Cervical_Tumor.json',
    'EGFR':'/staff/shijun/torch_projects/Med_Seg/converter/dcm_converter/static_files/EGFR.json',
    'LITS':'/staff/zzq/code/3DMISBenchmark/dataset/LiTS/LITS.json', # competition
}


DISEASE = 'LITS' 
MODE = '3d'
NET_NAME = 'unet_3d_kbr'
ENCODER_NAME = 'resnet18' # not important for transformer model and 3d model 
VERSION = 'test_kbr_v1.0.0'

DEVICE = '1'
# True if use internal pre-trained model
# Must be True when pre-training and inference
PRE_TRAINED = False
# True if use external pre-trained model 
EX_PRE_TRAINED = True if 'pretrain' in VERSION else False
# True if use resume model
CKPT_POINT = False

FOLD_NUM = 5
# [1-FOLD_NUM]
CURRENT_FOLD = 1
GPU_NUM = len(DEVICE.split(','))


with open(json_path[DISEASE], 'r') as fp:
    DATASET_INFO = json.load(fp)

# Arguments for trainer initialization
#--------------------------------- single or multiple
ROI_NUMBER = None# or 1,2,...
NUM_CLASSES = DATASET_INFO['annotation_num'] + 1 # 2 for binary, more for multiple classes
if ROI_NUMBER is not None:
    if isinstance(ROI_NUMBER,list):
        NUM_CLASSES = len(ROI_NUMBER) + 1
        ROI_NAME = 'Part_{}'.format(str(len(ROI_NUMBER)))
    else:
        NUM_CLASSES = 2
        ROI_NAME = DATASET_INFO['annotation_list'][ROI_NUMBER - 1]
else:
    ROI_NAME = 'All'

try:
    SCALE = DATASET_INFO['scale'][ROI_NAME]
    MEAN = STD = None
except:
    SCALE = None
    MEAN = DATASET_INFO['mean_std']['mean']
    STD = DATASET_INFO['mean_std']['std']
#---------------------------------



#--------------------------------- others
INPUT_SHAPE =  (256,256,256) if MODE =='3d' else (512,512)# (128,256,256) #(512,512) #
BATCH_SIZE = 1 if MODE =='3d' else 32
BASE_LR = 1e-3
if NET_NAME in ['TransUNet']:
    BATCH_SIZE = 16
    BASE_LR = 1e-4
elif NET_NAME in ['UNETR']:
    BATCH_SIZE = 6
    BASE_LR = 1e-4

# CKPT_PATH = './ckpt/{}/{}/{}/{}/fold{}'.format('Covid-Seg',MODE,'v1.0','Lesion',str(CURRENT_FOLD))
CKPT_PATH = './new_ckpt/{}/{}/{}/{}/fold{}'.format(DISEASE,MODE,VERSION,ROI_NAME,str(CURRENT_FOLD))
WEIGHT_PATH = get_weight_path(CKPT_PATH)

print(f"WEIGHT_PATH:{WEIGHT_PATH}")

INIT_TRAINER = {
  'net_name':NET_NAME,
  'encoder_name':ENCODER_NAME,
  'lr':BASE_LR, 
  'n_epoch':120,
  'channels':1,
  'num_classes':NUM_CLASSES, 
  'roi_number':ROI_NUMBER, 
  'scale':SCALE,
  'input_shape':INPUT_SHAPE,
  'crop':0,
  'batch_size':BATCH_SIZE,
  'num_workers':16,
  'device':DEVICE,
  'pre_trained':PRE_TRAINED,
  'ex_pre_trained':EX_PRE_TRAINED,
  'ckpt_point':CKPT_POINT,
  'weight_path':WEIGHT_PATH,
  'use_moco':None if 'moco' not in VERSION else 'moco',
  'weight_decay': 0.00001,
  'momentum': 0.9,
  'gamma': 0.1,
  'milestones': [30,60,90],
  'T_max':5,
  'mean':MEAN,
  'std':STD,
  'topk':20,
  'use_fp16':False if NET_NAME in __trans_net__  or NET_NAME in __3d_net__ else True, #False if the machine you used without tensor core
  'mode':MODE,
 }
#---------------------------------

__loss__ = ['TopKLoss','DiceLoss','CEPlusDice','CELabelSmoothingPlusDice','OHEM','Cross_Entropy']
# Arguments when perform the trainer 
loss_index = 0 if len(VERSION.split('.')) == 2 else eval(VERSION.split('.')[-1].split('-')[0])
# LOSS_FUN = 'TopkCEPlusDice' if ROI_NUMBER is not None else __loss__[loss_index]
LOSS_FUN = 'Cross_Entropy'

print('>>>>> loss fun:%s'%LOSS_FUN)

SETUP_TRAINER = {
  'output_dir':'./ckpt/{}/{}/{}/{}'.format(DISEASE,MODE,VERSION,ROI_NAME),
  'log_dir':'./log/{}/{}/{}/{}'.format(DISEASE,MODE,VERSION,ROI_NAME),
  'optimizer':'AdamW',
  'loss_fun':LOSS_FUN,
  'class_weight':None,
  'lr_scheduler':'MultiStepLR', #'CosineAnnealingLR','CosineAnnealingWarmRestarts''MultiStepLR'
  'freeze_encoder':False,
  'get_roi': False if 'roi' not in VERSION else True
}
#---------------------------------


#--------------------------------- test path setting
TEST_PATH = None

if DISEASE in ['Cervical','Nasopharynx','Lung','Lung_Tumor']:
    if MODE == '2d_clean':
        TEST_PATH = get_path_with_annotation(info['2d_data']['test_csv_path'],'path',ROI_NAME)
    elif MODE == '2d':
        TEST_PATH = get_path_with_column(info['2d_data']['test_csv_path'],'path')

        