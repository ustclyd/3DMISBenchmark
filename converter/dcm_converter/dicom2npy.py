import sys
sys.path.append('..')
import os
import glob
from tqdm import tqdm
import time
import shutil
import json
import numpy as np

from dcm_converter.dicom_reader import Dicom_Reader
from converter.utils import save_as_hdf5

# dicom series and rt in different directories.
def dicom_to_hdf5(input_path, save_path, annotation_list, target_format, resample=True):
    if resample:
        save_path = save_path + '-resample'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # else:
    #     shutil.rmtree(save_path)
    #     os.makedirs(save_path)

    path_list = list(set([case.split('_')[0] for case in os.listdir(input_path)]))
    start = time.time()
    for ID in tqdm(path_list):
        print('***** %s in Processing *****'%ID)
        series_path = glob.glob(os.path.join(input_path, '*' + ID + '*CT*'))[0]
        rt_path = glob.glob(os.path.join(input_path, '*' + ID + '*RT*'))[0]
        rt_path = glob.glob(os.path.join(rt_path, '*.dcm'))[0]
        
        try:
            reader = Dicom_Reader(series_path, target_format, rt_path, annotation_list,trunc_flag=False, normalize_flag=False)
        except:
            print("Error data: %s" % ID)
            continue
        else:
            if resample:
                images = reader.get_resample_images().astype(np.int16)
                labels = reader.get_resample_labels().astype(np.uint8)
            else:
                images = reader.get_raw_images().astype(np.int16)
                labels = reader.get_raw_labels().astype(np.uint8)
            hdf5_path = os.path.join(save_path, ID + '.hdf5')

            save_as_hdf5(images, hdf5_path, 'image')
            save_as_hdf5(labels, hdf5_path, 'label')

    print("run time: %.3f" % (time.time() - start))


# dicom series and rt in the same directory.
def dicom_to_hdf5_v2(input_path, save_path, annotation_list, target_format, resample=True):
    if resample:
        save_path = save_path + '-resample'
        
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        shutil.rmtree(save_path)
        os.makedirs(save_path)

    path_list = os.listdir(input_path)
    start = time.time()
    for ID in tqdm(path_list):
        print('***** %s in Processing *****'%ID)
        series_path = os.path.join(input_path, ID)
        rt_path = glob.glob(os.path.join(series_path, 'RTSTRUCT*'))[0]
        
        try:
            reader = Dicom_Reader(series_path, target_format, rt_path, annotation_list,trunc_flag=False, normalize_flag=False, with_postfix=False)
        except:
            print("Error data: %s" % ID)
            continue
        else:
            if resample:
                images = reader.get_resample_images().astype(np.int16)
                labels = reader.get_resample_labels().astype(np.uint8)
            else:
                images = reader.get_raw_images().astype(np.int16)
                labels = reader.get_raw_labels().astype(np.uint8)

            hdf5_path = os.path.join(save_path, ID + '.hdf5')

            save_as_hdf5(images, hdf5_path, 'image')
            save_as_hdf5(labels, hdf5_path, 'label')

    print("run time: %.3f" % (time.time() - start))



if __name__ == "__main__":
    # cervical_oar
    # json_file = './static_files/Cervical_Oar.json'
    # nasopharynx
    json_file = './static_files/Nasopharynx_Oar.json'
    # lung
    # json_file = './static_files/Lung_Oar.json'
    # lung tumor
    # json_file = './static_files/Lung_Tumor.json'
    # egfr
    # json_file = './static_files/EGFR.json'
    # Nasopharynx_Tumor
    # json_file = './static_files/Nasopharynx_Tumor.json'
    # Cervical_Tumor
    # json_file = './static_files/Cervical_Tumor.json'
    # Stomach_Oar
    # json_file = './static_files/Stomach_Oar.json'
    # Liver_Oar
    # json_file = './static_files/Liver_Oar.json'
    # LIDC
    json_file = '/staff/zzq/code/3DMISBenchmark/dataset/LIDC/LIDC.json'
    with open(json_file, 'r') as fp:
        info = json.load(fp)
    # dicom_to_hdf5_v2(info['dicom_path'], info['npy_path'], info['annotation_list'], info['target_format'])
    dicom_to_hdf5(info['dicom_path'], info['npy_path'], info['annotation_list'], info['target_format'],resample=False)