import os
import glob
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
import json


def metadata_reader(data_path):

    info = []
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(data_path)
    reader.SetFileNames(dicom_names)
    data = reader.Execute()
    size = list(data.GetSize()[:2])
    z_size = data.GetSize()[-1]
    thick_ness = data.GetSpacing()[-1]
    pixel_spacing = list(data.GetSpacing()[:2])
    info.append(size)
    info.append(z_size)
    info.append(thick_ness)
    info.append(pixel_spacing)
    return info

# CT and RT in different folders
def get_metadata(input_path, save_path):

    id_list = set([case.split('_')[0] for case in os.listdir(input_path)])
    info = []
    for ID in tqdm(id_list):
        info_item = [ID]
        series_path = glob.glob(os.path.join(input_path, '*' + ID + '*CT*'))[0]
        info_item.extend(metadata_reader(series_path))
        info.append(info_item)
    col = ['id', 'size', 'num', 'thickness', 'pixel_spacing']

    info_data = pd.DataFrame(columns=col,data=info)
    if os.path.exists(save_path):
        info_data.to_csv(save_path, index=False, header=None,  mode='a')
    else:
        info_data.to_csv(save_path, index=False)


# CT and RT in the same folders
def get_metadata_v2(input_path, save_path):

    id_list = os.listdir(input_path)
    info = []
    for ID in tqdm(id_list):
        info_item = [ID]
        series_path =os.path.join(input_path, ID)
        info_item.extend(metadata_reader(series_path))
        info.append(info_item)
    col = ['id', 'size', 'num', 'thickness', 'pixel_spacing']

    info_data = pd.DataFrame(columns=col,data=info)
    if os.path.exists(save_path):
        info_data.to_csv(save_path, index=False, header=None,  mode='a')
    else:
        info_data.to_csv(save_path, index=False)


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
    
    with open(json_file, 'r') as fp:
        info = json.load(fp)
    get_metadata(info['dicom_path'], info['metadata_path'])
