import os,glob
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
import json


def metadata_reader(data_path):

    info = []
    data = sitk.ReadImage(data_path)
    # print(data)
    size = list(data.GetSize()[:2])
    z_size = data.GetSize()[-1]
    thick_ness = data.GetSpacing()[-1]
    pixel_spacing = list(data.GetSpacing()[:2])
    info.append(size)
    info.append(z_size)
    info.append(thick_ness)
    info.append(pixel_spacing)
    return info

# Different samples are saved in different folder
def get_metadata(input_path, save_path, image_postfix='data.nii.gz'):

    id_list = os.listdir(input_path)
    info = []
    for ID in tqdm(id_list):
        info_item = [ID]
        data_path = os.path.join(input_path, ID)
        image_path = glob.glob(os.path.join(data_path, image_postfix))[0]
        info_item.extend(metadata_reader(image_path))
        info.append(info_item)
    col = ['id', 'size', 'num', 'thickness', 'pixel_spacing']
    info_data = pd.DataFrame(columns=col, data=info)
    info_data.to_csv(save_path, index=False)

# All samples are saved in the same folder
def get_metadata_v2(input_path, save_path, image_postfix='data.nii.gz'):

    path_list = os.listdir(input_path)
    # the <<id_list>> needs to be customized
    if 'Covid-Seg' in input_path:
        id_list = set([case.split('_')[0].split('-')[-1] for case in path_list])
    elif 'LiTS' in input_path:
        id_list = set([case.split('-')[-1].split('.')[0] for case in path_list]) #LITS
    print(len(id_list))
    info = []
    for ID in tqdm(id_list):
        info_item = [ID]
        if 'Covid-Seg' in input_path:
            image_path = glob.glob(os.path.join(input_path, '*' + ID + image_postfix))[0]
        elif 'LiTS' in input_path:
            image_path = os.path.join(input_path,image_postfix.replace('*',ID)) # LITS
        info_item.extend(metadata_reader(image_path))
        info.append(info_item)
    col = ['id', 'size', 'num', 'thickness', 'pixel_spacing']
    info_data = pd.DataFrame(columns=col, data=info)
    info_data.to_csv(save_path, index=False)

if __name__ == "__main__":
    # HaN
    # json_file = './static_files/Structseg_HaN.json'
    # HaN_GTV
    json_file = './static_files/HaN_GTV.json'
    # THOR
    # json_file = './static_files/Structseg_THOR.json'
    # THOR_GTV
    # json_file = './static_files/THOR_GTV.json'
    # segthor
    # json_file = './static_files/SegTHOR.json'
    # covid-seg
    # json_file = './static_files/Covid-Seg.json'
    # json_file = './static_files/LITS.json'
    with open(json_file, 'r') as fp:
        info = json.load(fp)
    get_metadata(info['nii_path'], info['metadata_path'],image_postfix=info['image_postfix'])
    #get_metadata_v2("/staff/shijun/dataset/Covid-Seg/COVID-19-20_TestSet", './covid-test.csv',image_postfix=info['image_postfix'])
