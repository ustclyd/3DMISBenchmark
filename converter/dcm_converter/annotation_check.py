import os
import glob
import pydicom
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import re
import difflib

def has_annotation_fuzzy(annotation, annotation_list):
    sim_list = []
    for anno in annotation_list:
        sim = difflib.SequenceMatcher(None, annotation, anno).quick_ratio()
        sim_list.append(sim)
    if np.max(sim_list) > 0.8:
        return True, np.argmax(sim_list)
    else:
        return False, -1

def has_annotation(annotation, annotation_list):
    annotation_list = [re.sub(r'[\s]*','',case.lower()) for case in annotation_list]
    annotation = re.sub(r'[\s]*','',annotation.lower())
    # more
    annotation_list = [case.replace('_','').replace('-','') for case in annotation_list]
    annotation = annotation.replace('_','').replace('-','')
    if len(annotation_list) == 1:
        if annotation in annotation_list:
            return True, annotation_list.index(annotation)
        else:
            return False, -1
    else:
        return has_annotation_fuzzy(annotation,annotation_list)

# CT and RT in different folders
def annotation_check(input_path, save_path, annotation_list):

    info = []
    except_id = []
    except_id = []

    patient_id = set([case.split('_')[0] for case in os.listdir(input_path)])

    for ID in tqdm(patient_id):
        info_item = []
        info_item.append(ID)

        index_list = list(np.zeros((len(annotation_list), ), dtype=np.int8))

        rt_path = glob.glob(os.path.join(input_path, '*' + ID + '*RT*'))[0]
        rt_slice = glob.glob(os.path.join(rt_path, '*.dcm'))[0]
        try:
            structure = pydicom.read_file(rt_slice)
        except:
            except_id.append(ID)
            print('RT Error:%s'%ID)
            continue    
        else:
            for i in range(len(structure.ROIContourSequence)):
                info_item.append(structure.StructureSetROISequence[i].ROIName)
                flag, index = has_annotation(
                    structure.StructureSetROISequence[i].ROIName, annotation_list)
                if flag:
                    try:
                        _ = [
                            s.ContourData for s in
                            structure.ROIContourSequence[i].ContourSequence
                        ]
                    except Exception:
                        break
                    else:
                        index_list[index] = index_list[index] + 1
            if not (np.min(index_list) == 1 and np.max(index_list) == 1):
                except_id.append(ID)
                lack_list = []
                for i in range(len(annotation_list)):
                    if index_list[i] != 1:
                        lack_list.append(annotation_list[i])
                print('%s without annotations:'%ID,lack_list)
                # print(lack_list)
            info_item.sort()
            info.append(info_item)

    info_csv = pd.DataFrame(data=info)
    if os.path.exists(save_path):
        info_csv.to_csv(save_path, index=False, header=None, mode='a')
    else:
        info_csv.to_csv(save_path, index=False)

    print(except_id)
    print(len(except_id))
    print(except_id)



# CT and RT in the same folders
def annotation_check_v2(input_path, save_path, annotation_list):

    info = []
    except_id = []
    error_id = []

    patient_id = os.listdir(input_path)

    for ID in tqdm(patient_id):
        # print(ID)
        info_item = []
        info_item.append(ID)

        index_list = list(np.zeros((len(annotation_list), ), dtype=np.int8))

        series_path = os.path.join(input_path, ID)
        
        try:
            rt_slice = glob.glob(os.path.join(series_path, 'RTSTRUCT*'))[0]
            structure = pydicom.read_file(rt_slice,force=True)
        except:
            error_id.append(ID)
            print('RT Error:%s'%ID)
            continue    
        else:
            for i in range(len(structure.ROIContourSequence)):
                # print(structure.StructureSetROISequence[i].ROIName)
                info_item.append(structure.StructureSetROISequence[i].ROIName)
                flag, index = has_annotation(
                    structure.StructureSetROISequence[i].ROIName, annotation_list)
                if flag:
                    try:
                        _ = [
                            s.ContourData for s in
                            structure.ROIContourSequence[i].ContourSequence
                        ]
                    except Exception:
                        break
                    else:
                        index_list[index] = index_list[index] + 1
            # print(index_list)
            if not (np.min(index_list) == 1 and np.max(index_list) == 1):
                except_id.append(ID)
                lack_list = []
                for i in range(len(annotation_list)):
                    if index_list[i] != 1:
                        lack_list.append(annotation_list[i])
                print('%s without annotations:'%ID,lack_list)
                # print(lack_list)
            info_item.sort()
            info.append(info_item)

    info_csv = pd.DataFrame(data=info)
    if os.path.exists(save_path):
        info_csv.to_csv(save_path, index=False, header=None,  mode='a')
    else:
        info_csv.to_csv(save_path, index=False)

    print(except_id)
    print(len(except_id))
    print(error_id)

if __name__ == "__main__":
    # cervical_oar
    # json_file = './static_files/Cervical_Oar.json'
    # nasopharynx
    json_file = './static_files/Nasopharynx_Oar.json'
    # lung
    # json_file = './static_files/Lung_Oar.json'
    # lung tumor
    # json_file = './static_files/Lung_Tumor.json'
    # EGFR
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
    annotation_check(info['dicom_path'], info['annotation_path'],
                     info['annotation_list'])
