import json
from random import shuffle
from utils import save_list_as_json
import os
import json

def drop_slice_suffix(path):
    splitted_path = path.split('_')
    return '_'.join(splitted_path[:-1])

def drop_case_suffix(path):
    splitted_path = path.split('.')
    return splitted_path[0]


def get_cross_validation_by_sample(path_list, from_file, fold_num=5, random=True, mode='2d'): 
    
    k_fold_path_list = []
    sample_list = get_sample_list(path_list, mode=mode)
    split_by_sample = split_sample(sample_list, save_path=from_file, fold_num=fold_num, random=random)
    
    for train_val_split in split_by_sample.values():
        train_id = train_val_split['train']
        validation_id = train_val_split['val']
        if mode == '2d' or mode == '2d_clean':
            k_fold_path_list.append(get_slice_by_sample_id(path_list, train_id, validation_id))
        elif mode == '3d':
            k_fold_path_list.append(get_case_by_sample_id(path_list, train_id, validation_id))
        else:
            raise NotImplementedError("Only 2d and 3d data mode are available.")
    
    return k_fold_path_list


def get_cross_validation_sample_id(path_list, save_path, fold_num = 5): 
    result, split_by_sample = [], []
        
    sample_list = list(set([drop_slice_suffix(case) for case in path_list]))
    sample_list.sort()
    print('number of sample:',len(sample_list))
    
    if fold_num > 1:

        _len_ = len(sample_list) // fold_num

        for current_fold in range(1,fold_num+1):
            train_id, validation_id = [], []
            end_index = current_fold * _len_
            start_index = end_index - _len_
            if current_fold == fold_num:
                validation_id.extend(sample_list[start_index:])
                train_id.extend(sample_list[:start_index])
            else:
                validation_id.extend(sample_list[start_index:end_index])
                train_id.extend(sample_list[:start_index])
                train_id.extend(sample_list[end_index:])
            split_by_sample.append([train_id, validation_id])

    else:
        _len_ = len(sample_list) // 5
        train_id, validation_id = [], []
        end_index = current_fold * _len_
        start_index = end_index - _len_
        validation_id.extend(sample_list[:end_index])
        train_id.extend(sample_list[end_index:])
        split_by_sample.append([train_id, validation_id])
        
    misc.save_list_as_json(split_by_sample, save_path)
    return split_by_sample


def get_case_by_sample_id(path_list, train_id, validation_id):
    train_path, validation_path = [], []
    for case_path in path_list:
        if drop_case_suffix(os.path.basename(case_path)) in train_id:
            train_path.append(case_path)
        else:
            validation_path.append(case_path)
    print("Train set length:", len(train_path),
            "\nVal set length:", len(validation_path))
    return [train_path, validation_path]


def get_slice_by_sample_id(path_list, train_id, validation_id):
    train_path, validation_path = [], []
    for slice_path in path_list:
        if drop_slice_suffix(os.path.basename(slice_path)) in train_id:
            train_path.append(slice_path)
        else:
            validation_path.append(slice_path)
    print("Train set length:", len(train_path),
            "\nVal set length:", len(validation_path))
    return [train_path, validation_path]



def get_sample_list(image_list, mode='2d'):   
    if mode == '2d': 
        slice_list = image_list
        print('number of slice:', len(slice_list))
        path_list = [drop_slice_suffix(slice_path) for slice_path in slice_list]
    elif mode == '3d':
        print('number of 3d image:', len(image_list))
        path_list = [drop_case_suffix(case_path) for case_path in image_list]
        # print(path_list)
    
    sample_list = list(set(path_list))
        
    return sample_list

def split_sample(sample_list, save_path=None, fold_num=5, random=True):
    
    assert fold_num > 1, f"Fold should be larger than 1, current fold num is {fold_num}."
    print('number of sample:',len(sample_list))
    
    if os.path.exists(save_path):
        assert save_path.endswith(".json"), "Only JSON format is available."
        with open(save_path, 'r') as fp:
            split_by_sample = json.load(fp)
            print("Dataset split exists. Use current dataset split.")
            
    else:
        if random==True:
            shuffle(sample_list)
        else:
            sample_list.sort()
        
        split_by_sample = {}
        
        _len_ = len(sample_list) // fold_num
        
        for current_fold in range(1,fold_num+1):
            train_id, validation_id = [], []
            end_index = current_fold * _len_
            start_index = end_index - _len_
            if current_fold == fold_num:
                validation_id.extend(sample_list[start_index:])
                train_id.extend(sample_list[:start_index])
            else:
                validation_id.extend(sample_list[start_index:end_index])
                train_id.extend(sample_list[:start_index])
                train_id.extend(sample_list[end_index:])
            split_by_sample[f"fold{current_fold}"] = {"train":train_id, "val":validation_id}
        if save_path is not None:
            save_list_as_json(split_by_sample, save_path)
            print(f"Dataset split save at {save_path}.")
            
    return split_by_sample
    
def get_cross_validation_by_sample_v0(path_list, fold_num, current_fold):

    sample_list = list(set([os.path.basename(case).split('_')[0] for case in path_list]))
    sample_list.sort()
    print('number of sample:',len(sample_list))
    _len_ = len(sample_list) // fold_num

    train_id = []
    validation_id = []
    end_index = current_fold * _len_
    start_index = end_index - _len_
    if current_fold == fold_num:
        validation_id.extend(sample_list[start_index:])
        train_id.extend(sample_list[:start_index])
    else:
        validation_id.extend(sample_list[start_index:end_index])
        train_id.extend(sample_list[:start_index])
        train_id.extend(sample_list[end_index:])

    train_path = []
    validation_path = []
    for case in path_list:
        if os.path.basename(case).split('_')[0] in train_id:
            train_path.append(case)
        else:
            validation_path.append(case)

    random.shuffle(train_path)
    random.shuffle(validation_path)
    print("Train set length ", len(train_path),
          "Val set length", len(validation_path))
    return train_path, validation_path
        



if __name__ == "__main__":
    
    image_dir = '/staff/zzq/dataset/cv/med/StructSeg-HaN/2d_data'
    path_list = os.listdir(image_dir)
    get_cross_validation_by_sample(path_list, from_file="./dataset/StructSeg-HaN/dataset_split.json", fold_num=5, mode='2d')
    
    