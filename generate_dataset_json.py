import json
from utils import save_list_as_json
def generate_dataset_json(dataset_split_file, save_path, cur_fold=1, dataset='SegTHOR'):
    with open(dataset_split_file, 'r') as fp:
        dataset_split = json.load(fp)
    dataset_json = {'training':[],'validation':[]}
    for train_idx in dataset_split['fold'+str(cur_fold)]['train']:
        if dataset == 'LiTS':
            dataset_json['training'].append({'image': f'volume-{train_idx}.nii',  'label': f'segmentation-{train_idx}.nii'})
        elif dataset == 'SegTHOR':
            dataset_json['training'].append({'image': f'{train_idx}/{train_idx}.nii.gz',  'label': f'{train_idx}/GT.nii.gz'})
    for val_idx in dataset_split['fold'+str(cur_fold)]['val']:
        if dataset == 'LiTS':
            dataset_json['validation'].append({'image': f'volume-{val_idx}.nii',  'label': f'segmentation-{val_idx}.nii'})
        elif dataset == 'SegTHOR':
            dataset_json['validation'].append({'image': f'{val_idx}/{val_idx}.nii.gz',  'label': f'{val_idx}/GT.nii.gz'}) 
    save_list_as_json(dataset_json, save_path)
    
    
if __name__ == "__main__":
    dataset = 'SegTHOR'
    if dataset == 'LITS':
        generate_dataset_json('/staff/zzq/code/3DMISBenchmark/dataset/LiTS/dataset_split.json', '/staff/zzq/code/3DMISBenchmark/dataset/LiTS/dataset.json')
    if dataset == 'SegTHOR':
        generate_dataset_json('/staff/zzq/code/3DMISBenchmark/dataset/SegTHOR/dataset_split.json', '/staff/zzq/code/3DMISBenchmark/dataset/SegTHOR/dataset.json')
        