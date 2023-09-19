import os 
import h5py
import numpy as np

data_path = '/staff/shijun/dataset/Med_Seg/'


def float2int(data_path):
    for item in os.scandir(data_path):
        hdf5_file = h5py.File(item.path, 'r')
        image = np.asarray(hdf5_file['image'], dtype=np.int16)
        label = np.asarray(hdf5_file['label'], dtype=np.uint8)
        hdf5_file.close()

        os.remove(item.path)

        hdf5_file = h5py.File(item.path,'w')
        hdf5_file.create_dataset('image', data=image)
        hdf5_file.create_dataset('label', data=label)
        hdf5_file.close()



def dfs_convert(data_path):
    for sub_path in os.scandir(data_path):
        if sub_path.is_dir():
            dfs_convert(sub_path.path)
        else:
            float2int(data_path)
            break


for item in os.scandir(data_path):
    if item.name not in['Cervical_Oar','LIDC','Covid-Seg']:
        print(item.path)
        dfs_convert(item.path)