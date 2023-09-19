import os 

# cervical
# path = '/staff/shijun/dataset/Cervical_OAR/Cervical_v2_dicom'
# nasopharynx
# path = '/staff/shijun/dataset/Nasopharynx_Oar/NPC20201217'
# lung
# path = '/staff/shijun/dataset/Lung_Missing_structure'
# egfr
# path = '/acsa-med/radiology/EGFR/901-1010'
# lung_tumor
# path = '/staff/shijun/dataset/Lung_Tumor/missing17'
# stomach_oar
# path = '/staff/shijun/dataset/Stomach_Oar/raw_data'
# liver_oar
# path = '/staff/shijun/dataset/Liver_Oar/missing_data/liver'
path = '/staff/shijun/dataset/Nasopharynx_Oar/test_data'
entry = os.scandir(path)
for item in entry:
    print(item.name)
    new_name = item.name.split('_')[1]
    # new_name = item.name.split(' ')[0]
    if 'RTst' in item.name:
        new_name = new_name + '_RT'
    else:
        new_name = new_name + '_CT'
        # new_name = new_name
    print(new_name)
    os.rename(item.path,os.path.join(path,new_name))



# nasopharynx_tumor
# path = '/staff/shijun/dataset/Nasopharynx_Tumor/raw_data'
# path = '/staff/shijun/dataset/Stomach_Oar/raw_data/missing_data/stomach'
# entry_list = os.scandir(path)
# for entry in entry_list:
#     print('entry name is %s'%entry.name)
#     for item in os.scandir(entry):
#         print(item.name)
#         # new_name = entry.name
#         new_name = item.name.split('_')[1]
#         if 'RTst' in item.name:
#             new_name = new_name + '_RT'
#         else:
#             new_name = new_name + '_CT'
#             # new_name = new_name
#         print(new_name)
#         os.rename(item.path,os.path.join(path,new_name))