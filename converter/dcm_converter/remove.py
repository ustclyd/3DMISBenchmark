import os 
import shutil

# nasopharynx
# path = '/staff/shijun/dataset/Nasopharynx_Oar'
# EGFR
path = "/acsa-med/radiology/EGFR"
entry = os.scandir(path)
bad_sample = ['0359', '0360', '0437', '0499', '0637', '0834', '0896']

for item in entry:
    if item.is_dir():
        # if item.name.split('_')[0] in bad_sample:
        if item.name in bad_sample:
            shutil.rmtree(item.path)


         


