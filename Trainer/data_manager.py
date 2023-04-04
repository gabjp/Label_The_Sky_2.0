import numpy as np
import os

CLF_READY_FOLDER = os.path.abspath("..") + "/Data/ready/clf/"
UNL_READY_FOLDER = os.path.abspath("..") + "/Data/ready/unl/"
CLF_TYPES = ['image', 'tabular', 'wiseflags', 'class']
UNL_TYPES = ['image', 'tabular']
SPLITS = ['train', 'val', 'test']

print(os.path.abspath(".."))
print(os.path.exists(CLF_READY_FOLDER))
print(os.path.exists(CLF_READY_FOLDER + 'clf_90_5_5' + '_' +'image' +'_' +'train' +'.npy'))

def load_data(ds_name, is_clf):
    output = {}
    types = CLF_TYPES if is_clf else UNL_TYPES
    for split in SPLITS:
        for type in types:
            with open(CLF_READY_FOLDER + ds_name + '_' +type +'_' +split +'.npy', "r")as file:
                output[type +'_' +split] = np.load(file)
    return output

    