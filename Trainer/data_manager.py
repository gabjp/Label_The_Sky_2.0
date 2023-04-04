import numpy as np

CLF_READY_FOLDER = "../Data/ready/clf/"
UNL_READY_FOLDER = "../Data/ready/unl/"
CLF_TYPES = ['images', 'tabular', 'wiseflags', 'class']
UNL_TYPES = ['images', 'tabular']
SPLITS = ['train', 'val', 'test']

def load_data(ds_name, is_clf):
    output = {}
    types = CLF_TYPES if is_clf else UNL_TYPES
    for split in SPLITS:
        for type in types:
            output[type +'_' +split] = np.load(CLF_READY_FOLDER + ds_name + '_' +type +'_' +split +'.npy')
    return output

    