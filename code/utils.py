from datetime import datetime
import os

def get_curr_time():
    return datetime.now().strftime("%Y.%m.%d.%H.%M.%S")

def get_start_time():
    return _start_time if _start_time else get_curr_time()

def input_dir(filename, path = '../dataset'):
    '''Check if input directory exists and contains all needed files'''

    path = os.path.abspath(path)
    if not os.path.isdir(path):
        raise IOError('Incorrect input_dir specified: no such directory')
    for filename in datasets:
        dataset_path = os.path.join(path, dataset_name)
        if not os.path.exists(dataset_path):
            raise IOError('Incorrect input_dir specified:'
                          ' %s set file not found' % dataset_path)
    return path