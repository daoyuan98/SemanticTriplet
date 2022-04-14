import os
import json
import numpy as np 
import pickle as pkl

def get_similarity_matrix(dataset="cifar100", source="wordnet"):
    file_name = '{}_similarity_matrix_{}.npy'.format(dataset, source)
    if os.path.exists(file_name):
        print("returning sim mat: ", file_name)
        return np.load(file_name)