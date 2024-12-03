import h5py
import os
import numpy as np
import shutil
lstamp = 152840
rstick = 155122
lprefix = os.environ['HOME'] + f'/imitation_learning_ros/src/imitation_learning/data/{lstamp}/'
rprefix = os.environ['HOME'] + f'/imitation_learning_ros/src/imitation_learning/data/{rstick}/'


def norm(prefix):
    src = prefix + 'training_data_all.hdf5'
    dst = prefix + 'training_data_all_normalized.hdf5'
    shutil.copy(src, dst)
    with h5py.File(dst, 'a') as a:
            if 'spread_list' in a:
                # Read the dataset into a NumPy array
                b = np.array(a['spread_list'])
                
                # Perform normalization
                b_min = np.min(b)
                b = b - b_min
                b = np.log(b + 1e-6)  # Apply logarithmic scaling
                b_min_log = np.min(b)
                b = b - b_min_log
                b_max = np.max(b)
                if b_max != 0:
                    b = b / b_max
                else:
                    print("Warning: Maximum value after normalization is 0. Skipping division.")
                
                # Overwrite the existing 'spread_list' dataset
                # del a['spread_list']  # Delete the original dataset
                a.create_dataset('conf', data=b)
    # a = h5py.File(dst, 'a')
    # for k in a.keys():
    #     print(np.array(k).shape)
    # b = np.array(a['spread_list'])
    # b -= np.min(b)
    # b = np.log(b+1e-6) # sets min val -> 6
    # b -= np.min(b)
    # b /= np.max(b)
    # del a['spread_list']
    # a.create_dataset('spread_list', data=b)
    # a.close()

# def stackLR(lprefix, rprefix, prenormalized=False):
#     suffix = '_normalized' if prenormalized else ''
#     l = h5py.File(lprefix + f'training_data_all_{suffix}.hdf5', 'r')
#     lf = h5py.File(lprefix + 'training_data_all_normalized.hdf5', 'r')

norm(lprefix)
norm(rprefix)