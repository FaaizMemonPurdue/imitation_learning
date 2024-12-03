import h5py
import os
import numpy as np
stamp = 152840

prefix = os.environ['HOME'] + f'/imitation_learning_ros/src/imitation_learning/data/{stamp}/'
a = h5py.File(prefix + 'training_data_50.hdf5', 'r')
b = np.vstack((np.array(a['spread_list']), np.array(a['rewards'])))
print(a.keys())