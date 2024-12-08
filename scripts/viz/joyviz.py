import torch 
import numpy as np
import os
import h5py
from joyv5 import pret
from joyv4 import pret4
lstamp = 152840
rstamp = 155122
left = True
stamp = lstamp if left else rstamp
desc = "Only Left Stick" if left else "Both Left and Right Stick"
hand_data = os.environ['HOME'] + '/imitation_learning_ros/src/imitation_learning/data/' + str(stamp) + '/training_data_all.hdf5'
f = h5py.File(hand_data, 'r')
hand_actions = np.array(f['actions'])

train_prefix = os.environ['HOME'] + '/imitation_learning_ros/src/imitation_learning/logs/' + str(stamp) + '/'
coll_actions = torch.load(f'{train_prefix}collect_actions.pt').numpy()
# print(hand_actions.shape)
lstick = hand_actions[:, 0:2]
rstick = np.hstack((hand_actions[:, 2:], np.zeros((hand_actions.shape[0], 1))))
# pret4(lstick)
# pret()
pret(lstick, rstick, title=f'Expert Collected Joystick Data using {desc}')
# print("A")