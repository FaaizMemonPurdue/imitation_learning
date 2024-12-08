import torch 
import numpy as np
import os
import h5py
from joyv6 import pret, pret4s
from joyv5 import pret5
def acs_to_sticks(actions):
    lstick = actions[:, 0:2]
    rstick = np.hstack((actions[:, 2:], np.zeros((actions.shape[0], 1))))
    return lstick, rstick
lstamp = 152840
rstamp = 155122
left = False
stamp = lstamp if left else rstamp
desc = "Only Left Stick" if left else "Both Left and Right Stick"
hand_data = os.environ['HOME'] + '/imitation_learning_ros/src/imitation_learning/data/' + str(stamp) + '/training_data_all.hdf5'
f = h5py.File(hand_data, 'r')
hand_actions = np.array(f['actions'])

train_prefix = os.environ['HOME'] + '/imitation_learning_ros/src/imitation_learning/logs/' + str(stamp) + '/'
# coll_actions = torch.load(f'{train_prefix}collect_actions.pt').numpy()
# print(hand_actions.shape)
handl, handr = acs_to_sticks(hand_actions)
pret(handl, handr, title=f'Expert Collected Joystick Data using {desc}')
# pret(handl, handr, title=f'Expert Collected Joystick Data using {desc}')
# coll1, coll2, coll3, coll4 = (coll_actions[0:2000], 
#                               coll_actions[5000:7000], 
#                               coll_actions[10000:12000], 
#                               coll_actions[15000:17000])

# coll1l, coll1r = acs_to_sticks(coll1)
# coll2l, coll2r = acs_to_sticks(coll2)
# coll3l, coll3r = acs_to_sticks(coll3)
# coll4l, coll4r = acs_to_sticks(coll4)

# pret(coll1l, coll1r, title=f'Collecting Sampled Actions over Steps 0-2000 {desc}')
# pret(coll2l, coll2r, title=f'Collecting Sampled Actions over Steps 5000-7000 {desc}')
# pret(coll3l, coll3r, title=f'Collecting Sampled Actions over Steps 10000-12000 {desc}')
# pret(coll4l, coll4r, title=f'Collecting Sampled Actions over Steps 15000-17000 {desc}')

eval_actions = np.load(f'{train_prefix}eval_actions.npz')
ev1, ev2, ev3, ev4 = (eval_actions['arr_0'],
                        eval_actions['arr_1'],
                        eval_actions['arr_2'],
                        eval_actions['arr_3'])
ev1l, ev1r = acs_to_sticks(ev1)
ev2l, ev2r = acs_to_sticks(ev2)
ev3l, ev3r = acs_to_sticks(ev3)
ev4l, ev4r = acs_to_sticks(ev4)
# pret4s([ev1l, ev2l, ev3l, ev4l], [ev1r, ev2r, ev3r, ev4r], title=f'Evaluation Actions over 4 different steps {desc}')
pret(ev1l, ev1r, title=f'Evaluation Actions after 5000 Exploratory Steps {desc}')
pret(ev2l, ev2r, title=f'Evaluation Actions after 10000 Exploratory Steps {desc}')
pret(ev3l, ev3r, title=f'Evaluation Actions after 15000 Exploratory Steps {desc}')
pret(ev4l, ev4r, title=f'Evaluation Actions after 20000 Exploratory Steps {desc}')