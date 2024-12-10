import argparse
from itertools import count

import gym
import gym.spaces
import scipy.optimize
import numpy as np
import math
# import Argument
import args
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models2 import *
from replay_memory import Memory
from torch.autograd import Variable
from trpo import trpo_step
from utilsI import *
from loss import *

stamp = "131408"
file_prefix = os.environ['HOME'] + '/imitation_learning_ros/src/imitation_learning/logs/' + str(stamp) + '_2/'
if not os.path.exists(file_prefix):
    os.makedirs(file_prefix)

"""
2IWIL: proposed method (--weight)
GAIL (U+C): no need to specify option
GAIL (Reweight): use only confidence data (--weight --only)
GAIL (C): use only confidence data without reweighting (--weight --only --noconf)
"""

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# get parameters and the modification is made (by temitope)
# args = Argument.arguments()


env = gym.make(args.env)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
# print(num_inputs, num_actions)
# import sys
# sys.exit()
env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

policy_net = PolicyNet(num_inputs, num_actions, args.hidden_dim)
stamp = "131408"
file_prefix = os.environ['HOME'] + '/imitation_learning_ros/src/imitation_learning/logs/' + str(stamp) + '_2/'
i_episode = 9
agent_path = f'{file_prefix}agent_{i_episode}.pth'
checkpoint = torch.load(agent_path)
policy_net.load_state_dict(checkpoint['policy'])
policy_net.eval()

def evaluate(episode):
    avg_reward = 0.0
    for _ in range(args.eval_epochs):
        state = env.reset()
        for _ in range(10000): # Don't infinite loop while learning
            state = torch.from_numpy(state).unsqueeze(0)
            action, _, _ = policy_net(Variable(state))
            action = action.data[0].numpy()
            next_state, reward, done, _ = env.step(action)
            avg_reward += reward
            if done:
                break
            state = next_state
    print('Episode {}\tAverage reward: {:.2f}'.format(episode, avg_reward / args.eval_epochs))
    writer.log(episode, avg_reward / args.eval_epochs)

if args.only:
    fname = 'olabel'
else:
    fname = ''
if args.noconf:
    fname = 'nc'

writer = Writer(args.env, args.seed, args.weight, 'mixture', args.prior, args.traj_size, folder=args.ofolder, fname=fname, noise=args.noise, cutype=args.loss_type)

for i_episode in range(args.num_epochs):
   
    evaluate(i_episode)

   
