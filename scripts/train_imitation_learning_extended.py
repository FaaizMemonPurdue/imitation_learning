#!/usr/bin/python3
fail_fast = False
watch_actions = True
import rclpy
from rclpy.node import Node
from rclpy import Parameter
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import EntityState, ModelStates
from gazebo_msgs.srv import SetEntityState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Empty
import math
import args

import math
import threading
import numpy as np
import time
import copy
import torch
from torch import optim
import os
import yaml
from tqdm import tqdm
import sys
import torch.nn.functional as F

from models2 import *
from replay_memory import Memory
from torch.autograd import Variable
from trpo import trpo_step
from utilsI import *
from loss import *

lstamp = "152840" #lstick
rstamp = "010607" #rstick
left = False
if left:
    stamp = lstamp
else:
    stamp = rstamp
file_prefix = os.environ['HOME'] + '/imitation_learning_ros/src/imitation_learning/logs/' + str(stamp) + '_2/'
if not os.path.exists(file_prefix):
    os.makedirs(file_prefix)

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True
torch.set_default_tensor_type('torch.DoubleTensor')

# don't try ros args, hardcode for now

robot_pose = np.array([-1.8, 1.8], float)
axes = np.array([0,0,0], float)
lidar_data = np.zeros(20)

device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
num_inputs = 23
num_actions = 3
torch.manual_seed(args.seed)
np.random.seed(args.seed)
policy_net = PolicyNet(num_inputs, num_actions, args.hidden_dim)
value_net = ValueNet(num_inputs, args.hidden_dim).to(device)
discriminator = Discriminator(num_inputs + num_actions, args.hidden_dim, args.initialization).to(device)
disc_criterion = nn.BCEWithLogitsLoss()
value_criterion = nn.MSELoss()
disc_optimizer = optim.Adam(discriminator.parameters(), args.lr)
value_optimizer = optim.Adam(value_net.parameters(), args.vf_lr)

def select_action(state):
    state = state.unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.tanh(torch.normal(action_mean, action_std))
    return action
def update_params(batch):
    rewards = torch.Tensor(batch.reward).to(device)
    masks = torch.Tensor(batch.mask).to(device)
    actions = torch.Tensor(np.concatenate(batch.action, 0)).to(device)
    states = torch.Tensor(np.array(batch.state)).to(device)
    values = value_net(Variable(states))

    returns = torch.Tensor(actions.size(0),1).to(device)
    deltas = torch.Tensor(actions.size(0),1).to(device)
    advantages = torch.Tensor(actions.size(0),1).to(device)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns)

    batch_size = math.ceil(states.shape[0] / args.vf_iters)
    idx = np.random.permutation(states.shape[0])
    for i in range(args.vf_iters):
        smp_idx = idx[i * batch_size: (i + 1) * batch_size]
        smp_states = states[smp_idx, :]
        smp_targets = targets[smp_idx, :]
        
        value_optimizer.zero_grad()
        value_loss = value_criterion(value_net(Variable(smp_states)), smp_targets)
        value_loss.backward()
        value_optimizer.step()

    advantages = (advantages - advantages.mean()) / advantages.std()

    action_means, action_log_stds, action_stds = policy_net(Variable(states.cpu()))
    fixed_log_prob = normal_log_density(Variable(actions.cpu()), action_means, action_log_stds, action_stds).data.clone()

    def get_loss():
        action_means, action_log_stds, action_stds = policy_net(Variable(states.cpu()))
        log_prob = normal_log_density(Variable(actions.cpu()), action_means, action_log_stds, action_stds)
        action_loss = -Variable(advantages.cpu()) * torch.exp(log_prob - Variable(fixed_log_prob))
        return action_loss.mean()


    def get_kl():
        mean1, log_std1, std1 = policy_net(Variable(states.cpu()))

        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    trpo_step(policy_net, get_loss, get_kl, args.max_kl, args.damping)

def expert_reward(states, actions):
    states = np.concatenate(states)
    actions = np.concatenate(actions)
    state_action = torch.Tensor(np.concatenate([states, actions], 1)).to(device)
    return -F.logsigmoid(discriminator(state_action)).cpu().detach().numpy()

def evaluate(episode, num_episodes):
    returns = []
    trajectories = []
    avg_reward = 0.0
    max_episode_steps = 100
    eval_met = {'suc': 0, 'timo': 0, 'ast': np.nan, 'col': 0}
    for _ in range(num_episodes):
        states = []
        actions = []
        rewards = []
        terminal = False
        state = gz_env.reset()
        t = 0
        # for _ in range(10000): # Don't infinite loop while learning
        while not terminal:
            state = state.unsqueeze(0)
            action, _, _ = policy_net(Variable(state))
            action = torch.clip(action.data[0], -1, 1).numpy()
            next_state, reward, terminal, collision, timo = gz_env.step(action, t, max_episode_steps)
            t += 1
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state
            if terminal:
                if timo:
                    eval_met['timo'] += 1
                elif collision:
                    eval_met['col'] += 1
                else:
                    if np.isnan(eval_met['ast']):
                        eval_met['ast'] = 0
                    eval_met['suc'] += 1
                    eval_met['ast'] += t
            avg_reward += reward
        trajectories.append((np.array(states), np.array(actions)))
        if eval_met['suc'] > 0:
            eval_met['ast'] /= eval_met['suc']
        returns.append(sum(rewards))            
    # writer.log(episode, avg_reward / args.eval_epochs)
    return returns, trajectories, eval_met, actions

plabel = ''
try:
    demo_pref = os.environ['HOME'] + '/imitation_learning_ros/src/imitation_learning/scripts'
    args.env = stamp
    expert_traj = np.load(demo_pref + "/{}/{}_mixture.npy".format(args.ifolder, args.env))
    expert_conf = np.load(demo_pref + "/{}/{}_mixture_conf.npy".format(args.ifolder, args.env))[:, np.newaxis] # 
    expert_conf += (np.random.randn(*expert_conf.shape) * args.noise)
    expert_conf = np.clip(expert_conf, 0.0, 1.0)
except:
    print('Mixture demonstrations not loaded successfully.')
    assert False

idx = np.random.choice(expert_traj.shape[0], args.traj_size, replace=False)
expert_traj = expert_traj[idx, :]
expert_conf = expert_conf[idx, :]


##### semi-confidence learning #####
num_label = int(args.prior * expert_conf.shape[0])

p_idx = np.random.permutation(expert_traj.shape[0])
expert_traj = expert_traj[p_idx, :]
expert_conf = expert_conf[p_idx, :]

if not args.only and args.weight:

    labeled_traj = torch.Tensor(expert_traj[:num_label, :]).to(device)
    unlabeled_traj = torch.Tensor(expert_traj[num_label:, :]).to(device)
    label = torch.Tensor(expert_conf[:num_label, :]).to(device)

    # classifier = Classifier(expert_traj.shape[1], 40).to(device)
    classifier = ClassifierWithAttention(expert_traj.shape[1], 40, args.initialization).to(device)
    optim = optim.Adam(classifier.parameters(), 3e-4, amsgrad=True)
    
    # Logic to check for the loss function type
    if args.loss_type == 'cu':
        cu_loss = CULoss(expert_conf, beta=1-args.prior, non=True) 
    elif args.loss_type == 'attentioncu':
        cu_loss = AttentionConfULoss(beta=1-args.prior, non=True)
    elif args.loss_type=='confU':
        cu_loss = ConfULoss(beta=1-args.prior, non=True)
    
    else:
        assert args.loss_type in ['cu', 'attentioncu', 'confU'], f"the loss function {args.loss_type} is not valid"
        
    batch = min(128, labeled_traj.shape[0])
    ubatch = int(batch / labeled_traj.shape[0] * unlabeled_traj.shape[0]) # same fraction of unlabeled data as we pulled from labeled data
    iters = 25000
    for i in range(iters):
        l_idx = np.random.choice(labeled_traj.shape[0], batch)
        u_idx = np.random.choice(unlabeled_traj.shape[0], ubatch)

        labeled = classifier(Variable(labeled_traj[l_idx, :]))
        unlabeled = classifier(Variable(unlabeled_traj[u_idx, :]))
        smp_conf = Variable(label[l_idx, :])

        optim.zero_grad()

        # handle various kind of loss
        if args.loss_type == 'attentioncu':
            attention_weight = classifier.get_raw_attention_weights()
            # print(attention_weight)
            risk = cu_loss(smp_conf, labeled, unlabeled, attention_weight)
            risk = risk.sum()
        else:
            risk = cu_loss(smp_conf, labeled, unlabeled)
        
        # print(risk)
        risk.backward()
        optim.step()
       
        if i % 1000 == 0:
            print('iteration: {}\tcu loss: {:.3f}'.format(i, risk.data.item()))

    classifier = classifier.eval()
    expert_conf = torch.sigmoid(classifier(torch.Tensor(expert_traj).to(device))).detach().cpu().numpy()
    expert_conf[:num_label, :] = label.cpu().detach().numpy()
    # save the classifier
    torch.save(classifier.state_dict(), f'{file_prefix}classifier.pth')

###################################
Z = expert_conf.mean()
if args.only:
    fname = 'olabel'
else:
    fname = ''
if args.noconf:
    fname = 'nc'

writer = Writer(args.env, args.seed, args.weight, 'mixture', args.prior, args.traj_size, folder=args.ofolder, fname=fname, noise=args.noise, cutype=args.loss_type)

class GazeboEnv(Node):

    def __init__(self, absorbing: bool, load_data: bool=False):
        super().__init__('env')
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.absorbing = absorbing

        #if absorbing:
        #    # Append absorbing indicator bit to state dimension (assumes 1D state space)
        #    self.observation_space = Box(low=np.concatenate([self.env.observation_space.low, np.zeros(1)]),
        #                                 high=np.concatenate([self.env.observation_space.high, np.ones(1)]))

        self.seed = 0
        self.wheel_vel1 = np.array([0,0,0,0], float)
        self.L = 0.125 # distance from the robot center to the wheel
        self.Rw = 0.03 # Radius ot the wheel
        
        self.set_state = self.create_client(SetEntityState, "/gazebo/set_entity_state")
        self.unpause = self.create_client(Empty, "/unpause_physics")
        self.pause = self.create_client(Empty, "/pause_physics")
        self.reset_world = self.create_client(Empty, "/reset_world")
        #self.reset_simulation = self.create_client(Empty, "/reset_simulation")
        self.req = Empty.Request

        self.publisher_robot_vel1 = self.create_publisher(Float64MultiArray, '/robot_1/forward_velocity_controller/commands', 10)
        
        self.set_box1_state = EntityState()
        self.set_box1_state.name = "box1"
        self.set_box1_state.pose.position.x = 0.0
        self.set_box1_state.pose.position.y = 0.0
        self.set_box1_state.pose.position.z = 0.1
        self.set_box1_state.pose.orientation.x = 0.0
        self.set_box1_state.pose.orientation.y = 0.0
        self.set_box1_state.pose.orientation.z = 0.0
        self.set_box1_state.pose.orientation.w = 1.0
        self.box1_state = SetEntityState.Request()

        self.set_box2_state = EntityState()
        self.set_box2_state.name = "box2"
        self.set_box2_state.pose.position.x = 0.0
        self.set_box2_state.pose.position.y = 0.0
        self.set_box2_state.pose.position.z = 0.1
        self.set_box2_state.pose.orientation.x = 0.0
        self.set_box2_state.pose.orientation.y = 0.0
        self.set_box2_state.pose.orientation.z = 0.0
        self.set_box2_state.pose.orientation.w = 1.0
        self.box2_state = SetEntityState.Request()

        self.set_box3_state = EntityState()
        self.set_box3_state.name = "box3"
        self.set_box3_state.pose.position.x = 0.0
        self.set_box3_state.pose.position.y = 0.0
        self.set_box3_state.pose.position.z = 0.1
        self.set_box3_state.pose.orientation.x = 0.0
        self.set_box3_state.pose.orientation.y = 0.0
        self.set_box3_state.pose.orientation.z = 0.0
        self.set_box3_state.pose.orientation.w = 1.0
        self.box3_state = SetEntityState.Request()

        self.set_box4_state = EntityState()
        self.set_box4_state.name = "box4"
        self.set_box4_state.pose.position.x = 0.0
        self.set_box4_state.pose.position.y = 0.0
        self.set_box4_state.pose.position.z = 0.1
        self.set_box4_state.pose.orientation.x = 0.0
        self.set_box4_state.pose.orientation.y = 0.0
        self.set_box4_state.pose.orientation.z = 0.0
        self.set_box4_state.pose.orientation.w = 1.0
        self.box4_state = SetEntityState.Request()

        self.set_box5_state = EntityState()
        self.set_box5_state.name = "box5"
        self.set_box5_state.pose.position.x = 0.0
        self.set_box5_state.pose.position.y = 0.0
        self.set_box5_state.pose.position.z = 0.1
        self.set_box5_state.pose.orientation.x = 0.0
        self.set_box5_state.pose.orientation.y = 0.0
        self.set_box5_state.pose.orientation.z = 0.0
        self.set_box5_state.pose.orientation.w = 1.0
        self.box5_state = SetEntityState.Request()

        self.set_box6_state = EntityState()
        self.set_box6_state.name = "box6"
        self.set_box6_state.pose.position.x = 0.0
        self.set_box6_state.pose.position.y = 0.0
        self.set_box6_state.pose.position.z = 0.1
        self.set_box6_state.pose.orientation.x = 0.0
        self.set_box6_state.pose.orientation.y = 0.0
        self.set_box6_state.pose.orientation.z = 0.0
        self.set_box6_state.pose.orientation.w = 1.0
        self.box6_state = SetEntityState.Request()

        self.set_box7_state = EntityState()
        self.set_box7_state.name = "box7"
        self.set_box7_state.pose.position.x = 0.0
        self.set_box7_state.pose.position.y = 0.0
        self.set_box7_state.pose.position.z = 0.1
        self.set_box7_state.pose.orientation.x = 0.0
        self.set_box7_state.pose.orientation.y = 0.0
        self.set_box7_state.pose.orientation.z = 0.0
        self.set_box7_state.pose.orientation.w = 1.0
        self.box7_state = SetEntityState.Request()

        self.set_box8_state = EntityState()
        self.set_box8_state.name = "box8"
        self.set_box8_state.pose.position.x = 0.0
        self.set_box8_state.pose.position.y = 0.0
        self.set_box8_state.pose.position.z = 0.1
        self.set_box8_state.pose.orientation.x = 0.0
        self.set_box8_state.pose.orientation.y = 0.0
        self.set_box8_state.pose.orientation.z = 0.0
        self.set_box8_state.pose.orientation.w = 1.0
        self.box8_state = SetEntityState.Request()

        self.set_box9_state = EntityState()
        self.set_box9_state.name = "box9"
        self.set_box9_state.pose.position.x = 0.0
        self.set_box9_state.pose.position.y = 0.0
        self.set_box9_state.pose.position.z = 0.1
        self.set_box9_state.pose.orientation.x = 0.0
        self.set_box9_state.pose.orientation.y = 0.0
        self.set_box9_state.pose.orientation.z = 0.0
        self.set_box9_state.pose.orientation.w = 1.0
        self.box9_state = SetEntityState.Request()

        self.set_box10_state = EntityState()
        self.set_box10_state.name = "box10"
        self.set_box10_state.pose.position.x = 0.0
        self.set_box10_state.pose.position.y = 0.0
        self.set_box10_state.pose.position.z = 0.1
        self.set_box10_state.pose.orientation.x = 0.0
        self.set_box10_state.pose.orientation.y = 0.0
        self.set_box10_state.pose.orientation.z = 0.0
        self.set_box10_state.pose.orientation.w = 1.0
        self.box10_state = SetEntityState.Request()

        self.set_box11_state = EntityState()
        self.set_box11_state.name = "box11"
        self.set_box11_state.pose.position.x = 0.0
        self.set_box11_state.pose.position.y = 0.0
        self.set_box11_state.pose.position.z = 0.1
        self.set_box11_state.pose.orientation.x = 0.0
        self.set_box11_state.pose.orientation.y = 0.0
        self.set_box11_state.pose.orientation.z = 0.0
        self.set_box11_state.pose.orientation.w = 1.0
        self.box11_state = SetEntityState.Request()

        self.set_box12_state = EntityState()
        self.set_box12_state.name = "box12"
        self.set_box12_state.pose.position.x = 0.0
        self.set_box12_state.pose.position.y = 0.0
        self.set_box12_state.pose.position.z = 0.1
        self.set_box12_state.pose.orientation.x = 0.0
        self.set_box12_state.pose.orientation.y = 0.0
        self.set_box12_state.pose.orientation.z = 0.0
        self.set_box12_state.pose.orientation.w = 1.0
        self.box12_state = SetEntityState.Request()

        self.set_box13_state = EntityState()
        self.set_box13_state.name = "box13"
        self.set_box13_state.pose.position.x = 0.0
        self.set_box13_state.pose.position.y = 0.0
        self.set_box13_state.pose.position.z = 0.1
        self.set_box13_state.pose.orientation.x = 0.0
        self.set_box13_state.pose.orientation.y = 0.0
        self.set_box13_state.pose.orientation.z = 0.0
        self.set_box13_state.pose.orientation.w = 1.0
        self.box13_state = SetEntityState.Request()

        self.set_box14_state = EntityState()
        self.set_box14_state.name = "box14"
        self.set_box14_state.pose.position.x = 0.0
        self.set_box14_state.pose.position.y = 0.0
        self.set_box14_state.pose.position.z = 0.1
        self.set_box14_state.pose.orientation.x = 0.0
        self.set_box14_state.pose.orientation.y = 0.0
        self.set_box14_state.pose.orientation.z = 0.0
        self.set_box14_state.pose.orientation.w = 1.0
        self.box14_state = SetEntityState.Request()

        #to move head_link to initial position
        self.set_robot_1_state = EntityState()
        self.set_robot_1_state.name = "robot_1"
        self.set_robot_1_state.pose.position.x = -1.8
        self.set_robot_1_state.pose.position.y = 1.8
        self.set_robot_1_state.pose.position.z = 0.15
        self.set_robot_1_state.pose.orientation.x = 0.0
        self.set_robot_1_state.pose.orientation.y = 0.0
        self.set_robot_1_state.pose.orientation.z = 0.0
        self.set_robot_1_state.pose.orientation.w = 1.0
        self.robot_1_state = SetEntityState.Request()                

        self.t = 0
        self.t_limit = 6000

        #self.obs_robot1 = np.array([0, 0], float)
        #self.done = False
        #self.actions = np.array([0,0,0], float)

        self.TIME_DELTA = 0.2
        self.timeouts = False
        self.next_obs = np.zeros(23)
        self.state_reset = np.zeros(23)
        self.goal_x = 1.8
        self.goal_y = -1.8
        self.rate = self.create_rate(1.0 / self.TIME_DELTA)

    def step(self, action, step, max_episode_steps):
        global lidar_data
        #self.done = False
        #self.actions[:] = axes[:]
        #obs = copy.copy(lidar_data)
        action = torch.Tensor(action).to('cpu').detach().numpy().copy()
        #self.get_logger().info(f"action:{action}")

        self.wheel_vel1[0] = (action[0]*math.sin(math.pi/4            ) + action[1]*math.cos(math.pi/4            ) + self.L*action[2])/self.Rw
        self.wheel_vel1[1] = (action[0]*math.sin(math.pi/4 + math.pi/2) + action[1]*math.cos(math.pi/4 + math.pi/2) + self.L*action[2])/self.Rw
        self.wheel_vel1[2] = (action[0]*math.sin(math.pi/4 - math.pi)   + action[1]*math.cos(math.pi/4 - math.pi)   + self.L*action[2])/self.Rw
        self.wheel_vel1[3] = (action[0]*math.sin(math.pi/4 - math.pi/2) + action[1]*math.cos(math.pi/4 - math.pi/2) + self.L*action[2])/self.Rw
    

        #publish robot1 commands
        #gz_env.publisher_vel1.publish(gz_env.vel_msg1)
        array_forPublish1_vel = Float64MultiArray(data=self.wheel_vel1)  
        self.publisher_robot_vel1.publish(array_forPublish1_vel)

        #'''
        while not gz_env.unpause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        try:
            self.unpause.call_async(Empty.Request())
        except:
            self.get_logger().info("/unpause_physics service call failed")

        # propagate state for TIME_DELTA seconds
        # time.sleep(self.TIME_DELTA)
        self.rate.sleep()
        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        try:
            self.pause.call_async(Empty.Request())
        except (rclpy.ServiceException) as e:
            self.get_logger().info("/gazebo/pause_physics service call failed")
        #'''

        self.next_obs[:20] = copy.copy(lidar_data)
        self.next_obs[20] = robot_pose[0] - self.goal_x
        self.next_obs[21] = robot_pose[1] - self.goal_y
        self.next_obs[22] = step
        dist = math.sqrt((robot_pose[0] - self.goal_x)**2 + (robot_pose[1] - self.goal_y)**2)
        reward = np.exp(-dist) # e^-0.35 * 100 = 70.46 from standing next to the goal
        mind = np.amin(self.next_obs[:20])
        collision = False
        if(dist <= 0.35):
            done = True 
            reward = 500 
            self.get_logger().info('Goal reached!')
        elif mind < 0.14: #could add collision listener but this p good
            reward = -30
            done = fail_fast
            collision = True
            self.get_logger().info('Collision!')
        elif mind < 0.25:
            reward = -1
            self.get_logger().info('Close to collision!')
            done = False
        else:
            done = False
        reward -= np.exp(step / max_episode_steps) * 50  # punish for taking too long by up to 50 (at timeout)
        # reward -= (step / max_episode_steps) * 50
        # exponential penalty was accumulating too much with time, so I added a linear decay to the reward
        # time out
        timo = False
        if step >= max_episode_steps:
            self.get_logger().info("time out")
            timo = True
            done = True

        next_obs_re = torch.tensor(self.next_obs, dtype=torch.double)#.unsqueeze(dim=0)  # Add batch dimension to state
        if self.absorbing:
            # Add absorbing indicator (zero) to state (absorbing state rewriting done in replay memory)
            next_obs_re = torch.cat([next_obs_re, torch.zeros(next_obs_re.size(0), 1)], dim=1) 

        return next_obs_re, reward, done, collision, timo #next_state, reward, terminal

    def reset(self):
        global axes, lidar_data
        #gz_env.get_logger().info('RESET!')
        '''
        while not self.reset_world.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('reset : service not available, waiting again...')

        try:
            #self.get_logger().info('Resetting the world')
            self.reset_world.call_async(Empty.Request())
        except:
            import traceback
            traceback.print_exc()
        '''
            
        #self.robot_1_state = SetEntityState.Request()
        '''
        self.robot_1_state._state = self.set_robot_1_state
        while not self.set_state.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('reset : service not available, waiting again...')

        try:
            self.set_state.call_async(self.robot_1_state)
        except rclpy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")
        '''

        rng = np.random.default_rng(self.seed)
        self.seed += 1
        boxes_pos = []
        for i in range(7):
            numbers = rng.choice(9, size=2, replace=False)
            boxes_pos.append(numbers)

        for j in range(7):
            if j==0:
                print(f"boxes0:{boxes_pos[j][0]}, {boxes_pos[j][1]}")
                if(boxes_pos[j][0] == 0):
                    self.set_box1_state.pose.position.x = -0.6
                    self.set_box1_state.pose.position.y = 2.4
                elif(boxes_pos[j][0] == 1):
                    self.set_box1_state.pose.position.x = 0.0
                    self.set_box1_state.pose.position.y = 2.4
                elif(boxes_pos[j][0] == 2):
                    self.set_box1_state.pose.position.x = 0.6
                    self.set_box1_state.pose.position.y = 2.4
                elif(boxes_pos[j][0] == 3):
                    self.set_box1_state.pose.position.x = -0.6
                    self.set_box1_state.pose.position.y = 1.8
                elif(boxes_pos[j][0] == 4):
                    self.set_box1_state.pose.position.x = 0.0
                    self.set_box1_state.pose.position.y = 1.8
                elif(boxes_pos[j][0] == 5):
                    self.set_box1_state.pose.position.x = 0.6
                    self.set_box1_state.pose.position.y = 1.8
                elif(boxes_pos[j][0] == 6):
                    self.set_box1_state.pose.position.x = -0.6
                    self.set_box1_state.pose.position.y = 1.2
                elif(boxes_pos[j][0] == 7):
                    self.set_box1_state.pose.position.x = 0.0
                    self.set_box1_state.pose.position.y = 1.2
                elif(boxes_pos[j][0] == 8):
                    self.set_box1_state.pose.position.x = 0.6
                    self.set_box1_state.pose.position.y = 1.2

                if(boxes_pos[j][1] == 0):
                    self.set_box2_state.pose.position.x = -0.6
                    self.set_box2_state.pose.position.y = 2.4
                elif(boxes_pos[j][1] == 1):
                    self.set_box2_state.pose.position.x = 0.0
                    self.set_box2_state.pose.position.y = 2.4
                elif(boxes_pos[j][1] == 2):
                    self.set_box2_state.pose.position.x = 0.6
                    self.set_box2_state.pose.position.y = 2.4
                elif(boxes_pos[j][1] == 3):
                    self.set_box2_state.pose.position.x = -0.6
                    self.set_box2_state.pose.position.y = 1.8
                elif(boxes_pos[j][1] == 4):
                    self.set_box2_state.pose.position.x = 0.0
                    self.set_box2_state.pose.position.y = 1.8
                elif(boxes_pos[j][1] == 5):
                    self.set_box2_state.pose.position.x = 0.6
                    self.set_box2_state.pose.position.y = 1.8
                elif(boxes_pos[j][1] == 6):
                    self.set_box2_state.pose.position.x = -0.6
                    self.set_box2_state.pose.position.y = 1.2
                elif(boxes_pos[j][1] == 7):
                    self.set_box2_state.pose.position.x = 0.0
                    self.set_box2_state.pose.position.y = 1.2
                elif(boxes_pos[j][1] == 8):
                    self.set_box2_state.pose.position.x = 0.6
                    self.set_box2_state.pose.position.y = 1.2

            if j==1:
                print(f"boxes2:{boxes_pos[j][0]}, {boxes_pos[j][1]}")
                if(boxes_pos[j][0] == 0):
                    self.set_box3_state.pose.position.x = 1.2
                    self.set_box3_state.pose.position.y = 2.4
                elif(boxes_pos[j][0] == 1):
                    self.set_box3_state.pose.position.x = 1.8
                    self.set_box3_state.pose.position.y = 2.4
                elif(boxes_pos[j][0] == 2):
                    self.set_box3_state.pose.position.x = 2.4
                    self.set_box3_state.pose.position.y = 2.4
                elif(boxes_pos[j][0] == 3):
                    self.set_box3_state.pose.position.x = 1.2
                    self.set_box3_state.pose.position.y = 1.8
                elif(boxes_pos[j][0] == 4):
                    self.set_box3_state.pose.position.x = 1.8
                    self.set_box3_state.pose.position.y = 1.8
                elif(boxes_pos[j][0] == 5):
                    self.set_box3_state.pose.position.x = 2.4
                    self.set_box3_state.pose.position.y = 1.8
                elif(boxes_pos[j][0] == 6):
                    self.set_box3_state.pose.position.x = 1.2
                    self.set_box3_state.pose.position.y = 1.2
                elif(boxes_pos[j][0] == 7):
                    self.set_box3_state.pose.position.x = 1.8
                    self.set_box3_state.pose.position.y = 1.2
                elif(boxes_pos[j][0] == 8):
                    self.set_box3_state.pose.position.x = 2.4
                    self.set_box3_state.pose.position.y = 1.2

                if(boxes_pos[j][1] == 0):
                    self.set_box4_state.pose.position.x = 1.2
                    self.set_box4_state.pose.position.y = 2.4
                elif(boxes_pos[j][1] == 1):
                    self.set_box4_state.pose.position.x = 1.8
                    self.set_box4_state.pose.position.y = 2.4
                elif(boxes_pos[j][1] == 2):
                    self.set_box4_state.pose.position.x = 2.4
                    self.set_box4_state.pose.position.y = 2.4
                elif(boxes_pos[j][1] == 3):
                    self.set_box4_state.pose.position.x = 1.2
                    self.set_box4_state.pose.position.y = 1.8
                elif(boxes_pos[j][1] == 4):
                    self.set_box4_state.pose.position.x = 1.8
                    self.set_box4_state.pose.position.y = 1.8
                elif(boxes_pos[j][1] == 5):
                    self.set_box4_state.pose.position.x = 2.4
                    self.set_box4_state.pose.position.y = 1.8
                elif(boxes_pos[j][1] == 6):
                    self.set_box4_state.pose.position.x = 1.2
                    self.set_box4_state.pose.position.y = 1.2
                elif(boxes_pos[j][1] == 7):
                    self.set_box4_state.pose.position.x = 1.8
                    self.set_box4_state.pose.position.y = 1.2
                elif(boxes_pos[j][1] == 8):
                    self.set_box4_state.pose.position.x = 2.4
                    self.set_box4_state.pose.position.y = 1.2

            if j==2:
                print(f"boxes3:{boxes_pos[j][0]}, {boxes_pos[j][1]}")
                if(boxes_pos[j][0] == 0):
                    self.set_box5_state.pose.position.x = -2.4
                    self.set_box5_state.pose.position.y = 0.6
                elif(boxes_pos[j][0] == 1):
                    self.set_box5_state.pose.position.x = -1.8
                    self.set_box5_state.pose.position.y = 0.6
                elif(boxes_pos[j][0] == 2):
                    self.set_box5_state.pose.position.x = -1.2
                    self.set_box5_state.pose.position.y = 0.6
                elif(boxes_pos[j][0] == 3):
                    self.set_box5_state.pose.position.x = -2.4
                    self.set_box5_state.pose.position.y = 0.0
                elif(boxes_pos[j][0] == 4):
                    self.set_box5_state.pose.position.x = -1.8
                    self.set_box5_state.pose.position.y = 0.0
                elif(boxes_pos[j][0] == 5):
                    self.set_box5_state.pose.position.x = -1.2
                    self.set_box5_state.pose.position.y = 0.0
                elif(boxes_pos[j][0] == 6):
                    self.set_box5_state.pose.position.x = -2.4
                    self.set_box5_state.pose.position.y = -0.6
                elif(boxes_pos[j][0] == 7):
                    self.set_box5_state.pose.position.x = -1.8
                    self.set_box5_state.pose.position.y = -0.6
                elif(boxes_pos[j][0] == 8):
                    self.set_box5_state.pose.position.x = -1.2
                    self.set_box5_state.pose.position.y = -0.6

                if(boxes_pos[j][1] == 0):
                    self.set_box6_state.pose.position.x = -2.4
                    self.set_box6_state.pose.position.y = 0.6
                elif(boxes_pos[j][1] == 1):
                    self.set_box6_state.pose.position.x = -1.8
                    self.set_box6_state.pose.position.y = 0.6
                elif(boxes_pos[j][1] == 2):
                    self.set_box6_state.pose.position.x = -1.2
                    self.set_box6_state.pose.position.y = 0.6
                elif(boxes_pos[j][1] == 3):
                    self.set_box6_state.pose.position.x = -2.4
                    self.set_box6_state.pose.position.y = 0.0
                elif(boxes_pos[j][1] == 4):
                    self.set_box6_state.pose.position.x = -1.8
                    self.set_box6_state.pose.position.y = 0.0
                elif(boxes_pos[j][1] == 5):
                    self.set_box6_state.pose.position.x = -1.2
                    self.set_box6_state.pose.position.y = 0.0
                elif(boxes_pos[j][1] == 6):
                    self.set_box6_state.pose.position.x = -2.4
                    self.set_box6_state.pose.position.y = -0.6
                elif(boxes_pos[j][1] == 7):
                    self.set_box6_state.pose.position.x = -1.8
                    self.set_box6_state.pose.position.y = -0.6
                elif(boxes_pos[j][1] == 8):
                    self.set_box6_state.pose.position.x = -1.2
                    self.set_box6_state.pose.position.y = -0.6

            if j==3:
                print(f"boxes4:{boxes_pos[j][0]}, {boxes_pos[j][1]}")
                if(boxes_pos[j][0] == 0):
                    self.set_box7_state.pose.position.x = -0.6
                    self.set_box7_state.pose.position.y = 0.6
                elif(boxes_pos[j][0] == 1):
                    self.set_box7_state.pose.position.x = 0.0
                    self.set_box7_state.pose.position.y = 0.6
                elif(boxes_pos[j][0] == 2):
                    self.set_box7_state.pose.position.x = 0.6
                    self.set_box7_state.pose.position.y = 0.6
                elif(boxes_pos[j][0] == 3):
                    self.set_box7_state.pose.position.x = -0.6
                    self.set_box7_state.pose.position.y = 0.0
                elif(boxes_pos[j][0] == 4):
                    self.set_box7_state.pose.position.x = 0.0
                    self.set_box7_state.pose.position.y = 0.0
                elif(boxes_pos[j][0] == 5):
                    self.set_box7_state.pose.position.x = 0.6
                    self.set_box7_state.pose.position.y = 0.0
                elif(boxes_pos[j][0] == 6):
                    self.set_box7_state.pose.position.x = -0.6
                    self.set_box7_state.pose.position.y = -0.6
                elif(boxes_pos[j][0] == 7):
                    self.set_box7_state.pose.position.x = 0.0
                    self.set_box7_state.pose.position.y = -0.6
                elif(boxes_pos[j][0] == 8):
                    self.set_box7_state.pose.position.x = 0.6
                    self.set_box7_state.pose.position.y = -0.6

                if(boxes_pos[j][1] == 0):
                    self.set_box8_state.pose.position.x = -0.6
                    self.set_box8_state.pose.position.y = 0.6
                elif(boxes_pos[j][1] == 1):
                    self.set_box8_state.pose.position.x = 0.0
                    self.set_box8_state.pose.position.y = 0.6
                elif(boxes_pos[j][1] == 2):
                    self.set_box8_state.pose.position.x = 0.6
                    self.set_box8_state.pose.position.y = 0.6
                elif(boxes_pos[j][1] == 3):
                    self.set_box8_state.pose.position.x = -0.6
                    self.set_box8_state.pose.position.y = 0.0
                elif(boxes_pos[j][1] == 4):
                    self.set_box8_state.pose.position.x = 0.0
                    self.set_box8_state.pose.position.y = 0.0
                elif(boxes_pos[j][1] == 5):
                    self.set_box8_state.pose.position.x = 0.6
                    self.set_box8_state.pose.position.y = 0.0
                elif(boxes_pos[j][1] == 6):
                    self.set_box8_state.pose.position.x = -0.6
                    self.set_box8_state.pose.position.y = -0.6
                elif(boxes_pos[j][1] == 7):
                    self.set_box8_state.pose.position.x = 0.0
                    self.set_box8_state.pose.position.y = -0.6
                elif(boxes_pos[j][1] == 8):
                    self.set_box8_state.pose.position.x = 0.6
                    self.set_box8_state.pose.position.y = -0.6

            if j==4:
                print(f"boxes5:{boxes_pos[j][0]}, {boxes_pos[j][1]}")
                if(boxes_pos[j][0] == 0):
                    self.set_box9_state.pose.position.x = 2.4
                    self.set_box9_state.pose.position.y = 0.6
                elif(boxes_pos[j][0] == 1):
                    self.set_box9_state.pose.position.x = 1.8
                    self.set_box9_state.pose.position.y = 0.6
                elif(boxes_pos[j][0] == 2):
                    self.set_box9_state.pose.position.x = 1.2
                    self.set_box9_state.pose.position.y = 0.6
                elif(boxes_pos[j][0] == 3):
                    self.set_box9_state.pose.position.x = 2.4
                    self.set_box9_state.pose.position.y = 0.0
                elif(boxes_pos[j][0] == 4):
                    self.set_box9_state.pose.position.x = 1.8
                    self.set_box9_state.pose.position.y = 0.0
                elif(boxes_pos[j][0] == 5):
                    self.set_box9_state.pose.position.x = 1.2
                    self.set_box9_state.pose.position.y = 0.0
                elif(boxes_pos[j][0] == 6):
                    self.set_box9_state.pose.position.x = 2.4
                    self.set_box9_state.pose.position.y = -0.6
                elif(boxes_pos[j][0] == 7):
                    self.set_box9_state.pose.position.x = 1.8
                    self.set_box9_state.pose.position.y = -0.6
                elif(boxes_pos[j][0] == 8):
                    self.set_box9_state.pose.position.x = 1.2
                    self.set_box9_state.pose.position.y = -0.6

                if(boxes_pos[j][1] == 0):
                    self.set_box10_state.pose.position.x = 2.4
                    self.set_box10_state.pose.position.y = 0.6
                elif(boxes_pos[j][1] == 1):
                    self.set_box10_state.pose.position.x = 1.8
                    self.set_box10_state.pose.position.y = 0.6
                elif(boxes_pos[j][1] == 2):
                    self.set_box10_state.pose.position.x = 1.2
                    self.set_box10_state.pose.position.y = 0.6
                elif(boxes_pos[j][1] == 3):
                    self.set_box10_state.pose.position.x = 2.4
                    self.set_box10_state.pose.position.y = 0.0
                elif(boxes_pos[j][1] == 4):
                    self.set_box10_state.pose.position.x = 1.8
                    self.set_box10_state.pose.position.y = 0.0
                elif(boxes_pos[j][1] == 5):
                    self.set_box10_state.pose.position.x = 1.2
                    self.set_box10_state.pose.position.y = 0.0
                elif(boxes_pos[j][1] == 6):
                    self.set_box10_state.pose.position.x = 2.4
                    self.set_box10_state.pose.position.y = -0.6
                elif(boxes_pos[j][1] == 7):
                    self.set_box10_state.pose.position.x = 1.8
                    self.set_box10_state.pose.position.y = -0.6
                elif(boxes_pos[j][1] == 8):
                    self.set_box10_state.pose.position.x = 1.2
                    self.set_box10_state.pose.position.y = -0.6

            if j==5:
                print(f"boxes6:{boxes_pos[j][0]}, {boxes_pos[j][1]}")
                if(boxes_pos[j][0] == 0):
                    self.set_box11_state.pose.position.x = -2.4
                    self.set_box11_state.pose.position.y = -1.2
                elif(boxes_pos[j][0] == 1):
                    self.set_box11_state.pose.position.x = -1.8
                    self.set_box11_state.pose.position.y = -1.2
                elif(boxes_pos[j][0] == 2):
                    self.set_box11_state.pose.position.x = -1.2
                    self.set_box11_state.pose.position.y = -1.2
                elif(boxes_pos[j][0] == 3):
                    self.set_box11_state.pose.position.x = -2.4
                    self.set_box11_state.pose.position.y = -1.8
                elif(boxes_pos[j][0] == 4):
                    self.set_box11_state.pose.position.x = -1.8
                    self.set_box11_state.pose.position.y = -1.8
                elif(boxes_pos[j][0] == 5):
                    self.set_box11_state.pose.position.x = -1.2
                    self.set_box11_state.pose.position.y = -1.8
                elif(boxes_pos[j][0] == 6):
                    self.set_box11_state.pose.position.x = -2.4
                    self.set_box11_state.pose.position.y = -2.4
                elif(boxes_pos[j][0] == 7):
                    self.set_box11_state.pose.position.x = -1.8
                    self.set_box11_state.pose.position.y = -2.4
                elif(boxes_pos[j][0] == 8):
                    self.set_box11_state.pose.position.x = -1.2
                    self.set_box11_state.pose.position.y = -2.4

                if(boxes_pos[j][1] == 0):
                    self.set_box12_state.pose.position.x = -2.4
                    self.set_box12_state.pose.position.y = -1.2
                elif(boxes_pos[j][1] == 1):
                    self.set_box12_state.pose.position.x = -1.8
                    self.set_box12_state.pose.position.y = -1.2
                elif(boxes_pos[j][1] == 2):
                    self.set_box12_state.pose.position.x = -1.2
                    self.set_box12_state.pose.position.y = -1.2
                elif(boxes_pos[j][1] == 3):
                    self.set_box12_state.pose.position.x = -2.4
                    self.set_box12_state.pose.position.y = -1.8
                elif(boxes_pos[j][1] == 4):
                    self.set_box12_state.pose.position.x = -1.8
                    self.set_box12_state.pose.position.y = -1.8
                elif(boxes_pos[j][1] == 5):
                    self.set_box12_state.pose.position.x = -1.2
                    self.set_box12_state.pose.position.y = -1.8
                elif(boxes_pos[j][1] == 6):
                    self.set_box12_state.pose.position.x = -2.4
                    self.set_box12_state.pose.position.y = -2.4
                elif(boxes_pos[j][1] == 7):
                    self.set_box12_state.pose.position.x = -1.8
                    self.set_box12_state.pose.position.y = -2.4
                elif(boxes_pos[j][0] == 8):
                    self.set_box12_state.pose.position.x = -1.2
                    self.set_box12_state.pose.position.y = -2.4

            if j==6:
                print(f"boxes7:{boxes_pos[j][0]}, {boxes_pos[j][1]}")
                if(boxes_pos[j][0] == 0):
                    self.set_box13_state.pose.position.x = -0.6
                    self.set_box13_state.pose.position.y = -1.2
                elif(boxes_pos[j][0] == 1):
                    self.set_box13_state.pose.position.x = 0.0
                    self.set_box13_state.pose.position.y = -1.2
                elif(boxes_pos[j][0] == 2):
                    self.set_box13_state.pose.position.x = 0.6
                    self.set_box13_state.pose.position.y = -1.2
                elif(boxes_pos[j][0] == 3):
                    self.set_box13_state.pose.position.x = -0.6
                    self.set_box13_state.pose.position.y = -1.8
                elif(boxes_pos[j][0] == 4):
                    self.set_box13_state.pose.position.x = 0.0
                    self.set_box13_state.pose.position.y = -1.8
                elif(boxes_pos[j][0] == 5):
                    self.set_box13_state.pose.position.x = 0.6
                    self.set_box13_state.pose.position.y = -1.8
                elif(boxes_pos[j][0] == 6):
                    self.set_box13_state.pose.position.x = -0.6
                    self.set_box13_state.pose.position.y = -2.4
                elif(boxes_pos[j][0] == 7):
                    self.set_box13_state.pose.position.x = 0.0
                    self.set_box13_state.pose.position.y = -2.4
                elif(boxes_pos[j][0] == 8):
                    self.set_box13_state.pose.position.x = 0.6
                    self.set_box13_state.pose.position.y = -2.4

                if(boxes_pos[j][1] == 0):
                    self.set_box14_state.pose.position.x = -0.6
                    self.set_box14_state.pose.position.y = -1.2
                elif(boxes_pos[j][1] == 1):
                    self.set_box14_state.pose.position.x = 0.0
                    self.set_box14_state.pose.position.y = -1.2
                elif(boxes_pos[j][1] == 2):
                    self.set_box14_state.pose.position.x = 0.6
                    self.set_box14_state.pose.position.y = -1.2
                elif(boxes_pos[j][1] == 3):
                    self.set_box14_state.pose.position.x = -0.6
                    self.set_box14_state.pose.position.y = -1.8
                elif(boxes_pos[j][1] == 4):
                    self.set_box14_state.pose.position.x = 0.0
                    self.set_box14_state.pose.position.y = -1.8
                elif(boxes_pos[j][1] == 5):
                    self.set_box14_state.pose.position.x = 0.6
                    self.set_box14_state.pose.position.y = -1.8
                elif(boxes_pos[j][1] == 6):
                    self.set_box14_state.pose.position.x = -0.6
                    self.set_box14_state.pose.position.y = -2.4
                elif(boxes_pos[j][0] == 7):
                    self.set_box14_state.pose.position.x = 0.0
                    self.set_box14_state.pose.position.y = -2.4
                elif(boxes_pos[j][0] == 8):
                    self.set_box14_state.pose.position.x = 0.6
                    self.set_box14_state.pose.position.y = -2.4

        # replace models
        self.box1_state._state = self.set_box1_state
        self.box2_state._state = self.set_box2_state
        self.box3_state._state = self.set_box3_state
        self.box4_state._state = self.set_box4_state
        self.box5_state._state = self.set_box5_state
        self.box6_state._state = self.set_box6_state
        self.box7_state._state = self.set_box7_state
        self.box8_state._state = self.set_box8_state
        self.box9_state._state = self.set_box9_state
        self.box10_state._state = self.set_box10_state
        self.box11_state._state = self.set_box11_state
        self.box12_state._state = self.set_box12_state
        self.box13_state._state = self.set_box13_state
        self.box14_state._state = self.set_box14_state
        self.robot_1_state._state = self.set_robot_1_state
        #'''
        while not self.set_state.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('reset : service not available, waiting again...')
        try:
            self.get_logger().info('reset positions')
            self.set_state.call_async(self.box1_state)
            self.set_state.call_async(self.box2_state)
            self.set_state.call_async(self.box3_state)
            self.set_state.call_async(self.box4_state)
            self.set_state.call_async(self.box5_state)
            self.set_state.call_async(self.box6_state)
            self.set_state.call_async(self.box7_state)
            self.set_state.call_async(self.box8_state)
            self.set_state.call_async(self.box9_state)
            self.set_state.call_async(self.box10_state)
            self.set_state.call_async(self.box11_state)
            self.set_state.call_async(self.box12_state)
            self.set_state.call_async(self.box13_state)
            self.set_state.call_async(self.box14_state)
            self.set_state.call_async(self.robot_1_state)
        except:
            import traceback
            traceback.print_exc()

        #self.obs_robot1[0] = -1.8
        #self.obs_robot1[1] = 1.8

        #state = copy.copy(lidar_data)
        self.state_reset [:20] = copy.copy(lidar_data)
        self.state_reset [20] = robot_pose[0] - self.goal_x
        self.state_reset [21] = robot_pose[1] - self.goal_y
        self.state_reset[22] = 0 # time restart
        state  = torch.tensor(self.state_reset , dtype=torch.double)#.unsqueeze(dim=0)  # Add batch dimension to state
        if self.absorbing:
            state  = torch.cat([state, torch.zeros(state.size(0), 1)], dim=1)  # Add absorbing indicator (zero) to state

        #/7.465 #7.465 is the longest distance from the robot to a wall
        #gz_env.get_logger().info(f"robot1: {self.obs_robot1}")

        return state 
    

class Get_modelstate(Node):

    def __init__(self):
        super().__init__('get_modelstate')
        self.subscription = self.create_subscription(
            ModelStates,
            '/gazebo/model_states',
            self.listener_callback,
            10)
        self.subscription

    def listener_callback(self, data):
        global robot_pose

        unit_sphere_id = data.name.index('robot_1')
        robot_pose[0] = data.pose[unit_sphere_id].position.x
        robot_pose[1] = data.pose[unit_sphere_id].position.y

class Lidar_subscriber(Node):

    def __init__(self):
        super().__init__('lidar_subscriber')
        self.subscription = self.create_subscription(
            LaserScan,
            '/robot_1_front/scan',
            self.listener_callback,
            10)
        self.subscription

    def listener_callback(self, data):
        global lidar_data
        # https://docs.ros.org/en/api/sensor_msgs/html/msg/LaserScan.html
        for i in range(20):
            lidar_data[i] = data.ranges[i]
            if(lidar_data[i] > 7.465):
                lidar_data[i] = 7.465

if __name__ == '__main__':
    rclpy.init(args=None)
    
    with open(os.environ['HOME'] + '/imitation_learning_ros/src/imitation_learning/conf/train_config.yaml', 'r') as yml:
        cfg = yaml.safe_load(yml)

    gz_env = GazeboEnv(cfg['imitation']['absorbing'], load_data=True)
    #eval_env = GazeboEnv(cfg['imitation']['absorbing'])
    get_modelstate = Get_modelstate()
    lidar_subscriber = Lidar_subscriber()

    assert cfg['defaults'][1]['algorithm'] in ['AdRIL', 'BC', 'DRIL', 'GAIL', 'GMMIL', 'PWIL', 'RED', 'SAC']
    cfg['memory']['size'] = min(cfg['steps'], cfg['memory']['size']) 
    assert cfg['bc_pretraining']['iterations'] >= 0
    assert cfg['imitation']['trajectories'] >= 0
    assert cfg['imitation']['mix_expert_data'] in ['none', 'mixed_batch', 'prefill_memory']
      # Technically possible, but makes the control flow for training the discriminator more complicated
    assert cfg['imitation']['mix_expert_data'] != 'prefill_memory'
    '''
      assert cfg.imitation.discriminator.reward_function in ['AIRL', 'FAIRL', 'GAIL']
      assert cfg.imitation.grad_penalty >= 0
      assert cfg.imitation.entropy_bonus >= 0
      assert cfg.imitation.loss_function in ['BCE', 'Mixup', 'PUGAIL']
      if cfg.imitation.loss_function == 'Mixup': assert cfg.imitation.mixup_alpha > 0
      if cfg.imitation.loss_function == 'PUGAIL': assert 0 <= cfg.imitation.pos_class_prior <= 1 and cfg.imitation.nonnegative_margin >= 0
    '''
    assert cfg['logging']['interval'] >= 0

    # Load expert trajectories dataset
    state_size = 24 # this already had 23 which is weird
    action_size = 3
    max_episode_steps = 100
    with open(f'{file_prefix}fail_fast_{fail_fast}_left_{left}', 'w') as f:
        f.write(str(fail_fast) + ' ' + str(left))
    # with open(f'{file_prefix}_algo_{cfg['defaults'][1]['algorithm']}', 'w') as f:
    #     f.write(str(cfg['defaults'][1]['algorithm']))
    # Set up agent

    # Set up imitation learning components
    gz_env.get_logger().info(f"algorithm : {cfg['defaults'][1]['algorithm']}")
    
    # Metrics
    metrics = dict(train_steps=[], train_returns=[], test_steps=[], test_returns=[], test_returns_normalized=[], update_steps=[], predicted_rewards=[], alphas=[], entropies=[], Q_values=[])
    score = []  # Score used for hyperparameter optimization 

    if cfg['check_time_usage']: start_time = time.time()  # Performance tracking

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(gz_env)
    executor.add_node(get_modelstate)
    executor.add_node(lidar_subscriber)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    collect_actions = []
    eval_actions = []

    #faaiz metrics for EVAL
    success_list = []
    timeout_list = []
    avg_success_time = []
    collision_list = []
    try:
        while rclpy.ok():

            # Training
            t = 0
            # state = gz_env.reset()
            # terminal = False
            # train_return = 0
            # discriminatorOld.eval()  # Set the "discriminator" to evaluation mode (except for DRIL, which explicitly uses dropout)
            total_step = 0
            pbar = tqdm(range(1, args.num_epochs), unit_scale=1, smoothing=0)
            for i_episode in pbar:
                memory = Memory()
                num_steps = 0
                num_episodes = 0
                
                reward_batch = []
                states = []
                actions = []
                mem_actions = []
                mem_mask = []
                mem_next = []
                while num_steps < args.batch_size:
                    state = gz_env.reset()

                    reward_sum = 0
                    for t in range(max_episode_steps+1): # not going past this anyways
                        action = select_action(state) # how he know what's state
                        action = action.data[0].numpy()
                        states.append(np.array([state]))
                        actions.append(np.array([action]))
                        collect_actions.append(action)
                        next_state, reward, done, _, _ = gz_env.step(action, t, max_episode_steps)
                        reward_sum += reward

                        mask = 1
                        if done:
                            gz_env.get_logger().info("terminal")
                            metrics['train_steps'].append(total_step)
                            metrics['train_returns'].append(reward_sum)
                            mask = 0

                        mem_mask.append(mask)
                        mem_next.append(next_state)

                        if done:
                            break

                        state = next_state
                        total_step += 1
                    num_steps += (t-1)
                    num_episodes += 1

                    reward_batch.append(reward_sum)
                rewards = expert_reward(states, actions)
                for idx in range(len(states)):
                    memory.push(states[idx][0], actions[idx], mem_mask[idx], mem_next[idx], \
                                rewards[idx][0])
                
                batch = memory.sample()
                # Train agent and imitation learning component
                update_params(batch) #batch only matters for updating policynet with trpo
                if i_episode % args.save_interval == 0:
                    torch.save(dict(policy=policy_net.state_dict(), disc=discriminator.state_dict()), 
                            f'{file_prefix}agent_{i_episode}.pth')

                ### update discriminator ###
                actions = torch.from_numpy(np.concatenate(actions))
                states = torch.from_numpy(np.concatenate(states))
                
                idx = np.random.randint(0, expert_traj.shape[0], num_steps)
                
                expert_state_action = expert_traj[idx, :]
                expert_pvalue = expert_conf[idx, :]
                expert_state_action = torch.Tensor(expert_state_action).to(device)
                expert_pvalue = torch.Tensor(expert_pvalue / Z).to(device)
                state_action = torch.cat((states, actions), 1).to(device)
                fake = discriminator(state_action)
                real = discriminator(expert_state_action)

                disc_optimizer.zero_grad()
                weighted_loss = nn.BCEWithLogitsLoss(weight=expert_pvalue)
                if args.weight:
                    disc_loss = disc_criterion(fake, torch.ones(states.shape[0], 1).to(device)) + \
                                weighted_loss(real, torch.zeros(expert_state_action.size(0), 1).to(device))
                else:
                    disc_loss = disc_criterion(fake, torch.ones(states.shape[0], 1).to(device)) + \
                                disc_criterion(real, torch.zeros(expert_state_action.size(0), 1).to(device))
                disc_loss.backward()
                disc_optimizer.step()

                # Evaluate agent and plot metrics
                if i_episode % 5 == 0:
                    gz_env.get_logger().info("Evaluation of the agent")
                    test_returns, trajectories, eval_met, actions = evaluate(i_episode, cfg['evaluation']['episodes'])
                    eval_actions.append(actions)
                    success_list.append(eval_met['suc'])
                    timeout_list.append(eval_met['timo'])
                    avg_success_time.append(eval_met['ast'])
                    collision_list.append(eval_met['col'])
                    torch.save(trajectories, f'{file_prefix}eval_trajectories_{i_episode}.pth')
                    normalization_max = 10
                    normalization_min = -1
                    test_returns_normalized = (np.array(test_returns) - normalization_min) / (normalization_max - normalization_min)
                    score.append(np.mean(test_returns_normalized))
                    metrics['test_steps'].append(i_episode)
                    metrics['test_returns'].append(test_returns)
                    metrics['test_returns_normalized'].append(list(test_returns_normalized))
                if i_episode % args.log_interval == 0:
                    print('Episode {}\tAverage reward: {:.2f}\tMax reward: {:.2f}\tLoss (disc): {:.2f}'.format(i_episode, np.mean(reward_batch), max(reward_batch), disc_loss.item()))
            # torch.save(torch.cat(collect_actions), f'{file_prefix}collect_actions.pt')
            np.save(f'{file_prefix}collect_actions.npy', collect_actions)
            # torch.save(f'{file_prefix}eval_actions.pt', torch.cat(eval_actions))
            np.savez(f'{file_prefix}eval_actions.npz', *eval_actions)
            gz_env.get_logger().info(f"metrics:{metrics}")
            np.savez(f'{file_prefix}eval_metrics.npz', success_list=success_list, 
                    timeout_list=timeout_list,
                    avg_success_time=avg_success_time, 
                    collision_list=collision_list)
            if cfg['check_time_usage']:
                metrics['training_time'] = time.time() - start_time
            
            torch.save(metrics, f'{file_prefix}metrics.pth')
            np.save(f'{file_prefix}old_score.npy', score)
            score = np.mean(score)
            gz_env.get_logger().info(f"score:{score}")
            break

    except KeyboardInterrupt:
        print("KeyboardInterrupt")

    rclpy.shutdown()
    executor_thread.join()
