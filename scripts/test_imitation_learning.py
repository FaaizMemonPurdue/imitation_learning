#!/usr/bin/python3
fail_fast = False
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import EntityState, ModelStates
from gazebo_msgs.srv import SetEntityState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Empty
from rclpy import Parameter

import math
import threading
import numpy as np
import time
import copy
import torch
from torch import optim
from torch import Tensor
from typing import Dict, List, Tuple, Union
import h5py
import os
import yaml

from memory import ReplayMemory
from models import GAILDiscriminator, GMMILDiscriminator, PWILDiscriminator, REDDiscriminator, SoftActor, \
                   RewardRelabeller, TwinCritic, create_target_network
stamp = "010607"
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
        self.publisher_stepspd = self.create_publisher(Float64MultiArray, '/stepspd', 10)
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
        self.rate = self.create_rate(1.0 / self.TIME_DELTA)
        self.timeouts = False
        self.next_obs = np.zeros(23)
        self.state_reset = np.zeros(23)
        self.goal_x = 1.8
        self.goal_y = -1.8

    def step(self, action, step, max_episode_steps):
        global lidar_data

        #self.done = False
        #self.actions[:] = axes[:]
        #obs = copy.copy(lidar_data)
        action = action[0].to('cpu').detach().numpy().copy()
        #self.get_logger().info(f"action:{action}")

        self.wheel_vel1[0] = (action[0]*math.sin(math.pi/4            ) + action[1]*math.cos(math.pi/4            ) + self.L*action[2])/self.Rw
        self.wheel_vel1[1] = (action[0]*math.sin(math.pi/4 + math.pi/2) + action[1]*math.cos(math.pi/4 + math.pi/2) + self.L*action[2])/self.Rw
        self.wheel_vel1[2] = (action[0]*math.sin(math.pi/4 - math.pi)   + action[1]*math.cos(math.pi/4 - math.pi)   + self.L*action[2])/self.Rw
        self.wheel_vel1[3] = (action[0]*math.sin(math.pi/4 - math.pi/2) + action[1]*math.cos(math.pi/4 - math.pi/2) + self.L*action[2])/self.Rw


        #publish robot1 commands
        #gz_env.publisher_vel1.publish(gz_env.vel_msg1)
        array_forPublish1_vel = Float64MultiArray(data=self.wheel_vel1)  
        self.publisher_robot_vel1.publish(array_forPublish1_vel)
        self.publisher_stepspd.publish(array_forPublish1_vel)
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
        # reward = 1/dist

        # if(dist <= 0.35):
        #     done = True
        #     reward = 10
        # elif(np.amin(self.next_obs[:20]) < 0.25):
        #     reward = -1
        #     done = False
        # else:
        #     done = False

        # # time out
        # if step >= max_episode_steps:
        #     self.get_logger().info("time out")
        #     done = True
        reward = np.exp(-dist) # e^-0.35 * 100 = 70.46 from standing next to the goal
        mind = np.amin(self.next_obs[:20])
        if(dist <= 0.35):
            done = True 
            reward = 500 
            self.get_logger().info('Goal reached!')
        elif mind < 0.11: #could add collision listener but this p good
            reward = -30
            done = fail_fast
            self.get_logger().info('Collision!')
        elif mind < 0.25:
            reward = -1
            self.get_logger().info('Close to collision!')
            done = False
        else:
            done = False
        # reward -= np.exp(step / max_episode_steps) * 50  # punish for taking too long by up to 50 (at timeout)
        reward -= (step / max_episode_steps) * 50
        # exponential penalty was accumulating too much with time, so I added a linear decay to the reward
        # time out
        if step >= max_episode_steps:
            self.get_logger().info("time out")
            done = True

        next_obs_re = torch.tensor(self.next_obs, dtype=torch.float32).unsqueeze(dim=0)  # Add batch dimension to state
        if self.absorbing:
            # Add absorbing indicator (zero) to state (absorbing state rewriting done in replay memory)
            next_obs_re = torch.cat([next_obs_re, torch.zeros(next_obs_re.size(0), 1)], dim=1) 
        self.get_logger().info(f"step:{step}")
        return next_obs_re, reward, done #next_state, reward, terminal

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

        rng = np.random.default_rng()
        #self.seed += 1
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
        state  = torch.tensor(self.state_reset , dtype=torch.float32).unsqueeze(dim=0)  # Add batch dimension to state
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

# Evaluate agent with deterministic policy π
def evaluate_agent(actor: SoftActor, num_episodes: int, return_trajectories: bool=False) -> Union[Tuple[List[List[float]], Dict[str, Tensor]], List[List[float]]]:
  returns = []
  trajectories = []
  #if render:
  #    env.render()
  max_episode_steps = 100

  with torch.inference_mode():
    for i in range(num_episodes):
      print(i)
      states = []
      actions = []
      rewards = []
      state = gz_env.reset()
      terminal = False
      t = 0
      while not terminal:
          action = actor.get_greedy_action(state)  # Take greedy action
          next_state, reward, terminal = gz_env.step(action, t, max_episode_steps)
          t += 1

          if return_trajectories:
            states.append(state)
            actions.append(action)
          rewards.append(reward)
          state = next_state
      returns.append(sum(rewards))

      if return_trajectories:
        # Collect trajectory data (including terminal signal, which may be needed for offline learning)
        terminals = torch.cat([torch.zeros(len(rewards) - 1), torch.ones(1)])
        trajectories.append({'states': torch.cat(states), 'actions': torch.cat(actions), 'rewards': torch.tensor(rewards, dtype=torch.float32), 'terminals': terminals})

  return (returns, trajectories) if return_trajectories else returns

if __name__ == '__main__':
    rclpy.init(args=None)
    
    with open(os.environ['HOME'] + '/imitation_learning_ros/src/imitation_learning/conf/train_config.yaml', 'r') as yml:
        cfg = yaml.safe_load(yml)

    gz_env = GazeboEnv(cfg['imitation']['absorbing'], load_data=False)
    #eval_env = GazeboEnv(cfg['imitation']['absorbing'])
    get_modelstate = Get_modelstate()
    lidar_subscriber = Lidar_subscriber()

    action_list = []
    next_state_list = []
    state_list = []
    reward_list = []
    done_list = []
    time_out_list = []   

    assert cfg['defaults'][1]['algorithm'] in ['AdRIL', 'BC', 'DRIL', 'GAIL', 'GMMIL', 'PWIL', 'RED', 'SAC']
    cfg['memory']['size'] = min(cfg['steps'], cfg['memory']['size']) 
    assert cfg['bc_pretraining']['iterations'] >= 0
    assert cfg['imitation']['trajectories'] >= 0
    assert cfg['imitation']['mix_expert_data'] in ['none', 'mixed_batch', 'prefill_memory']
    if cfg['defaults'][1]['algorithm'] == 'AdRIL': 
      assert cfg['imitation']['mix_expert_data'] == 'mixed_batch'
    elif cfg['defaults'][1]['algorithm'] == 'DRIL': 
      assert 0 <= cfg.imitation.quantile_cutoff <= 1
    elif cfg['defaults'][1]['algorithm'] == 'GAIL':
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
    normalization_max = 10
    normalization_min = -1

    state_size = 24
    action_size = 3
    max_episode_steps = 100

    # Set up agent
    actor = SoftActor(state_size, action_size, cfg['reinforcement']['actor'])

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(gz_env)
    executor.add_node(get_modelstate)
    executor.add_node(lidar_subscriber)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    # loading discriminator model
    #discriminator_path = os.environ['HOME'] + '/imitation_learning_ros/src/imitation_learning/logs/discriminator.pth'
    #discriminator_model = torch.load(discriminator_path)
    #discriminator.load(discriminator_model)

    # loading agent model
    agent_path = os.environ['HOME'] + f'/imitation_learning_ros/src/imitation_learning/logs/{stamp}/agent.pth'
    #agentmodel = torch.load(agent_path)
    checkpoint= torch.load(agent_path)
    #print(load_weights['actor'])
    #actor.load(load_weights['actor'])
    actor.load_state_dict(checkpoint['actor'])
    actor.eval()

    try:
        while rclpy.ok():
            test_returns = evaluate_agent(actor, cfg['evaluation']['episodes'])
            break

    except KeyboardInterrupt:
        print("KeyboardInterrupt")

    rclpy.shutdown()
    executor_thread.join()

