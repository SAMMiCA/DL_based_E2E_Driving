# File          : sacv4.py
# Creator       : Yoonha
# Last edit     : 2021 07 07
# Description   : Baseline code for Soft Actor-Critic version 2019

# ================== Required Libraries ==================

import sys
import numpy as np
import math
import gym
import gym_carla
import carla

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torchinfo import summary
import gc
import cv2
import matplotlib.pyplot as plt

import os
from datetime import date
import shutil
import time
import threading
from collections import deque

import random
import numpy as np
from collections import deque
from tqdm import tqdm
import pickle
import json
from argparse import ArgumentParser
from params_manager import ParamsManager

# ============================================================


# ================== Required dependencies from other python script ==================
from models_yna import MLPSoftQNetwork, MLPPolicyNetwork, CNNPolicyNetwork, CNNSoftQNetwork, PolicyNetwork, SoftQNetwork
from model_mixer_yna import PolicyNetwork_mixer, SoftQNetwork_mixer
from replay_buffers import BasicBuffer
from logger_utils import *

# ============================================================


# ================== Argument parser ==================
args = ArgumentParser("sacv2_agent")
args.add_argument("--env",
                  help="Name of the environment",
                  default=None,
                  type=str,
                  metavar="ENV"
                  )

args.add_argument("--seed",
                  help="Random seed",
                  default=None,
                  type=int,
                  metavar="SEED"
                  )

args.add_argument("--port",
                  help="Carla port",
                  default=None,
                  type=int,
                  metavar="PORT"
                  )

args = args.parse_args()
# ============================================================


# ================== Hyperparameters and training setups ==================\
params_manager = ParamsManager('./config.json')

if args.env is not None:
    params_manager.get_env_params()['name'] = args.env
    params_manager.export_json('./config.json', params_manager.params)

if args.seed is not None:
    params_manager.get_agent_params()['SEED'] = args.seed
    params_manager.export_json('./config.json', params_manager.params)

if args.port is not None:
    params_manager.get_env_params()['port'] = args.port
    params_manager.export_json('./config.json', params_manager.params)

ENV_NAME = params_manager.get_env_params()['town']
MAX_STEPS = params_manager.get_agent_params()['MAX_STEPS']
MEMORY_SIZE = params_manager.get_agent_params()['MEMORY_SIZE']
SEED_NUM = params_manager.get_agent_params()['SEED']
PI_LEARN_RATE = params_manager.get_agent_params()['PI_LEARN_RATE']
Q_LEARN_RATE = params_manager.get_agent_params()['Q_LEARN_RATE']
ALPHA_LEARN_RATE = params_manager.get_agent_params()['ALPHA_LEARN_RATE']
TRAIN_START = params_manager.get_agent_params()['TRAIN_START']
BATCH_SIZE = params_manager.get_agent_params()['BATCH_SIZE']
SMOOTH_UPDATE_RATE = params_manager.get_agent_params()['SMOOTH_UPDATE_RATE']
DISCOUNT_FACTOR = params_manager.get_agent_params()['DISCOUNT_FACTOR']
USE_EVAL = params_manager.get_agent_params()['USE_EVAL']
NUM_EVAL = params_manager.get_agent_params()['NUM_EVAL']
REWARD_SCALE = params_manager.get_agent_params()['REWARD_SCALE']
GPU_ID = params_manager.get_agent_params()['GPU_ID']

NUM_VEHICLES = params_manager.get_env_params()['number_of_vehicles']
NUM_WALKERS = params_manager.get_env_params()['number_of_walkers']
DISPLAY_SIZE = params_manager.get_env_params()['display_size']
MAX_PAST_STEP = params_manager.get_env_params()['max_past_step']
DT = params_manager.get_env_params()['dt']
EGO_VEHICLE = params_manager.get_env_params()['ego_vehicle_filter']
PORT = params_manager.get_env_params()['port']
TASK = params_manager.get_env_params()['task_mode']
MAX_EP_STEPS = params_manager.get_env_params()['max_time_episode']
MAX_WAYPOINT = params_manager.get_env_params()['max_waypt']
OBS_RANGE = params_manager.get_env_params()['obs_range']
LIDAR_BIN = params_manager.get_env_params()['lidar_bin']
D_BEHIND = params_manager.get_env_params()['d_behind']
OUT_LANE_THRES = params_manager.get_env_params()['out_lane_thres']
DESIRED_SPEED = params_manager.get_env_params()['desired_speed']
MAX_SPAWN = params_manager.get_env_params()['max_ego_spawn_times']
DISPLAY_ROUTE = params_manager.get_env_params()['display_route']
PIXOR_SIZE = params_manager.get_env_params()['pixor_size']
PIXOR = params_manager.get_env_params()['pixor']

torch.cuda.set_device(GPU_ID)

file_idx = SEED_NUM

torch.manual_seed(file_idx)
np.random.seed(file_idx)

today = date.today()
setup = '/sacv2_baseline_' + str(file_idx)  # sacv2_baseline_10(seednim)
folder_name = ENV_NAME + '/' + str(today) + setup  # Town02/2021-05-04/sacv2_baseline_10
WORKSPACE = './' + str('MLP_Baselines_New/') + folder_name + '/'
# DATASPACE = './' + str('MLPMiXER_Baselines_New/') + folder_name + '/z_data/'
# WORKSPACE = './' + str('CNN_Baselines_New/') + folder_name + '/'
# DATASPACE = './' + str('CNN_Baselines_New/') + folder_name + '/z_data/'
CARLAENVROOT = '/home/aswin/workspace_yna/sacv2/gym-carla/gym_carla/envs'

# FLAG ================================
SUMMARY_FLAG = False
CSTFLAG = False
STANDARDIZATION_FLAG = False

FLAG_Autopilot = True # For SL data

###네트워크 모드 선택===================
VAEMODE = False  # VAE
MLPMODE = False  # MKimage (128,128) not in use now
CNNMODE = False  # YNAimage for CNNnetwork
MLPMODE2 = False  # PYGAME sequential 2 images
MLPMODENEW = False  # MKimage (20,100) + (28,28)
MLPMODENEW_HIGH = True  # MKimgae (64, 32) / (192, 80)
MLPPYGAME = False  # PYGAME (100,100
MLPMIXERMODE = False  # MLPMIXER with 2images (20,100) + (28,28)
MLPMIXERMODE_SINGLE = False  # MLPMiXER with 1 image


class SACAgent():
    def __init__(self):

        # Selecting the device to use, wheter CUDA (GPU) if available or CPU
        self.device = torch.device("cuda:" + str(GPU_ID) if torch.cuda.is_available() else "cpu")

        # Environment parameters / you have to change ./config.json to adapt
        params = {

            # agent HYPERPARAMETER
            "ALPHA_LEARN_RATE": ALPHA_LEARN_RATE,
            "BATCH_SIZE": BATCH_SIZE,
            "DISCOUNT_FACTOR": DISCOUNT_FACTOR,
            "GPU_ID": GPU_ID,
            "MAX_STEPS": MAX_STEPS,
            "MEMORY_SIZE": MEMORY_SIZE,
            "NUM_EVAL": NUM_EVAL,
            "PI_LEARN_RATE": PI_LEARN_RATE,
            "Q_LEARN_RATE": Q_LEARN_RATE,
            "REWARD_SCALE": REWARD_SCALE,
            "SEED": SEED_NUM,
            "SMOOTH_UPDATE_RATE": SMOOTH_UPDATE_RATE,
            "TRAIN_START": TRAIN_START,
            "USE_EVAL": USE_EVAL,

            'discrete': False,  # whether to use discrete control space
            'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
            'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
            'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
            'continuous_steer_range': [-0.1, 0.1],  # continuous steering angle range

            # env
            "d_behind": D_BEHIND,  # distance behind the ego vehicle (meter)
            "desired_speed": DESIRED_SPEED,  # desired speed (m/s)
            "display_route": DISPLAY_ROUTE,  # whether to render the desired route
            "display_size": DISPLAY_SIZE,  # screen size of bird-eye render
            "dt": DT,  # time interval between two frames
            "ego_vehicle_filter": EGO_VEHICLE,  # filter for defining ego vehicle
            "lidar_bin": LIDAR_BIN,  # bin size of lidar sensor (meter)
            "max_ego_spawn_times": MAX_SPAWN,  # maximum times to spawn ego vehicle
            "max_past_step": MAX_PAST_STEP,  # the number of past steps to draw
            "max_time_episode": MAX_EP_STEPS,  # maximum timesteps per episode
            "max_waypt": MAX_WAYPOINT,  # maximum number of waypoints
            "number_of_vehicles": NUM_VEHICLES,
            "number_of_walkers": NUM_WALKERS,
            "obs_range": OBS_RANGE,  # observation range (meter)
            "out_lane_thres": OUT_LANE_THRES,  # threshold for out of lane
            "pixor": PIXOR,
            "pixor_size": PIXOR_SIZE,  # size of the pixor labels
            "port": PORT,  # connection port
            "task_mode": TASK,  # mode of the task, [random, roundabout (only for Town03)]
            "town": ENV_NAME  # which town to simulate

        }

        # Creating the Gym environments for training and evaluation
        self.env = gym.make('carla-v0', params=params)
        # self.eval_env = gym.make(ENV_NAME)

        self.action_range = [self.env.action_space.low, self.env.action_space.high]
        # ====================================================
        if VAEMODE:
            self.obs_dim = 32 * 2  # self.env.observation_space['state'].shape[0]
            self.input_state = 'camera'
        elif MLPMODE:
            self.width = 128
            self.height = 128
            self.num = 1
            self.obs_dim = self.width * self.height * self.num  ###yna
            self.input_state = 'BEV_image'

        elif MLPMODE2:
            self.width = 64
            self.height = 128
            self.num = 2
            self.obs_dim = self.width * self.height * self.num  ###yna
            self.input_state = 'BEV_image'

        elif MLPMODENEW:
            self.w1 = 20
            self.h1 = 100
            self.w2 = 28
            self.h2 = 28
            self.obs_dim = self.w1 * self.h1 + self.w2 * self.h2
            self.input_state = 'BEV_image'

        elif MLPMODENEW_HIGH:
            self.width = 32 #80
            self.height = 64 #192
            self.num = 1
            self.obs_dim = self.width * self.height * self.num
            self.input_state = 'BEV_image'

        elif MLPPYGAME:
            self.width = 100
            self.height = 100
            self.num = 1
            self.obs_dim = self.width * self.height * self.num
            self.input_state = 'birdeye'

        elif CNNMODE:
            # self.width = 208
            # self.height = 208
            # self.channel = 1
            # self.num = 1
            # self.obs_dim = (self.num, self.channel, self.height, self.width)
            # self.input_state = 'camera'
            self.width = 112
            self.height = 800
            self.channel = 1
            self.num = 1
            self.obs_dim = (self.num, self.channel, self.height, self.width)
            self.input_state = 'BEV_image'

        elif MLPMIXERMODE:

            # --------- base_model_param of MLPMixer---------
            in_channels = 1  # mono
            hidden_size = 512  # C (channel 길이)
            num_classes = 16  ###yna 이거 마지막 state 수
            patch_size = 4 * 4  ###yna area 원래는 한변으로 나타나있었음
            resolution = 100 * 20 + 28 * 28  ###yna area 원래는 한변으로 나타나있었음
            number_of_layers = 8
            token_dim = 256
            channel_dim = 2048
            # ------------------------------------

            self.w1 = 20
            self.h1 = 100
            self.w2 = 28
            self.h2 = 28

            self.obs_dim = num_classes  ###yna mlpmixer의 출력 클라스 수가 policy network 입력 수
            self.input_state = 'BEV_image'

        # ====================================================

        elif MLPMIXERMODE_SINGLE:
            # --------- base_model_param of MLPMixer---------
            self.mixer_parameter_dict = {
                'in_channels': 1,  # mono
                'hidden_size': 256,  # C (channel 길이)
                'num_classes': 16,  ###yna does not needed
                'patch_size': 16 * 16,  ###yna area 원래는 한변으로 나타나있었음
                'image_size': 112 * 800,  ###yna area 원래는 한변으로 나타나있었음
                'number_of_layers': 8,
                'token_dim': 256,
                'channel_dim': 2048
            }
            # ------------------------------------

            # SAC layer hyperparameter
            self.SAC_depth = 4
            self.hidden_size = [256, 128, 64]
            self.input_state = 'BEV_image'
            self.width = 112
            self.height = 800
            # ==================================

        self.action_dim = 1
        # self.action_dim = self.env.action_space.shape[0] ###yna, the number of actions 2
        # print(self.action_dim)

        # hyperparameters
        self.gamma = DISCOUNT_FACTOR
        self.tau = SMOOTH_UPDATE_RATE
        self.q_lr = Q_LEARN_RATE
        self.policy_lr = PI_LEARN_RATE

        self.buffer_maxlen = int(MEMORY_SIZE)
        self.batch_size = BATCH_SIZE

        # Scaling and bias factor for the actions
        print('start initializing network,,,,,,,,,,,,,,,,,,,,,,,,,,')
        self.scale = (self.action_range[1] - self.action_range[0]) / 2.0  # [3. 0.3]
        self.bias = (self.action_range[1] + self.action_range[0]) / 2.0
        self.scale_tensor = torch.FloatTensor(self.scale).to(self.device)

        # initialize networks
        if VAEMODE or MLPMIXERMODE:
            self.q_net1 = SoftQNetwork(self.obs_dim, self.action_dim).to(self.device)
            self.target_q_net1 = SoftQNetwork(self.obs_dim, self.action_dim).to(self.device)
            self.q_net2 = SoftQNetwork(self.obs_dim, self.action_dim).to(self.device)
            self.target_q_net2 = SoftQNetwork(self.obs_dim, self.action_dim).to(self.device)
            self.policy = PolicyNetwork(self.obs_dim, self.action_dim).to(self.device)

        elif CNNMODE:
            self.q_net1 = CNNSoftQNetwork(self.obs_dim, self.action_dim).to(self.device)
            self.target_q_net1 = CNNSoftQNetwork(self.obs_dim, self.action_dim).to(self.device)
            self.q_net2 = CNNSoftQNetwork(self.obs_dim, self.action_dim).to(self.device)
            self.target_q_net2 = CNNSoftQNetwork(self.obs_dim, self.action_dim).to(self.device)
            self.policy = CNNPolicyNetwork(self.obs_dim, self.action_dim).to(self.device)

        elif MLPMIXERMODE_SINGLE:
            self.q_net1 = SoftQNetwork_mixer(self.action_dim, self.SAC_depth, self.hidden_size,
                                             self.mixer_parameter_dict).to(self.device)
            self.target_q_net1 = SoftQNetwork_mixer(self.action_dim, self.SAC_depth, self.hidden_size,
                                                    self.mixer_parameter_dict).to(self.device)
            self.q_net2 = SoftQNetwork_mixer(self.action_dim, self.SAC_depth, self.hidden_size,
                                             self.mixer_parameter_dict).to(self.device)
            self.target_q_net2 = SoftQNetwork_mixer(self.action_dim, self.SAC_depth, self.hidden_size,
                                                    self.mixer_parameter_dict).to(self.device)
            self.policy = PolicyNetwork_mixer(self.action_dim, self.SAC_depth, self.hidden_size,
                                              self.mixer_parameter_dict).to(self.device)

        else:  # MLP
            self.q_net1 = MLPSoftQNetwork(self.obs_dim, self.action_dim).to(self.device)
            self.target_q_net1 = MLPSoftQNetwork(self.obs_dim, self.action_dim).to(self.device)
            self.q_net2 = MLPSoftQNetwork(self.obs_dim, self.action_dim).to(self.device)
            self.target_q_net2 = MLPSoftQNetwork(self.obs_dim, self.action_dim).to(self.device)
            self.policy = MLPPolicyNetwork(self.obs_dim, self.action_dim).to(self.device)

        # copy weight parameters to the target Q networks
        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(param)

        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(param)

        # initialize optimizers
        self.q1_optimizer = optim.Adam(self.q_net1.parameters(), lr=self.q_lr)
        self.q2_optimizer = optim.Adam(self.q_net2.parameters(), lr=self.q_lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.policy_lr)

        self.replay_buffer = BasicBuffer(self.buffer_maxlen)

        # entropy temperature
        self.alpha = 0.05
        self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=ALPHA_LEARN_RATE)
        self.alpha = self.log_alpha.exp()
        self.pid = os.getpid()
        print('end initializing,,,,,,,,,,,,,,,,,,,,,,,,,,')

    def torchsummary(self, model, input_shape):

        summary(model, input_size=(input_shape.shape))

    def torchsummary_multi(self, model, input_shape1, input_shape2):
        # print('before:',input_shape1.shape, input_shape2.shape)
        # summary(model, [(32,22400),(32,1)])
        summary(model, [(input_shape1.shape), (input_shape2.shape)])

    # This is for resizing input
    def get_state(self, obs, step):
        ###yna 이건 VAE 이용해서 256256이미지를 get state에서 리사이즈 하는 것
        if VAEMODE:
            obs = cv2.resize(obs, dsize=(208, 208))
            obs = obs.reshape(1, 208, 208)

            # ============car========================
            img_label1 = np.where((obs != 10), 0, obs)
            mask_label1 = np.where((img_label1 == 10), 1, img_label1)
            test_1 = mask_label1
            mask_label1 = np.expand_dims(mask_label1, axis=0)
            mask_label1 = torch.FloatTensor(mask_label1).to(self.device)
            _, _, encoded_state1, z1 = self.VAE[0].forward_encode(mask_label1)

            # ============road========================
            img_label2 = np.where((obs != 7) & (obs != 6), 0, obs)
            mask_label2 = np.where((img_label2 == 7) | (img_label2 == 6), 1, img_label2)
            test_2 = mask_label2
            mask_label2 = np.expand_dims(mask_label2, axis=0)
            mask_label2 = torch.FloatTensor(mask_label2).to(self.device)
            _, _, encoded_state2, z2 = self.VAE[1].forward_encode(mask_label2)

            # #============person========================
            # img_label3 = np.where( (obs!= 4) , 0, obs)
            # mask_label3 = np.where( (img_label3 == 4) , 0, img_label3)
            # test_3 = mask_label3
            # mask_label3 = np.expand_dims(mask_label3, axis=0)
            # mask_label3 = torch.FloatTensor(mask_label3).to(self.device)
            # _, _, encoded_state3, z3 = self.VAE[2].forward_encode(mask_label3)

            # #============lane========================
            img_label4 = np.where((obs != 6), 0, obs)
            mask_label4 = np.where((img_label4 == 6), 1, img_label4)
            test_4 = mask_label4
            mask_label4 = np.expand_dims(mask_label4, axis=0)
            mask_label4 = torch.FloatTensor(mask_label4).to(self.device)
            _, _, encoded_state4, z4 = self.VAE[3].forward_encode(mask_label4)

            encoded_state = torch.cat([encoded_state1, encoded_state2], axis=1)

            z1 = z1.squeeze(0).cpu().detach().numpy() * 255
            z2 = z2.squeeze(0).cpu().detach().numpy() * 255
            # z3 = z3.squeeze(0).cpu().detach().numpy()*255
            z4 = z4.squeeze(0).cpu().detach().numpy() * 255

            # print(mask_label3.squeeze(0).cpu().detach().numpy())
            if step % 2000 == 0:
                cv2.imwrite('{}1/{}_z1.png'.format(DATASPACE, step), z1.reshape(208, 208, 1))
                cv2.imwrite('{}2/{}_z2.png'.format(DATASPACE, step), z2.reshape(208, 208, 1))
                # cv2.imwrite('{}3/{}_z3.png'.format(DATASPACE, step), z3.reshape(208, 208, 1))
                cv2.imwrite('{}4/{}_z4.png'.format(DATASPACE, step), z4.reshape(208, 208, 1))
                cv2.imwrite('{}1/{}_t1.png'.format(DATASPACE, step), (test_1.reshape(208, 208, 1)) * 255)
                cv2.imwrite('{}2/{}_t2.png'.format(DATASPACE, step), (test_2.reshape(208, 208, 1)) * 255)
                # cv2.imwrite('{}3/{}_t4.png'.format(DATASPACE, step), (test_3.reshape(208, 208, 1))*255)
                cv2.imwrite('{}4/{}_t4.png'.format(DATASPACE, step), (test_4.reshape(208, 208, 1)) * 255)

            # This is the output, a latent vector
            state = encoded_state[0].cpu().detach().numpy()

        ###yna render에서 256256 pyhame 이미지 CNN network
        if MLPMODE:

            obs = obs.reshape(256, 256, 3)
            r = obs[:, :, 2]
            g = obs[:, :, 1]
            b = obs[:, :, 0]

            obs = 0.2989 * r + 0.5870 * g + 0.1140 * b  # (256,256)
            obs = np.array(obs).reshape(256, 256, 1)

            # cv2.imshow('gray birdeye', obs)

            # print(obs.shape)
            obs_nm = obs / 255  # obs (1,256,256)
            obs_resize = cv2.resize(obs_nm, dsize=(self.height, self.width * self.num))
            cv2.imwrite('./eval_state.png', obs_resize)

            state = obs_resize

        ###yna MLPMODE2 MLPPYGAME -- Use pygame image
        ###yna Flatten for MLPPOLICY
        elif MLPMODE2 or MLPPYGAME:

            obs = obs.reshape(256, 256, 3)
            r = obs[:, :, 2]
            g = obs[:, :, 1]
            b = obs[:, :, 0]

            obs = 0.2989 * r + 0.5870 * g + 0.1140 * b  # (256,256)
            obs = np.array(obs).reshape(256, 256, 1)

            # cv2.imshow('gray birdeye', obs)

            # print(obs.shape)
            obs_nm = obs / 255  # obs (1,256,256)
            obs_resize = cv2.resize(obs_nm, dsize=(self.width, self.height))
            # cv2.imwrite('./eval_state.png',obs_resize*255)

            cv2.imshow('SAC input state before flatten', obs_resize)
            cv2.waitKey(1)

            obs_resize = np.ravel(obs_resize, order='C')  # flatten

            # This is the output
            state = obs_resize

        ##yna MKimage(20, 100 +28,28) flatten
        elif MLPMODENEW:
            obs_temp1 = obs[0:self.h1 * self.w1]
            obs_1 = obs_temp1.reshape(self.h1, self.w1)
            obs_temp2 = obs[self.h1 * self.w1:]
            obs_2 = obs_temp2.reshape(self.h2, self.w2)

            # cv2.imshow("SAC BIG image state before flatten", obs_1)
            # cv2.imshow("SAC SMALL image state before flatten", obs_2)
            # cv2.waitKey(1)

            state = obs


        ##yna MKimage_high(80,198) flatten
        elif MLPMODENEW_HIGH:
            obs_flat = np.ravel(obs, order='C')  # flatten
            # cv2.imshow('SAC input state before flatten', obs)
            # cv2.waitKey(1)

            state = obs_flat

        elif CNNMODE:
            # print('obs seg shape',obs.shape)
            # # #============lane========================
            # img_label = np.where( (obs != 6) , 0, obs)
            # lane_label = np.where( (img_label == 6) , 1, img_label)
            # lane = lane_label

            # obs_lane = lane.reshape(256,256,1)
            # obs_lane_nm = obs/255 #obs (1,1,256)

            # obs_resize_cv = cv2.resize(obs_lane_nm, (208,208))
            # obs_resize = obs_resize_cv.reshape(1,208,208)

            # cv2.imshow('obs from pygame', obs)
            # cv2.imshow('obs lane label before CNN', lane)
            # cv2.imshow('SAC resized input state before CNN', obs_resize_cv)
            # cv2.waitKey(1)

            # state = obs_resize
            state = obs.reshape(1, self.height, self.width)

        elif MLPMIXERMODE:
            # 100*20 + 28*28 --> 2784
            # obs shape (2784,)
            # obs_temp1 = obs[0:self.h1*self.w1]
            # obs_1 =obs_temp1.reshape(self.h1,self.w1)
            # obs_temp2 = obs[self.h1*self.w1:]
            # obs_2 = obs_temp2.reshape(self.h2,self.w2)
            # cv2.imshow("SAC BIG image state before flatten", obs_1)
            # cv2.imshow("SAC SMALL image state before flatten", obs_2)
            # cv2.waitKey(1)

            obs = obs.reshape(1, 1, self.height, self.width)
            obs = torch.FloatTensor(obs).to(self.device)

            # obs shape([1,1,2784])
            obs_mixer = self.model_mixer(obs)
            state = obs_mixer.squeeze(0).cpu().detach().numpy()

        elif MLPMIXERMODE_SINGLE:
            obs = obs.reshape(1, self.height, self.width)  # (1, 800, 112)
            state = obs

        if CSTFLAG:
            print('state shape in get action', state.shape)

        return state

    def get_action(self, state, stochastic):
        # state: the state input to the pi network
        # stochastic: boolean (True -> use noisy action, False -> use noiseless (deterministic action))
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # make batch=1
        # print('get_action_input_state shape {}'.format(state.shape)) #[1,1,208,208]
        # Get mean and sigma from the policy network
        mean, log_std = self.policy.forward(state)
        std = log_std.exp()

        if SUMMARY_FLAG:
            print('=============get_action model information==================== ')
            self.torchsummary(self.policy, state)

        # Stochastic mode is used for training, non-stochastic mode is used for evaluation
        if stochastic:
            normal = Normal(mean, std)
            z = normal.sample()
            action = torch.tanh(z)
            action = action.cpu().detach().numpy()
        else:

            # for test
            normal = mean
            normal = torch.normal(mean=mean, std=torch.FloatTensor([0.0]).to(self.device))
            action = torch.tanh(normal)

            action = action.cpu().detach().squeeze(0).numpy()

        # print(action, self.rescale_action(action))

        return self.rescale_action(action), mean, std

    def rescale_action(self, action):
        # print(action)
        return action * self.scale[1] + self.bias[1]

    def rescale_action_tensor(self, action):
        # print(action)

        return action * self.scale_tensor[1]

    def update(self, batch_size):

        # release CUDA memory
        gc.collect()
        torch.cuda.empty_cache()

        # print('start updating')
        # Sampling experiences from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Convert numpy arrays of experience tuples into pytorch tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(
            self.device)  # action_dim=1 : [batch_size, 1] / action__dim = self.env.action_space.shape[0] : [batch_size, 2]
        rewards = REWARD_SCALE * torch.FloatTensor(rewards).to(self.device)  # REWARD_SCALE = 1
        rewards = rewards.reshape(batch_size, 1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        dones = dones.view(dones.size(0), -1)
        # print('actions.size form replay buffer:0 ', actions.size(), 'actions.shape: ',actions.shape)

        # ================== Critic update (computing the loss) ==================
        # Please refer to equation (6) in the paper for details

        # Sample actions for the next states (s_t+1) using the current policy
        next_actions, next_log_pi, _, _ = self.policy.sample(next_states,
                                                             self.scale_tensor)  ######1 ###yna self.scale_tensor(max of acc 3.0, max of steer 0.3)
        if SUMMARY_FLAG:
            print('/////////// self.policy.sample(in update function) model information//////////////// ')
            self.torchsummary(self.policy, next_states)

        next_actions = self.rescale_action_tensor(
            next_actions)  ###yna since the output of the policy network is [-1,1], we need to rescale

        # Compute Q(s_t+1,a_t+1) and choose the minimum from 2 target Q networks
        next_q1 = self.target_q_net1(next_states, next_actions)  ######2
        next_q2 = self.target_q_net2(next_states, next_actions)  ######3
        min_q = torch.min(next_q1, next_q2)
        if SUMMARY_FLAG:
            print('//////////// self.q_net.forward(in update function) model information//////////////// ')
            self.torchsummary_multi(self.target_q_net1, next_states, next_actions)

        # Compute the next Q_target (Q(s_t,a_t)-alpha(next_log_pi))
        next_q_target = (min_q - self.alpha * next_log_pi)

        # Compute the Q(s_t,a_t) using s_t and a_t from the replay buffer
        curr_q1 = self.q_net1.forward(states, actions)  ######4
        action_test = torch.FloatTensor([[1, -0.3], [1, 0], [1, 0.3]]).to(self.device)
        states_test = torch.cat(
            [torch.unsqueeze(states[0], 0), torch.unsqueeze(states[0], 0), torch.unsqueeze(states[0], 0)])
        # print(action_test.size(), states_test.size())
        # print(self.q_net1.forward(states_test, action_test))
        curr_q2 = self.q_net2.forward(states, actions)  ######5

        # Find expected Q, i.e., r(t) + gamma*next_q_target
        expected_q = rewards + (1 - dones) * self.gamma * next_q_target
        # Compute loss between Q network and expected Q
        q1_loss = F.mse_loss(curr_q1, expected_q.detach())
        q2_loss = F.mse_loss(curr_q2, expected_q.detach())

        # Backpropagate the losses and update Q network parameters
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        # ============================================================

        # ================== Policy update (computing the loss) ==================
        # Sample new actions for the current states (s_t) using the current policy
        new_actions, log_pi, _, _ = self.policy.sample(states, self.scale_tensor)  ######6
        new_actions = self.rescale_action_tensor(new_actions)

        # Compute Q(s_t,a_t) and choose the minimum from 2 Q networks
        new_q1 = self.q_net1.forward(states, new_actions)  ######7
        new_q2 = self.q_net2.forward(states, new_actions)  ######8
        min_q = torch.min(new_q1, new_q2)

        # Compute the next policy loss, i.e., alpha*log_pi - Q(s_t,a_t) eq. (7)
        policy_loss = (self.alpha * log_pi - min_q).mean()

        # Backpropagate the losses and update policy network parameters
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        # ============================================================

        # Updating target networks with soft update using update rate tau
        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        # update temperature
        alpha_loss = (self.log_alpha * (-log_pi - self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

    def save_models(self, steps):
        if self.steps % steps == 0:
            # with open(WORKSPACE+'buffer/buffer_'+str(self.steps)+'.pkl','wb') as output:
            #     buffer = self.replay_buffer
            #     pickle.dump(buffer,output,pickle.HIGHEST_PROTOCOL)
            torch.save(self.policy.state_dict(), WORKSPACE + 'models/policy_' + str(self.steps) + '.pth',
                       _use_new_zipfile_serialization=False)
            # torch.save(self.q_net1.state_dict(),WORKSPACE+'models/q1_'+str(self.steps)+'.pth',_use_new_zipfile_serialization=False)
            # torch.save(self.q_net2.state_dict(),WORKSPACE+'models/q2_'+str(self.steps)+'.pth',_use_new_zipfile_serialization=False)

    def run_eval(self, episode):

        print("\nExecuting evaluation ......")
        eval_reward_store = np.zeros(NUM_EVAL)
        for i in range(NUM_EVAL):

            eval_state = self.env.reset()

            eval_episode_reward = 0
            eval_done = False
            eval_state_camera = self.get_state(eval_state[self.input_state], count)
            eval_state_car = eval_state['state']
            count = 0
            current_steer_angle = 0
            while not eval_done:
                # Use False (deterministic) action for evaluation
                count += 1
                eval_current_state = np.concatenate([eval_state_camera, eval_state_car], axis=0)
                # eval_action,_,_ = self.get_action(eval_current_state, False)
                eval_action, _, _ = self.get_action(eval_state_camera, False)
                if abs(current_steer_angle - eval_action[0]) > 0.03:
                    if current_steer_angle - eval_action[0] <= 0:
                        current_steer_angle += 0.03
                    else:
                        current_steer_angle -= 0.03
                else:
                    # current_steer_angle = eval_action[0]
                    current_steer_angle = eval_action[0][0]  ###yna

                action_ = np.array([1.3])
                # action[0]=1.3
                action_ = np.concatenate([action_, [current_steer_angle]])
                eval_next_state, eval_reward, eval_done = self.env.step(action_)
                eval_reward += -5 * np.abs(current_steer_angle)
                eval_episode_reward += eval_reward
                eval_state_camera = self.get_state(eval_state[self.input_state], self.steps)
                eval_state_car = eval_next_state['state']

                # if count == 50:
                if count == 500:
                    eval_done = True
                if eval_done:
                    eval_reward_store[i] = eval_episode_reward
                    # write_to_file(WORKSPACE,"eval", self.steps,count
                    #     ,eval_episode_reward)

        mean_eval = np.mean(eval_reward_store)
        print(NUM_EVAL, "evaluations finished with average reward :", mean_eval, 'and stdev,',
              np.std(eval_reward_store), '\n\n')
        write_to_file(WORKSPACE, "eval_mean", self.steps, episode, mean_eval)

    def run(self):

        self.env.reset()

    def run_(self):
        self.steps = 0
        episode = 0
        eval_counter = 0

        while self.steps < MAX_STEPS + 10000:

            state = self.env.reset()
            episode_reward = 0
            done = False
            steps_count = 0

            # yna: get_state = normalize state camera, and resize
            state_camera = self.get_state(state[self.input_state], self.steps)  # resized and normalized BEV image
            # state_camera = self.get_state(state['camera'], self.steps)
            # state_car = state['state'] # This one is only the velocity of the car

            # current_state = np.concatenate([state_camera, state_car], axis=0)
            # print(state_camera)

            # yna: normalize state camera, and resize
            current_state = state_camera
            current_steer_angle = 0
            while not done:
                # Run the policy to get the action a_t for the current state s_t
                # Use True (stochastic) action for exploration
                # THis is for generating the action, so the input is the latent vector.
                action, mean, std = self.get_action(current_state, True)
                current_steer_angle = action[0][0]

                action_ = np.array([1.3])
                # if self.steps <TRAIN_START:
                # action_[0] = np.random.uniform(0.0,3.0)

                action_ = np.concatenate([action_, [current_steer_angle]])  # action_ :: longitudinal velocity

                mean = mean.cpu().detach().squeeze(0).numpy()
                std = std.cpu().detach().squeeze(0).numpy()

                # Run the action to the environment and obtain s_t+1, r_t, termination info
                # print( "mean: {} std : {}".format(mean, std))
                next_state, reward, done = self.env.step(action_, self.steps)
                # reward += -5 * np.abs(current_steer_angle)
                if CSTFLAG:
                    print("current_steer_angle: ", current_steer_angle, ' reward : ', reward)
                    print('STEP in this episode: ', steps_count)
                    print('Episode: ', episode, ' Total steps: ', self.steps)
                # print( "action: {} tan: {} std : {}".format(current_steer_angle ,np.tanh(mean), reward))
                next_state_camera = self.get_state(next_state[self.input_state], self.steps)
                next_state_car = next_state['state']
                # next_state = np.concatenate([next_state_camera, next_state_car], axis=0)
                next_state = next_state_camera

                # Update steps count and episodic reward for the current episode
                steps_count += 1
                self.steps += 1
                episode_reward += reward

                # if steps_count < 1000:
                if FLAG_Autopilot is not True:
                    self.replay_buffer.push(current_state, action[0], reward, next_state, done, mean, std)
                    self.save_models(5000)
                    # if len(self.replay_buffer) > TRAIN_START:
                    if len(self.replay_buffer) > 50:  # Now, batch size ==128
                        # print('start updating!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                        self.update(BATCH_SIZE)
                    # if done:
                    #     for _ in range(steps_count):
                    #         self.update(BATCH_SIZE)
                # Do network parameter updates if the number of experiences in the buffer is > TRAIN_START

                # if (self.steps % 10000== 0):
                # # if (self.steps % 100== 0):
                #     self.run_eval(episode)
                #     done = True

                # Print the episodic reward to console if the agent dies
                if done:
                    # if len(self.replay_buffer) > TRAIN_START:
                    #     for i in range(500):
                    #         self.update(BATCH_SIZE)
                    print("------------------------------------------------------------------------------")
                    print("Episode : ", episode)
                    print("Steps in this episode", steps_count)
                    print("Steps : ", self.steps, )
                    print('Epsiode reward : ', episode_reward)
                    print('==============================================================================')
                    # print("\rSteps ", self.steps, ": " + str(episode_reward), len(self.replay_buffer))
                    write_to_file(WORKSPACE, "eval", self.steps, episode, episode_reward)
                    episode += 1

                # Set the current state as the next_state of the environment before looping back
                current_state = next_state


if __name__ == "__main__":
    if not os.path.exists(WORKSPACE):
        os.makedirs(WORKSPACE, 0o777)
        print("File created at ", str(WORKSPACE))
    else:
        print("Folder exists: " + str(WORKSPACE))
        delete = input("Are you sure you want to overwrite? (y/n)")
        if delete == "y":
            shutil.rmtree(WORKSPACE)
            os.makedirs(WORKSPACE, 0o777)
            print("Overwrite succesful! -> " + str(WORKSPACE))
        else:
            setup = 'logger_temporary'
            folder_name = ENV_NAME + '/' + setup
            WORKSPACE = './' + str('BaseLines/') + folder_name + '/'
            print("File not overwritten, training saved to temporary!! " + str(WORKSPACE))
            if not os.path.exists(WORKSPACE):
                os.makedirs(WORKSPACE, 0o777)
                print("File created at ", str(WORKSPACE))
            else:
                shutil.rmtree(WORKSPACE)
                os.makedirs(WORKSPACE, 0o777)
    os.mkdir(WORKSPACE + str("models"))
    os.mkdir(WORKSPACE + str("buffer"))

    create_file(WORKSPACE, "eval")
    create_file(WORKSPACE, "eval_mean")

    shutil.copyfile('./config.json', WORKSPACE + 'config.json')
    shutil.copyfile(CARLAENVROOT + '/carla_env.py', WORKSPACE + str(today) + '_carla_env.py')
    if not VAEMODE and not CNNMODE:
        shutil.copyfile(CARLAENVROOT + '/MK_render_2images.py', WORKSPACE + 'MK_render_2images.py')
        shutil.copyfile(CARLAENVROOT + '/MK_render_1image_highresolution.py',
                        WORKSPACE + 'MK_render_1image_highresolution.py')
    shutil.copyfile('./' + __file__, WORKSPACE + __file__)

    gc.collect()
    torch.cuda.empty_cache()

    agent = SACAgent()
    # agent.run()
    agent.run_()





