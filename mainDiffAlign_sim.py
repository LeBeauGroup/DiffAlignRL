import torch.nn
from stable_baselines3 import SAC
import gym
from stable_baselines3.common.monitor import Monitor
import os
from DiffAlign_env_sim import DiffAlign_env_sim
##### Using Stable-Baselines with PyTorch
from stable_baselines3.common.callbacks import EvalCallback
# import callbacks
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.utils import set_random_seed

import numpy as np
from datetime import date

today = date.today()
date_str = today.strftime("%Y-%m-%d")
set_random_seed(21)
# Actual training and testing
# Instantiate the env
name = 'sim'
step = 100
neuron = 128
tol = 3
dir = '.'
model_str = name + "_" + str(neuron) + "_tanh_auto_"+str(tol)+"px"
log_path = dir+model_str+'/'
env = DiffAlign_env_sim(step_size=step, mode='human', log_path=log_path, tol=tol)
env = Monitor(env, log_path)

# Train the agent
policy_kwargs = dict(net_arch=[neuron, neuron], activation_fn = torch.nn.Tanh)
model = SAC('MlpPolicy', env, ent_coef='auto', verbose=2,policy_kwargs=policy_kwargs,
            device ="cuda", gamma = 0.99, tensorboard_log=log_path)
model.learn(20000, tb_log_name='sac')#, eval_log_path = log_save_path_eval, callback=eval_callback)

model.save(log_path+model_str)

env.close()
print('completed '+model_str)