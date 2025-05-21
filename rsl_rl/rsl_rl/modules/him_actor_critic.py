# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from .actor_critic import ActorCritic, get_activation
from rsl_rl.modules.him_estimator import HIMEstimator

class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape, device):  # shape:the dimension of input data
        self.n = 1e-4
        self.uninitialized = True
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)

    def update(self, x):
        count = self.n
        batch_count = x.size(0)
        tot_count = count + batch_count

        old_mean = self.mean.clone()
        delta = torch.mean(x, dim=0) - old_mean

        self.mean = old_mean + delta * batch_count / tot_count
        m_a = self.var * count
        m_b = x.var(dim=0) * batch_count
        M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count
        self.var = M2 / tot_count
        self.n = tot_count

class Normalization:
    def __init__(self, shape, device='cuda:0'):
        self.running_ms = RunningMeanStd(shape=shape, device=device)

    def __call__(self, x, update=False):
        # Whether to update the mean and std,during the evaluating,update=Flase
        if update:  
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (torch.sqrt(self.running_ms.var) + 1e-4)

        return x

class HIMActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_scandots,
                        num_critic_obs,
                        num_one_step_obs,
                        num_actions,
                        actor_hidden_dims=[512, 256, 128],
                        critic_hidden_dims=[512, 256, 128],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(HIMActorCritic, self).__init__()

        activation = get_activation(activation)

        self.history_size = int(num_actor_obs/num_one_step_obs)
        # self.history_size = int((num_actor_obs - 117)/num_one_step_obs)
        self.num_actor_obs = num_actor_obs
        self.num_scandots = num_scandots
        self.num_actions = num_actions
        self.num_one_step_obs = num_one_step_obs

        mlp_input_dim_a = num_one_step_obs + num_scandots + 3 + 16
        # mlp_input_dim_a = num_one_step_obs + 117 + 3 + 16
        mlp_input_dim_c = num_critic_obs

        # Estimator
        self.estimator = HIMEstimator(temporal_steps=self.history_size, num_one_step_obs=num_one_step_obs, num_scandots=num_scandots)

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
                # actor_layers.append(nn.Tanh())
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f'Estimator: {self.estimator.encoder}')

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs_history, scandots):
        # print(obs_history.shape)
        with torch.no_grad():
            print("RUNNING")
            vel, latent = self.estimator(obs_history, scandots)
        actor_input = torch.cat((obs_history[:,:self.num_one_step_obs], scandots, vel, latent), dim=-1)
        # actor_input = torch.cat((obs_history[:,:self.num_one_step_obs], 
        #                          obs_history[:,-117:], 
        #                          vel, latent), dim=-1)
        print("Actor Input shape:", actor_input.shape)
        mean = self.actor(actor_input)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, obs_history=None, scandots=None, **kwargs):
        self.update_distribution(obs_history, scandots)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_history, scandots, observations=None):
        print("RUUNING2")
        vel, latent = self.estimator(obs_history, scandots)
        actions_mean = self.actor(torch.cat((obs_history[:,:self.num_one_step_obs], scandots, vel, latent), dim=-1))
        # actions_mean = self.actor(torch.cat((obs_history[:,:self.num_one_step_obs], 
        #                          obs_history[:,-117:], 
        #                          vel, latent), dim=-1))
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
    
    def flip_obs(self, obs):
        flip_idx = torch.tensor([5,6,7,8,9, 0,1,2,3,4, 10, 15,16,17,18, 11,12,13,14], device=obs.device)

        obs_dim = self.num_one_step_obs
        history_len = obs.shape[1] // obs_dim

        for h in range(history_len):
            start = h * obs_dim

            # Commands: x,y,yaw
            obs[:, start+1] *= -1
            obs[:, start+2] *= -1

            # Base ang vel: x,y,z
            obs[:, start+3] *= -1
            obs[:, start+5] *= -1

            # DOF pos
            obs[:, start+9:start+28] = obs[:, start+9:start+28][:, flip_idx]
            # DOF vel
            obs[:, start+28:start+47] = obs[:, start+28:start+47][:, flip_idx]
            # Actions
            obs[:, start+47:start+66] = obs[:, start+47:start+66][:, flip_idx]

        return obs
    
    def flip_act(self, mu):
        flip_idx = torch.tensor([5,6,7,8,9, 0,1,2,3,4, 10, 15,16,17,18, 11,12,13,14], device=mu.device)
        mu = mu[:, flip_idx]
        return mu
    
    def flip_critic_obs(self, critic_obs):
        """
        Flip critic observations for left-right symmetry (X-Z plane).

        Input shape: (num_envs, 66+3+3+117)
        - 66: obs (no history)
        - 3: base linear vel
        - 3: disturbance (x, y, z)
        - 117: heightmap (13x9)

        Returns: flipped critic_obs (same shape)
        """

        num_batch = critic_obs.shape[0]
    
        flip_idx = torch.tensor([5,6,7,8,9, 0,1,2,3,4, 10, 15,16,17,18, 11,12,13,14], device=critic_obs.device)

        # Commands: x, y, yaw at [0, 1, 2]
        critic_obs[:, 1] *= -1
        critic_obs[:, 2] *= -1

        # Base ang vel: x, y, z at [3, 4, 5]
        critic_obs[:, 3] *= -1
        critic_obs[:, 5] *= -1

        # DOF pos at [9:28]
        critic_obs[:, 9:28] = critic_obs[:, 9:28][:, flip_idx]

        # DOF vel at [28:47]
        critic_obs[:, 28:47] = critic_obs[:, 28:47][:, flip_idx]

        # Actions at [47:66]
        critic_obs[:, 47:66] = critic_obs[:, 47:66][:, flip_idx]

        # Base linear vel: x, y, z at [66,67,68]
        critic_obs[:, 67] *= -1

        # Disturbance at [69,70,71] (x, y, z)
        critic_obs[:, 70] *= -1

        # Heights: [72:72+117] => reshape (num_envs, 13, 9), flip along X (dim=2)
        heights = critic_obs[:, 72:72+117]
        heights = heights.view(num_batch, 13, 9)
        heights = heights[:, :, torch.arange(8, -1, -1)]
        critic_obs[:, 72:72+117] = heights.view(num_batch, -1)

        return critic_obs
    
    def flip_scandots(self, scandots):
        num_batch = scandots.shape[0]
        heights = scandots
        heights = heights.view(num_batch, 13, 9)
        heights = heights[:, :, torch.arange(8, -1, -1)]
        heights = heights.view(num_batch, -1)

        return heights