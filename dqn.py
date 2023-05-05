# This code was adapted from: https://github.com/lambders/drl-experiments/blob/master/dqn.py

import math
import random
import argparse
import numpy as np 
from collections import namedtuple
import matplotlib.pyplot as plt
import cv2
import matplotlib.cm as cm
import tensorboard
import os
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['SDL_VIDEODRIVER']='dummy'

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from tensorboardX import SummaryWriter

from game.wrapper import Game


CUDA_DEVICE = torch.cuda.is_available()

class DQN(nn.Module):

    def __init__(self, options):
        super().__init__()

        self.opt = options
        
        self.layers_1 = nn.Sequential(
            nn.Conv2d(self.opt.len_agent_history, 32, 8, 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(inplace=True),
        )
        self.layers_2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.opt.n_actions)
        )
    

        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            init.kaiming_uniform_(m.weight, a=0.01)
            if m.bias is not None:
                init.constant_(m.bias, 0.01)

    def forward(self, x):
        x_input = x
        x = self.layers_1(x)
        heatmap_layer = x 

        # input
        image_rescaled = (255*x_input[0][0]).clamp(0, 255).byte()  # Rescale and convert to byte tensor
        image_rgb = image_rescaled.unsqueeze(2).repeat(1, 1, 3)  # Add a third dimension and repeat values along it
        image_rgb = image_rgb.cpu()
        cv2.imwrite('input_image.png', image_rgb.numpy())  # Save the image using cv2.imwrite()

        # Compute heatmap
        x_1_np = heatmap_layer.detach().cpu().numpy()
        heatmap = np.mean(x_1_np, axis=1)  # Average over channels
        heatmap = np.squeeze(heatmap)  # Remove the channel dimension
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))  # Normalize to [0, 1]

        # Convert the heatmap to RGBA
        cmap = cm.get_cmap('jet')
        rgba_heatmap = cmap(heatmap)

        # Convert RGBA to BGR
        bgr_heatmap = cv2.cvtColor((rgba_heatmap * 255).astype(np.uint8), cv2.COLOR_RGBA2BGR)
        
        # Upscale the heatmap
        upscale_factor = 10
        new_height, new_width = bgr_heatmap.shape[0] * upscale_factor, bgr_heatmap.shape[1] * upscale_factor
        bgr_heatmap_resized = cv2.resize(bgr_heatmap, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Save the upscaled heatmap
        cv2.imwrite('heatmap.png', bgr_heatmap_resized)
        return self.layers_2(x)

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory():


    def __init__(self, options):
        self.mem = []
        self.capacity = options.replay_mem_size



    def add(self, experience):
        self.mem.append(experience)

        if len(self.mem) > self.capacity:
            self.mem.pop(0)


    def sample(self, batch_size):
        if batch_size > len(self.mem):
            return None

        sample = random.sample(self.mem, batch_size)

        state = torch.stack([torch.tensor(exp.state) for exp in sample])
        action = torch.tensor([exp.action for exp in sample]).unsqueeze(1)
        reward = torch.tensor([exp.reward for exp in sample])
        next_state = torch.stack([torch.tensor(exp.next_state) for exp in sample])
        done = torch.tensor([exp.done for exp in sample])
    
 
        if CUDA_DEVICE:
            state = state.cuda()
            action = action.cuda()
            reward = reward.cuda()
            next_state = next_state.cuda()
            done = done.cuda()
    
        return {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
        }



class DQNAgent:

    def __init__(self, options):
        self.opt = options

        self.replay_mem = ReplayMemory(self.opt)

        self.epsilon = np.linspace(
            self.opt.initial_exploration, 
            self.opt.final_exploration, 
            self.opt.final_exploration_frame
        )

        self.net = DQN(self.opt)
        if self.opt.mode == 'train':
            self.net.apply(self.net.init_weights)
            if self.opt.weights_dir:
                self.net.load_state_dict(torch.load(self.opt.weights_dir))
        if self.opt.mode == 'eval':
            self.net.load_state_dict(torch.load(self.opt.weights_dir, map_location=torch.device('cpu')))


        if CUDA_DEVICE:
            self.net = self.net.cuda()

        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.opt.learning_rate
        )

        self.game = Game(self.opt.frame_size) 

        if self.opt.mode == 'train':
            self.writer = SummaryWriter(self.opt.exp_name)

        self.loss = torch.nn.MSELoss()


    def select_action(self, state, step):

        state = state.unsqueeze(0)
        if CUDA_DEVICE:
            state = state.cuda()

        epsilon = self.epsilon[min(step, self.opt.final_exploration_frame - 1)]
    

        if random.random() <= epsilon:
            return np.random.choice(self.opt.n_actions, p=[0.95, 0.05])
        else:
            return torch.argmax(self.net(state)).item()


    def optimize_model(self):

        batch = self.replay_mem.sample(self.opt.batch_size)
        if batch is None:
            return

        q = self.net(batch['state']).gather(1, batch['action']).squeeze()


        q_1, _ = torch.max(self.net(batch['next_state']), dim=1)
        y = torch.where(batch['done'], batch['reward'], 
                           batch['reward'] + self.opt.discount_factor * q_1)
        if CUDA_DEVICE:
            y = y.cuda()
        y = y.detach()

        loss = self.loss(q, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


    def train(self):

        eplen = 0
        frame, reward, done = self.game.step(0)
        state = torch.cat([frame for i in range(self.opt.len_agent_history)])
        for i in range(1, self.opt.n_train_iterations):
            action = self.select_action(state, i)
            frame, reward, done = self.game.step(action)
            next_state = torch.cat([state[1:], frame])
            self.replay_mem.add(
                Experience(state, action, reward, next_state, done)
            )
            loss = self.optimize_model()
            state = next_state
            if i % self.opt.save_frequency == 0:
                if not os.path.exists(self.opt.exp_name):
                    os.mkdir(self.opt.exp_name)
                torch.save(self.net.state_dict(), f'{self.opt.exp_name}/{str(i).zfill(7)}.pt')
            if i % self.opt.log_frequency == 0:
                self.writer.add_scalar('loss', loss, i)
            eplen += 1
            if done:
                self.writer.add_scalar('episode_length', eplen, i)
                eplen = 0


    def play_game(self):
        len=0

        with torch.no_grad(): 
            frame, reward, done = self.game.step(0)
            state = torch.cat([frame for i in range(self.opt.len_agent_history)])

            while True:
                len+=1
                state = state.unsqueeze(0)
                if CUDA_DEVICE:
                    state = state.cuda()
                action = torch.argmax(self.net(state)[0])
                frame, reward, done = self.game.step(action)
                if CUDA_DEVICE:
                    frame = frame.cuda()
                next_state = torch.cat([state[0][1:], frame])

                state = next_state

                if done:
                    print("len", len)
                    break