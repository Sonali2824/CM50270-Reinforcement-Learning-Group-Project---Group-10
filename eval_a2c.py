import os
import torch
import numpy as np 
from tensorboardX import SummaryWriter
from collections import namedtuple
from a2c_network import ActorCriticNetwork

from game.wrapper import Game
from network import Actor, Critic
import torch.nn.functional as F


def play_game():
    game_simulations = [Game(hyperparameters["frame_size"]) for i in range(hyperparameters["workers"])]
    game_simulations = game_simulations[0]
    frame, reward, done = game_simulations.step(0)
    current_state = torch.cat([frame for i in range(hyperparameters["no_frames_to_network"])])
    len = 0
    while True:
        len+=1
        current_state = current_state.unsqueeze(0)
        if CUDA_DEVICE:
            current_state = current_state.cuda()
        _, action, _ = actorCritic.take_action(current_state)
        if CUDA_DEVICE:
            action = action.cuda()
        frame, reward, done = game_simulations.step(action)
        if CUDA_DEVICE:
            frame = frame.cuda()
        next_state = torch.cat([current_state[0][1:], frame])

        current_state = next_state

        if done:
            print(len, "len")
            break


hyperparameters = {
    "logs_dir":"exp_test_retest_2",
    "model_dir":"",
    "no_train_iterations":200000,
    "lr":1e-4,
    "no_frames_to_network":4,
    "gamma":0.99,
    "workers":8,
    "update_freq":20,
    "entropy_coefficient":0.01,
    "value_loss_coefficient":0.5,
    "norm_clip_grad":40,
    "clip":0.1,
    "log_freq":100,
    "save_freq":10000,
    "actions_n":2,
    "frame_size":84
    }
CUDA_DEVICE = torch.cuda.is_available()
actorCritic = ActorCriticNetwork(hyperparameters)
actorCritic.apply(actorCritic.init_weights)
weights_dir = "exp_test_4/0099000.pt"
actorCritic.load_state_dict(torch.load(weights_dir))
play_game()

