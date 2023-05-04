import os
import torch
import numpy as np 
from tensorboardX import SummaryWriter
from collections import namedtuple

from game.wrapper import Game
from ppo_network import Actor, Critic
import torch.nn.functional as F

from ppo_main import get_action

def play_game():
    """
    Play Flappy Bird using the trained network.
    """
    game = [Game(hyperparameters["frame_size"]) for i in range(hyperparameters["workers"])]
    # Initialize the environment and state (do nothing)
    game = game[0]
    frame, reward, done = game.step(0)
    state = torch.cat([frame for i in range(hyperparameters["no_frames_to_network"])])
    len = 0
    # Start playing
    while True:
        len+=1

        # Perform an action
        state = state.unsqueeze(0)
        if CUDA_DEVICE:
            state = state.cuda()
        _, action, _ = get_action(actor, critic, state)
        if CUDA_DEVICE:
            action = action.cuda()
        frame, reward, done = game.step(action)
        if CUDA_DEVICE:
            frame = frame.cuda()
        next_state = torch.cat([state[0][1:], frame])

        # Move on to the next state
        state = next_state

        # If we lost, exit
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
actor = Actor(hyperparameters)
critic = Critic(hyperparameters)
actor.apply(actor.init_weights)
weights_dir = "exp_test_clip_0.2/190000_actor_.pt"
actor.load_state_dict(torch.load(weights_dir))
play_game()