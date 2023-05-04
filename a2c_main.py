# This code was adapted from: https://github.com/lambders/drl-experiments/blob/master/a2c.py
import os
import torch
import numpy as np 
from tensorboardX import SummaryWriter
from collections import namedtuple

from game.wrapper import Game
from a2c_network import ActorCriticNetwork
import torch.nn.functional as F

def optimise(buffer):

    action_tensor = torch.stack(buffer.action)
    #action_log_prob_tensor = torch.stack(buffer.action_log_prob).detach()
    reward_tensor = torch.tensor(buffer.reward)
    state_tensor = torch.stack(buffer.state)
    termination_tensor = torch.stack(buffer.termination)
    #val_tensor = torch.stack(buffer.val ).detach()

    batch_data = {
        'action': action_tensor,
        'reward': reward_tensor,
        'state': state_tensor,
        'termination': termination_tensor
    }

    state_dim = batch_data['state'].size()[2:]
    action_dim = batch_data['action'].size()[-1]

    next_value, _ = net(batch_data['state'][-1])

    returns = torch.zeros(hyperparameters["update_freq"] + 1, hyperparameters["workers"], 1)
    returns[-1] = next_value
    for i in reversed(range(hyperparameters["update_freq"])):
        returns[i] = returns[i+1] * hyperparameters["gamma"] * batch_data['termination'][i] + batch_data['reward'][i]
    returns = returns[:-1]

    vals, action_log_probabilities, distrib_entropy = net.action_evaluation(batch_data['state'].view(-1, *state_dim), batch_data['action'].view(-1, action_dim)) 
    vals = vals.view(hyperparameters["update_freq"], hyperparameters["workers"], 1)
    action_log_probabilities = action_log_probabilities.view(hyperparameters["update_freq"], hyperparameters["workers"], 1)

    advs = returns - vals
    val_loss = advs.pow(2).mean()

    action_loss = -(advs*action_log_probabilities).mean()


    total_loss = hyperparameters["value_loss_coefficient"] * val_loss + action_loss - distrib_entropy * hyperparameters["entropy_coefficient"]

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm(net.parameters(), hyperparameters["norm_clip_grad"])
    optimizer.step()

    val_loss = val_loss * hyperparameters["value_loss_coefficient"]
    distrib_entropy = - distrib_entropy * hyperparameters["entropy_coefficient"]

    return total_loss, val_loss, action_loss, distrib_entropy 


# Environment specific env_step function
def env_step( current_states, actions):
    next_state_list, reward_list, done_list = [], [], []
    for i in range(hyperparameters["workers"]):
        frame, reward, done = game_simulations[i].step(actions[i])
        if current_states is None:
            next_state = torch.cat([frame for i in range(hyperparameters["no_frames_to_network"])])
        else:
            next_state = torch.cat([current_states[i][1:], frame])
        next_state_list.append(next_state)
        reward_list.append([reward])
        done_list.append(done)

    return torch.stack(next_state_list), reward_list, done_list

if __name__ == '__main__': 

    hyperparameters = {
    "logs_dir":"exp_test_5",
    "model_dir":"",
    "no_train_iterations":100000,
    "lr":3e-4,
    "no_frames_to_network":4,
    "gamma":0.99,
    "workers":8,
    "update_freq":20,
    "entropy_coefficient":0.01,
    "value_loss_coefficient":0.5,
    "norm_clip_grad":40,
    "clip":0.1,
    "log_freq":100,
    "save_freq":1000,
    "actions_n":2,
    "frame_size":84
    }

    Experience = namedtuple('Experience', ('state', 'action', 'action_log_prob', 'val', 'reward', 'termination'))

    CUDA_DEVICE = torch.cuda.is_available()

   
    net = ActorCriticNetwork(hyperparameters)
    net.apply(net.init_weights)
    

    if CUDA_DEVICE:
        net = net.cuda()


    optimizer = torch.optim.Adam(net.parameters(), lr=hyperparameters["lr"])

    game_simulations = [Game(hyperparameters["frame_size"]) for i in range(hyperparameters["workers"])]

    scalar_writer = SummaryWriter(hyperparameters["logs_dir"])

    buffer = []
    
    #Training starts here
    ep_lens = np.zeros(hyperparameters["workers"])
    starting_actions = np.zeros(hyperparameters["workers"])
    current_states,rewards_, done_ = env_step(None, starting_actions)

    for current_itr in range(1, hyperparameters["no_train_iterations"]):

        vals, actions, action_log_probabilities = net.take_action(current_states)

        next_states, rewards, completions = env_step(current_states, actions)
        terminations = torch.FloatTensor([[0.0] if completed else [1.0] for completed in completions])

        buffer.append(Experience(current_states.data, actions.data, action_log_probabilities.data, vals.data, rewards, terminations))

        if current_itr % hyperparameters["update_freq"] == 0:
            total_loss, val_loss, action_loss, entropy_loss = optimise(Experience(*zip(*buffer)))
            buffer = []

        for worker in range(hyperparameters["workers"]):
            if not completions[worker]:
                ep_lens[worker] += 1
            else:
                scalar_writer.add_scalar('episode_length/' + str(worker), ep_lens[worker], current_itr)
                print(worker, ep_lens[worker])
                ep_lens[worker] = 0

        if current_itr % hyperparameters["save_freq"] == 0:
            if not os.path.exists(hyperparameters["logs_dir"]):
                os.mkdir(hyperparameters["logs_dir"])
            torch.save(net.state_dict(), f'{hyperparameters["logs_dir"]}/{str(current_itr).zfill(7)}.pt')

        if current_itr % hyperparameters["log_freq"] == 0:
            scalar_writer.add_scalar('loss/total', total_loss, current_itr)
            scalar_writer.add_scalar('loss/action', action_loss, current_itr)
            scalar_writer.add_scalar('loss/value', val_loss, current_itr)
            scalar_writer.add_scalar('loss/entropy', entropy_loss, current_itr)
            

        current_states = next_states
   