import os
import torch
import numpy as np 
from tensorboardX import SummaryWriter
from collections import namedtuple

from game.wrapper import Game
from ppo_network import Actor, Critic
import torch.nn.functional as F


def ac_forward(actor, critic, state):
    action_logits = actor.forward(state)
    vals = critic.forward(state)
    probs = actor.softmax(action_logits)
    log_probs = actor.logsoftmax(action_logits)
    return vals, probs, log_probs

def ac_evaluate_actions(actions, log_probs, probs):
    action_log_probabilities = log_probs.gather(1, actions)
    distrib_entropy = -(log_probs * probs).sum(-1).mean()
    return action_log_probabilities, distrib_entropy

def get_action(actor, critic, state):
    vals, probs, log_probs = ac_forward(actor, critic, state)
    actions = probs.multinomial(1)
    action_log_probabilities, distrib_entropy = ac_evaluate_actions(actions, log_probs, probs)
    return vals, actions, action_log_probabilities

def evaluate_actions(actor, critic, state, actions):
    val, probs, log_probs = ac_forward(actor, critic, state)
    action_log_probabilities, distrib_entropy = ac_evaluate_actions(actions, log_probs, probs)
    return val, action_log_probabilities, distrib_entropy

def optimise(buffer):

    action_tensor = torch.stack(buffer.action).detach()
    action_log_prob_tensor = torch.stack(buffer.action_log_prob).detach()
    reward_tensor = torch.tensor(buffer.reward).detach()
    state_tensor = torch.stack(buffer.state).detach()
    termination_tensor = torch.stack(buffer.termination).detach()
    val_tensor = torch.stack(buffer.val).detach()

    batch_data = {
        'action': action_tensor,
        'action_log_prob': action_log_prob_tensor,
        'reward': reward_tensor,
        'state': state_tensor,
        'termination': termination_tensor,
        'val': val_tensor
    }

    state_dim = batch_data['state'].size()[2:]
    action_dim = batch_data['action'].size()[-1]

    returns = torch.zeros(hyperparameters["update_freq"] + 1, hyperparameters["workers"], 1)
    for i in reversed(range(hyperparameters["update_freq"])):
        returns[i] = returns[i+1] * hyperparameters["gamma"] * batch_data['termination'][i] + batch_data['reward'][i]
    returns = returns[:-1]
    returns = (returns - returns.mean()) / (returns.std() + 1e-5)

    vals, action_log_probabilities, distrib_entropy = evaluate_actions(actor, critic, batch_data['state'].view(-1, *state_dim), batch_data['action'].view(-1, action_dim)) 
    vals = vals.view(hyperparameters["update_freq"], hyperparameters["workers"], 1)
    action_log_probabilities = action_log_probabilities.view(hyperparameters["update_freq"], hyperparameters["workers"], 1)

    advs = returns - vals.detach()

    ratio = torch.exp(action_log_probabilities - batch_data['action_log_prob'].detach())
    surrogate_1 = ratio * advs
    surrogate_2 = torch.clamp(ratio, 1 - hyperparameters["clip"], 1 + hyperparameters["clip"]) * advs
    action_loss = -torch.min(surrogate_1, surrogate_2).mean()

    val_loss = F.mse_loss(vals, returns)
    val_loss = hyperparameters["value_loss_coefficient"] * val_loss

    total_loss = val_loss + action_loss - distrib_entropy * hyperparameters["entropy_coefficient"]

    actor_optimiser.zero_grad()
    critic_optimiser.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm(actor.parameters(), hyperparameters["norm_clip_grad"])
    torch.nn.utils.clip_grad_norm(critic.parameters(), hyperparameters["norm_clip_grad"])
    actor_optimiser.step()
    critic_optimiser.step()

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
    "logs_dir":"exp_test_clip_0.2_longer_34",
    "model_dir":"",
    "no_train_iterations":1000000,
    "lr":3e-4,
    "no_frames_to_network":4,
    "gamma":0.99,
    "workers":8,
    "update_freq":20,
    "entropy_coefficient":0.01,
    "value_loss_coefficient":0.5,
    "norm_clip_grad":40,
    "clip":0.2,
    "log_freq":100,
    "save_freq":10000,
    "actions_n":2,
    "frame_size":84
    }

    Experience = namedtuple('Experience', ('state', 'action', 'action_log_prob', 'val', 'reward', 'termination'))

    CUDA_DEVICE = torch.cuda.is_available()

    # Create Actor Critic Networks
    actor = Actor(hyperparameters)
    critic = Critic(hyperparameters)

    actor.apply(actor.init_weights)
    critic.apply(critic.init_weights)

    if CUDA_DEVICE:
        actor = actor.cuda()
        critic = critic.cuda()

    actor_optimiser = torch.optim.Adam(actor.parameters(), lr=hyperparameters["lr"])
    critic_optimiser = torch.optim.Adam(critic.parameters(), lr=hyperparameters["lr"])

    game_simulations = [Game(hyperparameters["frame_size"]) for i in range(hyperparameters["workers"])]

    scalar_writer = SummaryWriter(hyperparameters["logs_dir"])

    buffer = []
    
    #Training starts here
    ep_lens = np.zeros(hyperparameters["workers"])
    starting_actions = np.zeros(hyperparameters["workers"])
    current_states,rewards_, done_ = env_step(None, starting_actions)

    for current_itr in range(1, hyperparameters["no_train_iterations"]):

        vals, actions, action_log_probabilities = get_action(actor, critic, current_states)

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
            torch.save(actor.state_dict(), f'{hyperparameters["logs_dir"]}/{(str(current_itr)+"_actor_").zfill(7)}.pt')
            torch.save(critic.state_dict(), f'{hyperparameters["logs_dir"]}/{(str(current_itr)+"_critic_").zfill(7)}.pt')

        if current_itr % hyperparameters["log_freq"] == 0:
            scalar_writer.add_scalar('loss/total', total_loss, current_itr)
            scalar_writer.add_scalar('loss/action', action_loss, current_itr)
            scalar_writer.add_scalar('loss/value', val_loss, current_itr)
            scalar_writer.add_scalar('loss/entropy', entropy_loss, current_itr)
            

        current_states = next_states
   