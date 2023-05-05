# This code was adapted from: https://github.com/lambders/drl-experiments/blob/master/dqn.py
from dqn import DQNAgent

# Setting attributing in a namespace
class DQNParameters:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

if __name__ == '__main__': 
    params = DQNParameters(
        mode="train", 
        exp_name="exp7", 
        weights_dir="", 
        n_train_iterations=200001, 
        learning_rate=5e-5, 
        len_agent_history=4, 
        discount_factor=0.99, 
        batch_size=32, 
        initial_exploration=0.99, 
        final_exploration=0.01, 
        final_exploration_frame=1000000, 
        replay_mem_size=25000, 
        log_frequency=100, 
        save_frequency=100000, 
        n_actions=2, 
        frame_size=84)

    agent = DQNAgent(params)


    if params.mode == 'train':
        agent.train()
    elif params.mode == 'eval':
        agent.play_game()
