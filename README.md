# CM50270-Reinforcement-Learning-Group-Project---Group-10
Project Requirement : Applying reinforcement learning methods to solve a problem of the group's choice. The problem's state/action space must be large enough so that tabular reinforcement learning methods cannot be used to solve it effectively. The objective is to solve the chosen problem to the best of the group's ability.

The environment used: Flappy Bird 🐦. Training is performed on pixelated data and not features.

## ⚙️ Running the code
Before running the code, ensure you download the enviornment "game" code from the github repository using the <a href="https://github.com/lambders/flappy-bird/tree/6ff0b956886f7b2ac8e08907ce7883bf810ca338"> <b>Game folder link</b></a>. The "game" folder is required to setup the Flappy Bird Enviornment.


```sh
# For DQN agent training
python dqn_main.py # ensure the mode in the params namespace is set to "train"

# For DQN agent evaluation
python dqn_main.py # ensure the mode in the params namespace is set to "eval"

# For A2C agent training
python a2c_main.py

# For A2C agent evaluation
python eval_a2c.py

# For PPO agent training
python ppo_main.py

# For PPO agent evaluation
python eval_ppo.py

# You can also  visualise the learning curves via TensorBoard
tensorboard --logdir <exp_name> # exp_name refers to the log directory
```


## 📌 Additonal Information

1. A2C Actor-Critic Network: a2c_network.py
2. PPO Actor-Critic Network: ppo_network.py
3. DQN Network: dqn_main.py
4. The networks also comprise the code implementation of heatmaps. The heatmaps are stored in the parent folder.
5. The parameter "logs_dir" (in the cases of A2C and PPO) and "exp_name" (in case of DQN) has to be changed to a new logging folder for every training process.
6. The actor-critic models and the logging information is saved by the training code under the folder specified by the "logs_dir" in the cases of PPO and A2C.
7. The DQN model and the logging information is saved by the training code under the folder specified by the "exp_name" in the case of DQN.
8. In both eval_a2c.py and eval_ppo.py the "weights_dir" variable is assigned the model which has to be tested.
9. In order to disable the rendering add the below code
```sh
import os
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['SDL_VIDEODRIVER']='dummy'
```

## 📖 Arguments and Hyperparameters
Arguments and hyperparameters are passed to both the PPO and A2C agent using the dictionary "hyperparameters" found in their respective main functions.
``` sh
hyperparameters = {
    "logs_dir":"exp_test_clip_0.2_longer_34", # Name of the logging directory
    "model_dir":"", # Optional: name of the directory to store and access the actor-critic models separately from the logging directory, ensure you change the saving directory in the code accordingly.
    "no_train_iterations":1000000, # Number of training iterations
    "lr":3e-4, # learning rate
    "no_frames_to_network":4, # Number of frames passed to the actor-critic network
    "gamma":0.99, # Discount factor
    "workers":8, # Number of workers
    "update_freq":20, # Number of batch frequency updates, refreshes the buffer after every x actions
    "entropy_coefficient":0.01, # Entropy coefficient for exploration
    "value_loss_coefficient":0.5, # Value loss coefficient 
    "norm_clip_grad":40, # Bound for clipping gradients
    "clip":0.2, # bound for clipping gradients to prevent instable learning in case of PPO
    "log_freq":100, # Frequency at which logging takes place
    "save_freq":10000, # Frequency at which saving takes place
    "actions_n":2, # Denotes the 2 actions of the enviornment, flapping the bird's wing or no action
    "frame_size":84 # Size of game frame in pixels
    }
```

Arguments and hyperparameters different to DQN agent. DQN uses the namespace "params" found in dqn_main.py.
``` sh
    params = DQNParameters(
        func_mode="train",  # "train" or "eval" mode
        size_of_batch=32, # Batch size
        init_explor=1.0, # Epsilon greedy action selection parameter
        fin_explor=0.01, # Epsilon greedy action selection parameter
        fin_explor_frame=1000000, # Epsilon greedy action selection parameter
        replay_size=25000, # maximum number of transitions in replay memory
        )
```
Note: The graph results are not reproducable as the enviornment does not comprise a seed function.

To implement the environment, we utilised the `drl-experiments` repository by [@lambders](https://github.com/lambders/drl-experiments). The DQN, A2C and PPO implementations in [@lambders](https://github.com/lambders/drl-experiments) were considered as baselines for our project. Our network and general flow of the algorithms was adapted from [@lambders](https://github.com/lambders/drl-experiments).
