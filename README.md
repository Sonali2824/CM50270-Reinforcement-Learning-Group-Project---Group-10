# CM50270-Reinforcement-Learning-Group-Project---Group-10
Project Requirement : Applying reinforcement learning methods to solve a problem of the group's choice. The problem's state/action space must be large enough so that tabular reinforcement learning methods cannot be used to solve it effectively. The objective is to solve the chosen problem to the best of the group's ability.


## ⚙️ Running the code

```sh
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
3. The parameter "logs_dir" has to be changed to a new logging folder for every training process.
4. The actor-critic models and the logging information is saved by the training code under the folder specified by the "logs_dir".
5. In both eval_a2c.py and eval_ppo.py the "weights_dir" variable is assigned the path of the model which has to be tested.

## 📖 Arguments and Hyperparameters
Arguments and hyperparameters are passed to both the PPO and A2C agent using the dictionary "hyperparameters" found in their respective main functions.
``` sh
hyperparameters = {
    "logs_dir":"exp_test_clip_0.2_longer_34", # Name of the logging directory
    "model_dir":"", # Optional: name of the directory to store the actor-critic models seprately from the logging directory
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
