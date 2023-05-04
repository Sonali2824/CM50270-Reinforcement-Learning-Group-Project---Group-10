# This code was adapted from: https://github.com/lambders/drl-experiments/blob/master/a2c.py
import torch
# Heatmap Imports
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import numpy as np

class ActorCriticNetwork(torch.nn.Module):

    def __init__(self, hyperparameters):
        super(ActorCriticNetwork, self).__init__()
        
        self.convolution_1 = torch.nn.Conv2d(hyperparameters["no_frames_to_network"], 16, 8, 4)
        self.relu_1 = torch.nn.ReLU()
        self.convolution_2 = torch.nn.Conv2d(16, 32, 4, 2)
        self.relu_2 = torch.nn.ReLU()
        self.fully_connected_layer_3 = torch.nn.Linear(2592, 256) # TODO: Don't hard code
        self.relu_3 = torch.nn.ReLU()
        
        self.convolution_4 = torch.nn.Conv2d(hyperparameters["no_frames_to_network"], 16, 8, 4)
        self.relu_4 = torch.nn.ReLU()
        self.convolution_5 = torch.nn.Conv2d(16, 32, 4, 2)
        self.relu_5 = torch.nn.ReLU()
        self.fully_connected_layer_6 = torch.nn.Linear(2592, 256) # TODO: Don't hard code
        self.relu_6 = torch.nn.ReLU()
        
        self.actor = torch.nn.Linear(256, hyperparameters["actions_n"])
        self.critic = torch.nn.Linear(256, 1)
        self.softmax = torch.nn.Softmax()
        self.logsoftmax = torch.nn.LogSoftmax()


    def init_weights(self, m):
        if type(m) == torch.nn.Conv2d or type(m) == torch.nn.Linear:
            torch.nn.init.uniform(m.weight, -0.01, 0.01)
            m.bias.data.fill_(0.01)


    def forward(self, x):
        # Forward pass
        x_input = x
        x_a = self.relu_1(self.convolution_1(x_input))
        x_a = self.relu_2(self.convolution_2(x_a))
        x_1 = x_a
        x_a = x_a.view(x_a.size()[0], -1)
        x_a = self.relu_3(self.fully_connected_layer_3(x_a))
        
        x_c = self.relu_4(self.convolution_4(x_input))
        x_c = self.relu_5(self.convolution_5(x_c))
        x_c = x_c.view(x_c.size()[0], -1)
        x_c = self.relu_6(self.fully_connected_layer_6(x_c))
        
        action_logits = self.actor(x_a)
        value = self.critic(x_c)

        # input
        image_rescaled = (255*x_input[0][0]).clamp(0, 255).byte()  # Rescale and convert to byte tensor
        image_rgb = image_rescaled.unsqueeze(2).repeat(1, 1, 3)  # Add a third dimension and repeat values along it
        cv2.imwrite('input_image1.png', image_rgb.numpy())  # Save the image using cv2.imwrite()
        
        # Compute heatmap
        x_1_np = x_1.detach().cpu().numpy()
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
        cv2.imwrite('heatmap1.png', bgr_heatmap_resized)
        return value, action_logits


    def take_action(self, x):
         # Forward pass
        vals, action_logits = self.forward(x)
        probs = self.softmax(action_logits)
        log_probs = self.logsoftmax(action_logits)

        # Choose action stochastically
        actions = probs.multinomial(1)

        # Evaluate action
        action_log_probs = log_probs.gather(1, actions)
        dist_entropy = -(log_probs * probs).sum(-1).mean()
        return vals, actions, action_log_probs

    def action_evaluation(self, x, actions):
        # Forward pass 
        value, action_logits = self.forward(x)
        probs = self.softmax(action_logits)
        log_probs = self.logsoftmax(action_logits)

        # Evaluate actions
        action_log_probs = log_probs.gather(1, actions)
        dist_entropy = -(log_probs * probs).sum(-1).mean()
        return value, action_log_probs, dist_entropy