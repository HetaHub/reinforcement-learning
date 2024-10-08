import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    # we use varies number for output parameters according to the number of input parameters, such as in 84 * 84, we have 3136 outputs
    # it will return the number of parameters in the function
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    # we use forward function as flatten layer, for example if we have a (2, 3, 4) shape tensor, it is a 24 elements 3 dimensional tensor,
    # we can reshape it with T.view(6, 4) as 6 row 4 column 2 dimensional tensor, calling T.view(-1, 4) or T.view(6, -1) has the same result
    # -1 means let the system decide the remaining number according the current number given. The input of the 4 dimensional tensor is 
    # batch size, color, screenshot width and height, output is 2 dimensional, batch size and other parameters.
    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)