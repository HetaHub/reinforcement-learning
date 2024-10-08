import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class NoisyLinear(nn.Linear):
    #we set a matrix for sigma o , value of u wil put in nn.Linear matrix. We put sigma into nn.Parameters
    #to let it become trainable. register_buffer will create a tensor, which update by nn.Module when 
    #back propagation. Then we create parameters and buffer for the bias in Noisy Layer. Then we use
    #reset_parameters to override nn.Linear moethod, which will initialize noise layer
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()

    #Initialize nn.Linear, set weighting and bias
    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    #we random sample from weight and bias buffer as noise, and perform linear transformation to input as 
    #nn.Linear
    def forward(self, input):
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.data
        return F.linear(input, self.weight + self.sigma_weight * self.epsilon_weight.data, bias)

#we transform basic DQN to NoisyNet, which use NoisyLinear or NoisyFactorizedLinear, we need to delete
#orginial epsilon greedy method, we use signal to noise ratio (SNR) to check the noise level, it is the
#ratio of RMS(u) to RMS(o), RMS is weighted mean square, SNR here means the frequency of 
#fixed component > noise
class NoisyFactorizedLinear(nn.Linear):
    """
    NoisyNet layer with factorized gaussian noise
    N.B. nn.Linear already initializes weight and bias to
    """
    def __init__(self, in_features, out_features, sigma_zero=0.4, bias=True):
        super(NoisyFactorizedLinear, self).__init__(in_features, out_features, bias=bias)
        sigma_init = sigma_zero / math.sqrt(in_features)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer("epsilon_input", torch.zeros(1, in_features))
        self.register_buffer("epsilon_output", torch.zeros(out_features, 1))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))

    def forward(self, input):
        self.epsilon_input.normal_()
        self.epsilon_output.normal_()

        func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))
        eps_in = func(self.epsilon_input.data)
        eps_out = func(self.epsilon_output.data)

        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t()
        noise_v = torch.mul(eps_in, eps_out)
        return F.linear(input, self.weight + self.sigma_weight * noise_v, bias)
    
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