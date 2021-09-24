import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.nn.init as init
from torch.distributions import Normal
from einops.layers.torch import Rearrange, Reduce

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class VAE(nn.Module):
    def __init__(self, cdim=1, hdim=32, channels=[32, 64, 128, 128, 128]):
        super(VAE, self).__init__()
        self.hdim = hdim
        self.channels = channels
        self.Encoder = nn.Sequential(
            nn.Conv2d(cdim, self.channels[0],3,1,1),
            nn.BatchNorm2d(self.channels[0]),
            nn.ReLU(True),
            nn.Conv2d(self.channels[0], self.channels[1],4,1,2),
            nn.BatchNorm2d(self.channels[1]),
            nn.ReLU(True),
            nn.AvgPool2d(2),
            nn.Conv2d(self.channels[1], self.channels[2],4,1,2),
            nn.BatchNorm2d(self.channels[2]),
            nn.ReLU(True),
            nn.AvgPool2d(2),
            nn.Conv2d(self.channels[2], self.channels[3],4,1,2),
            nn.BatchNorm2d(self.channels[3]),
            nn.ReLU(True),
            nn.AvgPool2d(2),
            nn.Conv2d(self.channels[3], self.channels[4],4,1,2),
            nn.BatchNorm2d(self.channels[4],1,2),
            nn.ReLU(True),  
            nn.AvgPool2d(2),                   
            View((-1,13*13*self.channels[4])),
            nn.Linear(13*13*self.channels[4], self.hdim*2),
            )
        self.Decoder = nn.Sequential(
            nn.Linear(self.hdim, 13*13*self.channels[4]),
            View((-1,self.channels[4],13,13)),
            nn.Conv2d(self.channels[4], self.channels[3],5,1,2),
            nn.BatchNorm2d(self.channels[3]),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),            
            nn.Conv2d(self.channels[3], self.channels[2],5,1,2),
            nn.BatchNorm2d(self.channels[2]),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest') ,                       
            nn.Conv2d(self.channels[2], self.channels[1],5,1,2),
            nn.BatchNorm2d(self.channels[1]),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),                        
            nn.Conv2d(self.channels[1], self.channels[0],5,1,2),
            nn.BatchNorm2d(self.channels[0]),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),                        
            nn.Conv2d(self.channels[0], 1,3,1,1),
            nn.Sigmoid()
            )

        self.weight_init()

    def reparameterize(self, mu, logvar, training):
        if training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x)  
        z = self.reparameterize(mu, logvar, True)
        return mu, logvar, z, self.decode(z)

    def forward_encode(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, False)
        return mu, logvar, z, self.decode(z)
    
    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def encode(self, x):
        x = self.Encoder(x)
        mu, logvar = x.chunk(2, dim=1)
        return mu, logvar

    def decode(self, x):
        x = self.Decoder(x)
        return x

class ValueNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class SoftQNetwork(nn.Module):
    
    def __init__(self, num_inputs, num_actions, hidden_size=256, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.linear3 = nn.Linear(hidden_size, 1)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    
    def __init__(self, num_inputs, num_actions, hidden_size=256, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state,scale,epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # if deterministic:
        #     action = torch.tanh(mean)

        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
 
        log_pi = normal.log_prob(z) -  torch.log(scale *(1 - action.pow(2)) + epsilon)
        log_pi = log_pi.sum(1, keepdim=True)

        return action, log_pi, mean, std


class MLPSoftQNetwork(nn.Module):
    
    def __init__(self, num_inputs, num_actions, hidden_size1=1400, hidden_size2=1024,  hidden_size3=256, init_w=3e-3):
        super(MLPSoftQNetwork, self).__init__()
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, hidden_size3) ###yna added
        
        self.linear4 = nn.Linear(hidden_size3, 1)
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        # print(state.shape, action.shape)
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x






class MLPPolicyNetwork(nn.Module):
    
    def __init__(self, num_inputs, num_actions, hidden_size1=1400, hidden_size2=1024, hidden_size3=256,init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(MLPPolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.num_inputs = num_inputs
        self.num_actions = num_actions


        self.linear1 = nn.Linear(num_inputs, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, hidden_size3) ###yna added

        self.mean_linear = nn.Linear(hidden_size3, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size3, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        # print('policy state size:',state.shape,', num_inputs: ', self.num_inputs, 'num_actions: ', self.num_actions)
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))

        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state,scale,epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # if deterministic:
        #     action = torch.tanh(mean)

        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
 
        log_pi = normal.log_prob(z) -  torch.log(scale *(1 - action.pow(2)) + epsilon)
        log_pi = log_pi.sum(1, keepdim=True)

        return action, log_pi, mean, std



class CNNSoftQNetwork(nn.Module):
    
    def __init__(self, obs_dim, num_actions, num_states = 256, channels=[32,64,64], hidden_size=256, init_w=3e-3):
        super(CNNSoftQNetwork, self).__init__()

        self.num_states = num_states # the number of CNN output vector
        self.channels = channels # the number of kernel of Convolution layers
        self.states_dim = obs_dim[1] # obs_dim (1,1,256,256)
        
        self.conv1 = self.conv(self.states_dim, self.channels[0])  #states_dim=1(channel of Input gray image)
        self.conv2 = self.conv(self.channels[0],self.channels[1])
        self.conv3 = self.conv(self.channels[1],self.channels[2])        
        self.fc = nn.Linear(1400*self.channels[2], num_states)


        self.linear1 = nn.Linear(num_states + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)


    def conv(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_num),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )


    def forward(self, state, action):

        out = self.conv1(state)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(-1, 1400*self.channels[2])
        # out = out.squeeze()
        # print(out.shape)
        out = self.fc(out)


        x = torch.cat([out, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class CNNPolicyNetwork(nn.Module):
    
    def __init__(self, obs_dim, num_actions,  num_states=256, channels=[32,64,64], hidden_size=256, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(CNNPolicyNetwork, self).__init__()

        self.num_states = num_states # the number of CNN output vector
        self.channels = channels # the number of kernel of Convolution layers
        self.states_dim = obs_dim[1] # obs_dim (1,1,256,256)
        
        self.conv1 = self.conv(self.states_dim, self.channels[0])  #states_dim=1(channel of Input gray image)
        self.conv2 = self.conv(self.channels[0],self.channels[1])
        self.conv3 = self.conv(self.channels[1],self.channels[2])        
        self.fc = nn.Linear(1400*self.channels[2], num_states)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_states, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)


    def conv(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_num),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, state):

  
        # Input: raw pixel gray image tensor / Output: 256 vectors
        out = self.conv1(state)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(-1, 1400*self.channels[2])
        # print('after convolution: ',out.shape)
        out = self.fc(out)

        # Input: 256 vectors to MLP Policy network
        x = F.relu(self.linear1(out))
        x = F.relu(self.linear2(x))

        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std


    def sample(self, state,scale,epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # if deterministic:
        #     action = torch.tanh(mean)

        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
 
        log_pi = normal.log_prob(z) -  torch.log(scale *(1 - action.pow(2)) + epsilon)
        log_pi = log_pi.sum(1, keepdim=True)

        return action, log_pi, mean, std




class CNNetwork(nn.Module):
    def __init__(self, cdim=1, num_states= 256, channels=[32,64,64]):
        super(CNNetwork,self).__init__()
        self.num_states = num_states
        self.channels = channels
        self.conv1 = self.conv(cdim, self.channels[0])
        self.conv2 = self.conv(self.channels[0],self.channels[1])
        self.conv3 = self.conv(self.channels[1],self.channels[2])        
        self.fc = nn.Linear(32*32*self.channels[2], num_states)
   

    def conv(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_num),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(-1, 32*32*self.channels[2])
        out = self.fc(out)
              
        return out




























## **********************************************************************
##                                    SL                              ##
## **********************************************************************

class NVIDIA_CNN_layer(nn.Module):
    def __init__(self, in_planes, out_planes, KERNEL_SIZE, STRIDE):
        super(NVIDIA_CNN_layer, self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size = KERNEL_SIZE, stride = STRIDE, padding = 1),
                                    nn.BatchNorm2d(out_planes))                                                                                                    

    def forward(self, x):
        out = F.relu(self.layer(x))

        return out

class NVIDIA_CNN_Network(nn.Module):
    def __init__(self):
        super(NVIDIA_CNN_Network, self).__init__()
        self.conv1 = NVIDIA_CNN_layer(1, 24, KERNEL_SIZE=5, STRIDE=2)
        self.conv2 = NVIDIA_CNN_layer(24, 36, KERNEL_SIZE=5, STRIDE=2)
        self.conv3 = NVIDIA_CNN_layer(36, 48, KERNEL_SIZE=5, STRIDE=2)
        self.conv4 = NVIDIA_CNN_layer(48, 64, KERNEL_SIZE=3, STRIDE=1)
        self.conv5 = NVIDIA_CNN_layer(64, 64, KERNEL_SIZE=3, STRIDE=1)
        
        self.fc1 = nn.Linear(1344, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = torch.flatten(out, 1)

        # out = torch.cat([out, indicator], dim=1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)

        return out