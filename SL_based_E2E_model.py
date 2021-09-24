import torch
import torch.nn as nn
import torch.nn.functional as F

class make_layer(nn.Module):
    def __init__(self, in_planes, out_planes, option):
        super(make_layer, self).__init__()
        self.option = option
        self.layer1 = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = 1, padding = 1),
                                    nn.BatchNorm2d(out_planes))                                    
        self.layer2 = nn.Sequential(nn.Conv2d(out_planes, out_planes, kernel_size = 3, stride = 1, padding = 1),
                                    nn.BatchNorm2d(out_planes))                                                                   

    def forward(self, x):
        out = F.relu(self.layer1(x))            
        if self.option == 2:
            out = F.relu(self.layer2(out))

        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = make_layer(1, 64, option = 1)
        self.conv2 = make_layer(64, 128, option = 1)
        self.conv3 = make_layer(128, 256, option = 2)
        self.conv4 = make_layer(256, 512, option = 2)
        self.conv5 = make_layer(512, 512, option = 2)

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(6144, 2048)
        self.fc2 = nn.Linear(2050, 2048)
        self.fc3 = nn.Linear(2048, 512)
        self.fc4 = nn.Linear(512, 1)

    def forward(self, x, indicator):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = self.maxpool(out)
        out = self.conv3(out)
        out = self.maxpool(out)
        out = self.conv4(out)
        out = self.maxpool(out)
        out = self.conv5(out)
        out = self.maxpool(out)

        out = torch.flatten(out, 1)

        out = F.relu(self.fc1(out))
        out = torch.cat([out, indicator], dim=1)
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)

        return out