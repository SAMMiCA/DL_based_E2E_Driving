import os
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from skimage import io
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import time
from SL_based_E2E_model import Net # hvae to change

## Class to create a dataset ##
class e2e_Dataset(Dataset):
    def __init__(self, csv_file, number, transform=None):
        super(e2e_Dataset, self).__init__()
        
        self.landmarks_frame = pd.read_csv(csv_file)[0:number]
        self.transform = transform

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(str(self.landmarks_frame.iloc[idx, 0]))
        img = io.imread(img_name)

        img_roi = img[375:480, 0:640].copy()
        img_PIL = Image.fromarray(img_roi)

        if self.transform:
            img_result = self.transform(img_PIL)

        steering = float(self.landmarks_frame.iloc[idx, 1])
        steering = np.array([steering])
        steering = torch.from_numpy(steering).float()
        
        ########## indicator #########
        ## if left turn    : [1, 0] ##
        ## elif right turn : [0, 1] ##
        ## elif straight   : [0, 0] ##
        left_indicator = float(self.landmarks_frame.iloc[idx, 2])
        right_indicator = float(self.landmarks_frame.iloc[idx, 3])
        indicator = np.array([left_indicator, right_indicator])
        indicator = torch.from_numpy(indicator).float() * 1000

        return img_result, steering, indicator

    def __len__(self):
        return len(self.landmarks_frame)


## train function ##
def train(model, loader_tra, optimizer, log_interval):
    model.train()
    error_each_batch_list = []
    for batch_idx, (image, steer, indicator) in enumerate(loader_tra):
        image = image.to(DEVICE)
        steer = steer.to(DEVICE)
        indicator = indicator.to(DEVICE)

        optimizer.zero_grad()
        output = model(image, indicator)
        loss = criterion(output, steer)
        loss.backward()
        optimizer.step()
        
        output = output.cpu().data.numpy()
        steer = steer.cpu().data.numpy()

        error_each_batch = np.mean(np.abs(output - steer)).item()
        error_each_batch_list.append(error_each_batch)

        if batch_idx % log_interval == 0:
            print("{} - Train Epoch: {} [{}/{}({:.0f})%)  {:.4f}sec)]   Train Error: {:.6f}".format(
                model_name, Epoch, batch_idx*len(image), len(loader_tra.dataset), 
                100. * batch_idx / len(loader_tra), time.time()-start_time, error_each_batch))
    
    error_tra = np.mean(error_each_batch_list).item()

    return error_tra


## evaluation function ##
def evaluate(model, loader_val):
    model.eval()
    error_each_batch_list = []

    with torch.no_grad():
        for image, steer, indicator in loader_val:
            image = image.to(DEVICE)
            steer = steer.to(DEVICE)
            indicator = indicator.to(DEVICE)

            output = model(image, indicator)    
            output = output.cpu().data.numpy()
            steer = steer.cpu().data.numpy()
            
            error_each_batch = np.mean(np.abs(output - steer)).item()
            error_each_batch_list.append(error_each_batch)

    error_val = np.mean(error_each_batch_list).item()

    return error_val


## Define the root mean square error loss ## 
class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()
    def forward(self, x, y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss


TRANSFORM = transforms.Compose([transforms.Resize((64, 200)),
                                transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5), (0.5))
                                ])

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
start_time = time.time()

model_name = 'INDICATOR_MODEL'
model_dict_PATH = 'model_e2e_sl_' + model_name + '.dict.pth'

## Create the train/validation loss csv file ##
tra_loss_file_name = '' + 'tra_' + model_name + '.csv'
csv_tra = open(tra_loss_file_name, "w")
csv_tra.close()

val_loss_file_name = '' + 'val_' + model_name + '.csv'
csv_val = open(val_loss_file_name, "w")
csv_val.close()

## create model ##
model = Net().to(DEVICE)

## Hyper parameters ##
BATCH_SIZE = 256
EPOCH = 50
LEARNING_RATE = 0.0001
TRAIN_DATA_NUM = 53000
VAL_DATA_NUM = 5900
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = RMSELoss()

## Train/Validation data information csv file ##
csv_file_tra = ''
csv_file_val = ''

## load data ##
print("DATA LOADING...\n")
dataset_tra = e2e_Dataset(csv_file_tra, TRAIN_DATA_NUM, transform=TRANSFORM)
dataset_val = e2e_Dataset(csv_file_val, VAL_DATA_NUM, transform=TRANSFORM)
loader_tra = DataLoader(dataset = dataset_tra, batch_size = BATCH_SIZE, shuffle = True)
loader_val = DataLoader(dataset = dataset_val, batch_size = BATCH_SIZE, shuffle = False)

print("\n***********************************************************")
print("***********************************************************")
for (img, steering, indicator) in loader_tra:
    print("Image data shape : {}\nSteer data shape : {}".format(img.shape, steering.shape))
    print("Indicator data shape : {}".format(indicator.shape))
    break
print("Number of [Training data : {} | Validation data : {}]".format(len(dataset_tra), len(dataset_val)))
print("***********************************************************")
print("***********************************************************\n\n\n")

## RUN ##
error_tra_list = []
error_val_list = []
for Epoch in range(EPOCH):
    error_tra = train(model, loader_tra, optimizer, log_interval = 10)
    error_val = evaluate(model, loader_val)

    error_tra_list.append(error_tra)
    error_val_list.append(error_val)

    print("\n[Epoch: {}],   Validation Error: {:.4f}\n".format(Epoch, error_val))
    
    plt.plot(error_tra_list, 'k', ls='-', label='Training Loss')
    plt.plot(error_val_list, 'b', ls='-', label='Validation Loss')
    plt.legend(loc='upper right')
    plt.xlabel('Epochs', fontsize=17)
    plt.ylabel('Error [degree]', fontsize=17)
    plt.title(model_name, fontsize=17)
    plt.grid(True)
    plt.savefig('RESULT_' + model_name + '.png')
    plt.clf()

    csv_tra = open(tra_loss_file_name, "a+")
    csv_tra.write(str(error_tra) + "\n")
    csv_tra.close()

    csv_val = open(val_loss_file_name, "a+")
    csv_val.write(str(error_val) + "\n")
    csv_val.close()

torch.save(model.state_dict(), model_dict_PATH, _use_new_zipfile_serialization=False)