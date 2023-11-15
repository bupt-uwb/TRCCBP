import torch
import os
from torch.utils.data import Dataset, DataLoader, random_split
from models.model import *
from data_loader import *
import numpy as np
import scipy.io as sio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
epochs = 50
lr = 0.0001

dataset = sio.loadmat('./data/dataset_28.mat')
# print(len(dataset))
full_dataset = LoadDataset(dataset)
train_size = int(len(full_dataset) * 0.8)
test_size = len(full_dataset) - train_size
# print(train_size,test_size)
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], torch.manual_seed(0))
# print(train_dataset)
isolate_dataset = LoadDataset(sio.loadmat('./data/dataset_3.mat'))

train_loader = DataLoader(train_dataset, batch_size, drop_last=False, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size, drop_last=False, shuffle=True)
isolate_test_loader = DataLoader(isolate_dataset, batch_size, drop_last=False, shuffle=True)


def train(model, train_loader, loss, optimizer, epoch):
    model.train()
    Loss = 0
    for batch_idx, data in enumerate(train_loader):
        Data = data["feature"].to(device)
        target = data["gt"].to(device)
        ids = data["id"]
        # print(type(ids))
        status = data['status']
        # print(target)
        # print(len(status))
        # print(Data.size())

        # print(target.squeeze())
        optimizer.zero_grad()

        output = model(Data)

        loss_curr = loss(output, target)

        Loss += loss_curr
        loss_curr.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(Data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss_curr.item()))
    print('\nTrain set: Loss: {:.4f}'.format(
        Loss / (batch_idx + 1)))
    return Loss / (batch_idx + 1)


def test(model, test_loader, loss, best=float('inf'), isolation=False):
    model.eval()
    total_Loss = 0

    pred_list, target_list = np.zeros((1, 2)), np.zeros((1, 2))
    ids_list, status_list = np.zeros((1, 1)), np.zeros((1, 1))
    # print(isolation)
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            # print(batch_idx)
            Data = data["feature"].to(device)
            target = data["gt"].to(device)
            ids = np.array(data["id"]).reshape(len(data["id"]),1)
            status = np.array(data['status'],dtype=np.object).reshape(len(data["status"]),1)
            # print(status.shape)
            output = model(Data)

            loss_curr = loss(output, target).item()
            total_Loss += loss_curr
            pred_list = np.append(pred_list, output.cpu().numpy(), axis=0)
            target_list = np.append(target_list, target.cpu().numpy(), axis=0)
            ids_list = np.append(ids_list, ids, axis=0)
            status_list = np.append(status_list, status, axis=0)

            # print(output.shape)
            # print(pred.shape)
            # print(pred.eq(target.view_as(pred)).sum().item())
    # print(status_list[1:,:].shape)
    # print(pred_list.shape)
    # print('\nTest set: Loss: {:.4f}\n'.format(total_Loss / (batch_idx + 1)))
    if not isolation and total_Loss / (batch_idx + 1) < best:
        sio.savemat('./results/result.mat', {'BP_ES': pred_list[1:,:],'BP_GT': target_list[1:,:],'ids_list':ids_list[1:,:],'status_list':status_list[1:,:]})
        # print('test', batch_idx + 1)
    if isolation and total_Loss / (batch_idx + 1) < best:
        sio.savemat('./results/isolate_result.mat', {'BP_ES': pred_list[1:,:],'BP_GT': target_list[1:,:],'ids_list':ids_list[1:,:],'status_list':status_list[1:,:]})
        # print('isolate', batch_idx + 1)
    return total_Loss / (batch_idx + 1)


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss


# def rmse(predictions, targets):
#     differences = predictions - targets
#     differences_squared = differences ** 2
#     mean_of_differences_squared = differences_squared.mean()
#     rmse_val = np.sqrt(mean_of_differences_squared)
#     return rmse_val


model = Net()
# if torch.cuda.device_count() > 1:
#     model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
model = model.to(device)
# loss = nn.MSELoss()
loss = nn.L1Loss()
# loss = RMSELoss()
loss = loss.to(device)
optimizer = optim.Adam(model.parameters(), lr)
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
best_score, best_epoch, test_loss, isolate_loss= float('inf'), 0, float('inf'), float('inf')
train_loss_list = np.zeros([epochs, 1], dtype=float)
test_loss_list = np.zeros([epochs, 1], dtype=float)
isolate_loss_list = np.zeros([epochs, 1], dtype=float)

# 训练模型
for epoch in range(epochs):
    train_loss_list[epoch] = train(model, train_loader, loss, optimizer, epoch).item()
    # print(train(model, train_loader, loss, optimizer, epoch))
    test_loss = test(model, test_loader, loss, best_score, isolation=False)
    test_loss_list[epoch] = test_loss
    isolate_loss = test(model, isolate_test_loader, loss, best_score, isolation=True)
    isolate_loss_list[epoch] = isolate_loss

    if test_loss < best_score:  # save best model
        best_score = test_loss
        best_epoch = epoch
        torch.save(model, 'checkpoints/best.pth')
    print('Test_Loss={:.4f} Epoch:{}'.format(test_loss, epoch))
    print('Isolate_Loss={:.4f} Epoch:{}'.format(isolate_loss, epoch))
    print('Best_Loss={:.4f} Epoch:{}\n'.format(best_score, best_epoch))
    scheduler.step()

sio.savemat('./results/loss.mat', {'train_loss': train_loss_list, 'test_loss': test_loss_list, 'isolate_loss':isolate_loss_list})
# accuracy = test(model, test_loader, loss)
# torch.save(model, 'checkpoints/latest_' + str(round(accuracy, 2)) + '%' + '.pth')
