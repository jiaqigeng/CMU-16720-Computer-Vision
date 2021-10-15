import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.io
from nn import *
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import copy
import matplotlib.pyplot as plt


batch_size = 16

train_loader = DataLoader(datasets.MNIST('../data', download=True, train=True, transform=transforms.ToTensor()),
                          batch_size=batch_size)
valid_loader = DataLoader(datasets.MNIST('../data', download=True, train=False, transform=transforms.ToTensor()),
                          batch_size=batch_size)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = x.view(-1, 16 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

best_model_state = copy.deepcopy(model.state_dict())
best_acc = 0.0
train_acc_history = []
train_loss_history = []
val_acc_history = []
val_loss_history = []

for epoch in range(10):
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.
        running_total = 0.
        running_correct = 0.

        if phase == 'train':
            dataset_loader = train_loader
        else:
            dataset_loader = valid_loader

        for i, data in enumerate(dataset_loader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if phase == 'train':
                loss.backward()
                optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            running_total += labels.size(0)
            running_correct += (predicted == labels).sum().item()
            running_loss += loss.item() * inputs.size(0)

        if phase == 'train':
            print("Train Acc:", running_correct / running_total, " Train Loss:", running_loss / running_total)
            train_acc_history.append(running_correct / running_total)
            train_loss_history.append(running_loss / running_total)
        else:
            acc = running_correct / running_total
            if acc > best_acc:
                best_acc = acc
                best_model_state = copy.deepcopy(model.state_dict())

            print("Val Acc:", running_correct / running_total, " Val Loss:", running_loss / running_total)
            val_acc_history.append(running_correct / running_total)
            val_loss_history.append(running_loss / running_total)

print('Finished Training')

plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(train_acc_history, label='Training Acc')
plt.plot(val_acc_history, label='Val Acc')
plt.legend()
plt.show()

plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(train_loss_history, label='Training Loss')
plt.plot(val_loss_history, label='Val Loss')
plt.legend()
plt.show()
