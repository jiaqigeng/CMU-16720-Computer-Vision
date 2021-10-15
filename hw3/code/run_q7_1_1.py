import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.io
from nn import *
import matplotlib.pyplot as plt


train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = torch.from_numpy(train_data['train_data']), torch.from_numpy(train_data['train_labels'])
valid_x, valid_y = torch.from_numpy(valid_data['valid_data']), torch.from_numpy(valid_data['valid_labels'])
test_x, test_y = torch.from_numpy(test_data['test_data']), torch.from_numpy(test_data['test_labels'])


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(train_x.shape[1], 64)
        self.fc2 = nn.Linear(64, train_y.shape[1])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

train_batches = get_random_batches(train_x, train_y, 16)

train_acc_history = []
train_loss_history = []
val_acc_history = []
val_loss_history = []

for epoch in range(50):
    for phase in ['train', 'val']:
        running_loss = 0.0
        total = 0
        correct = 0

        if phase == 'train':
            batches = train_batches
            net.train()
        else:
            batches = [(valid_x, valid_y)]
            net.eval()

        for xb, yb in batches:
            inputs, labels = xb.float(), np.argmax(yb.float(), axis=1)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            if phase == 'train':
                loss.backward()
                optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item() * inputs.size(0)

        if phase == 'train':
            print("Train Acc:", correct / total, " Train Loss:", running_loss / total)
            train_acc_history.append(correct / total)
            train_loss_history.append(running_loss / total)
        else:
            print("Val Acc:", correct / total, " Val Loss:", running_loss / total)
            val_acc_history.append(correct / total)
            val_loss_history.append(running_loss / total)

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


net.eval()
correct = 0
total = 0
with torch.no_grad():
    images, labels = test_x.float(), np.argmax(test_y.float(), axis=1)
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('Test Acc: %d %%' % (100 * correct / total))
