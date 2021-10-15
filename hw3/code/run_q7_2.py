import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from nn import *
from q4 import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import torch
from torchvision import transforms, datasets
import torchvision
import copy


train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

flower17_trainset = datasets.ImageFolder(root='../data/oxford-flowers17/train', transform=train_transform)
train_loader = torch.utils.data.DataLoader(flower17_trainset, batch_size=8, shuffle=True, num_workers=4)
flower17_valset = datasets.ImageFolder(root='../data/oxford-flowers17/val', transform=val_transform)
val_loader = torch.utils.data.DataLoader(flower17_valset, batch_size=8, shuffle=False, num_workers=4)
flower17_testset = datasets.ImageFolder(root='../data/oxford-flowers17/test', transform=val_transform)
test_loader = torch.utils.data.DataLoader(flower17_testset, batch_size=8, shuffle=False, num_workers=4)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 24 * 24, 512)
        self.fc2 = nn.Linear(512, 17)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 32 * 24 * 24)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    # change model name to switch between custom and fine_tuned
    # this is the only place you need to change in this file

    # model_name = "custom"
    model_name = "fine_tuned"

    if model_name == "custom":
        model = Net()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    else:
        model = torchvision.models.squeezenet1_1(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[1] = nn.Conv2d(512, 17, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = 17

        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

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
                dataset_loader = val_loader

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

    if model_name == "custom":
        torch.save(best_model_state, "./q7_2_custom_new.pth")
    else:
        torch.save(best_model_state, "./q7_2_fine_tuned.pth")

    if model_name == "custom":
        model.load_state_dict(torch.load("./q7_2_custom_new.pth"))
    else:
        model.load_state_dict(torch.load("./q7_2_fine_tuned.pth"))

    model.eval()
    total = 0
    correct = 0
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print()
    print("Test Accuracy:", correct / total)


if __name__ == '__main__':
    main()
