import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import warnings
import skimage.io
import skimage.transform
import os
from nn import *
from q4 import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches


batch_size = 16

train_loader = DataLoader(datasets.EMNIST('../data', download=True, train=True, split='balanced',
                                          transform=transforms.Compose([transforms.ToTensor()])),
                          batch_size=batch_size, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 47)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = x.view(-1, 16 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = Net()

# uncomment to train
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

for epoch in range(5):

    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data

        # Normalization
        batch_imgs = inputs[:, 0, :, :].detach().numpy()
        batch_imgs = batch_imgs.reshape(batch_imgs.shape[0], -1)
        batch_imgs = -(batch_imgs - np.mean(batch_imgs, axis=1).reshape(-1, 1)) / \
                     np.std(batch_imgs, axis=1).reshape(-1, 1)
        inputs = torch.from_numpy(batch_imgs.reshape(batch_imgs.shape[0], 1, 28, 28))

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.shape[0]

    print("loss:", running_loss / len(train_loader.dataset))

print('Finished Training')
torch.save(net.state_dict(), "./q7_1_4.pth")

net.load_state_dict(torch.load("./q7_1_4.pth"))
net.eval()

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images', img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw)
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox

        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                            fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()

    # find the rows using..RANSAC, counting, clustering, etc.
    ##########################
    bboxes = sorted(bboxes, key=lambda x: x[0])
    prev_y1, prev_x1, prev_y2, prev_x2 = bboxes[0]
    rows = [[bboxes[0]]]
    for i in range(1, len(bboxes)):
        y1, x1, y2, x2 = bboxes[i]
        if y1 > prev_y2:
            rows.append([bboxes[i]])
        else:
            rows[-1].append(bboxes[i])

        prev_y1, prev_x1, prev_y2, prev_x2 = y1, x1, y2, x2

    for i in range(0, len(rows)):
        rows[i] = sorted(rows[i], key=lambda x: x[1])
    ##########################

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    ##########################
    crop_rows = []
    for row in rows:
        crops = []
        for bbox in row:
            y1, x1, y2, x2 = bbox
            side = max(y2-y1, x2-x1)
            center_y, center_x = (y2+y1) / 2, (x2+x1) / 2
            crop = bw[int(center_y-side/2): int(center_y+side/2), int(center_x-side/2):int(center_x+side/2)] * (-1.)
            crop = skimage.transform.resize(crop, (20, 20))
            crop = np.pad(crop, ((4, 4), (4, 4)), constant_values=crop.max())
            crop = crop.T.flatten()
            crops.append(crop)

        crops = np.array(crops)
        crops = (crops - np.mean(crops, axis=1).reshape(-1, 1)) / np.std(crops, axis=1).reshape(-1, 1)
        crop_rows.append(crops)
    ##########################

    # load the weights
    # run the crops through your neural network and print them out
    letters = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                        'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'd', 'e',
                        'f', 'g', 'h', 'n', 'q', 'r', 't'])

    ##########################
    print(img)

    correct = None
    if img == '02_letters.jpg':
        correct = ["ABCDEFG", "HIJKLMN", "OPQRSTU", "VWXYZ", "1234567890"]
    elif img == '04_deep.jpg':
        correct = ["DEEPLEARNING", "DEEPERLEARNING", "DEEPESTLEARNING"]
    elif img == '01_list.jpg':
        correct = ["TODOLIST", "1MAKEATODOLIST", "2CHECKOFFTHEFIRST", "THINGONTODOLIST", "3REALIZEYOUHVEALREADY",
                   "COMPLETED2THINGS", "4REWARDYOURSELFWITH", "ANAP"]
    elif img == '03_haiku.jpg':
        correct = ["HAIKUSAREEASY", "BUTSOMETIMESTHEYDONTMAKESENSE", "REFRIGERATOR"]

    count = 0
    total = 0
    for idx, crops in enumerate(crop_rows):
        crops = crops.reshape(-1, 1, 28, 28)
        outputs = net(torch.from_numpy(crops).float())
        _, predicted = torch.max(outputs.data, 1)
        output = letters[predicted]
        output = ''.join(output)
        print(output)

        total += len(output)
        if correct is not None:
            for i in range(len(output)):
                if output[i] == correct[idx][i]:
                    count += 1

    print("Accuracy:", count / total)
    print()
