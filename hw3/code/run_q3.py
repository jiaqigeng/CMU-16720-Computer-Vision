import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt


train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

train_x = (train_x - train_x.mean(axis=1)[:, None]) / train_x.std(axis=1)[:, None]
valid_x = (valid_x - valid_x.mean(axis=1)[:, None]) / valid_x.std(axis=1)[:, None]
test_x = (test_x - test_x.mean(axis=1)[:, None]) / test_x.std(axis=1)[:, None]


max_iters = 50
# pick a batch size, learning rate
batch_size = None
learning_rate = None
hidden_size = 64
##########################
batch_size = 16
# change the learning rate here to generate all 6 plots
# 1e-2, 1e-3, 1e-4
learning_rate = 1e-3
##########################

train_batches = get_random_batches(train_x, train_y, batch_size)
batch_num = len(train_batches)

params = {}

# initialize layers here
##########################
initialize_weights(train_x.shape[1], hidden_size, params, 'layer1')
initialize_weights(hidden_size, train_y.shape[1], params, 'output')

'''
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

fig = plt.figure(1, (8., 8.))
if hidden_size < 128:
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(8, 8),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    img_w = params['Wlayer1'].reshape((32, 32, hidden_size))
    for i in range(hidden_size):
        grid[i].imshow(img_w[:, :, i])  # The AxesGrid object work as a list of axes.

    plt.show()
'''
##########################

train_acc_history = []
train_loss_history = []
val_acc_history = []
val_loss_history = []

# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    for phase in ['train', 'val']:
        total_loss = 0
        avg_acc = 0

        if phase == 'train':
            batches = train_batches
            data_size = train_x.shape[0]
        else:
            batches = [(valid_x, valid_y)]
            data_size = valid_x.shape[0]

        for xb, yb in batches:
            # training loop can be exactly the same as q2!
            ##########################
            h1 = forward(xb, params, 'layer1')
            probs = forward(h1, params, 'output', softmax)
            loss, acc = compute_loss_and_acc(yb, probs)
            total_loss += loss / data_size
            avg_acc += acc * xb.shape[0] / data_size

            if phase == 'train':
                delta1 = probs
                delta1[np.arange(probs.shape[0]), np.argmax(yb, axis=1)] -= 1
                delta2 = backwards(delta1, params, 'output', linear_deriv)
                backwards(delta2, params, 'layer1', sigmoid_deriv)

                params['Wlayer1'] -= learning_rate * params['grad_Wlayer1']
                params['blayer1'] -= learning_rate * params['grad_blayer1']
                params['Woutput'] -= learning_rate * params['grad_Woutput']
                params['boutput'] -= learning_rate * params['grad_boutput']

        if phase == 'train':
            if itr % 2 == 0:
                print()
                print("Train:")
            train_acc_history.append(avg_acc)
            train_loss_history.append(total_loss)
        else:
            if itr % 2 == 0:
                print("Val:")
            val_acc_history.append(avg_acc)
            val_loss_history.append(total_loss)

        ##########################
        if itr % 2 == 0:
            print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr, total_loss, avg_acc))

print()
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

print()
test_acc = None
##########################
h1 = forward(test_x, params, 'layer1')
probs = forward(h1, params, 'output', softmax)
test_loss, test_acc = compute_loss_and_acc(test_y, probs)
##########################
print('Test accuracy: ', test_acc)

# run on validation set and report accuracy! should be above 75%
valid_acc = None
##########################
h1 = forward(valid_x, params, 'layer1')
probs = forward(h1, params, 'output', softmax)
valid_loss, valid_acc = compute_loss_and_acc(valid_y, probs)
##########################
print('Validation accuracy: ', valid_acc)


if False:  # view the data
    for crop in xb:
        import matplotlib.pyplot as plt

        plt.imshow(crop.reshape(32, 32).T)
        plt.show()

# uncomment this part if you want to save q3_weights
# might affect later problems (shoulnd't be too much difference)
'''
import pickle

saved_params = {k: v for k, v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
'''

# Q3.1.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

fig = plt.figure(1, (8., 8.))
if hidden_size < 128:
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(8, 8),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    img_w = params['Wlayer1'].reshape((32, 32, hidden_size))
    for i in range(hidden_size):
        grid[i].imshow(img_w[:, :, i])  # The AxesGrid object work as a list of axes.

    plt.show()

# Q3.1.4

fig = plt.figure(1, (6., 8.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(12, 6),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

indices = params['cache_output'][2].argmax(axis=0)
images = valid_x[indices]
images = images.reshape(36, 32, 32)

vis = np.zeros((36, 1024))
inps = np.eye(36)
for i, inp in enumerate(inps):
    vis[i] = inp @ params['Woutput'].T @ params['Wlayer1'].T
vis = vis.reshape(36, 32, 32)

displayed = np.zeros((72, 32, 32))
displayed[::2] = images
displayed[1::2] = vis
for ax, im in zip(grid, displayed):
    ax.imshow(im.T)
plt.savefig("out.jpg")
plt.show()


# Q3.1.5
confusion_matrix = np.zeros((test_y.shape[1], test_y.shape[1]))

# compute comfusion matrix here
##########################
h1 = forward(test_x, params, 'layer1')
probs = forward(h1, params, 'output', softmax)
pred = np.argmax(probs, axis=1)
actual = np.argmax(test_y, axis=1)
for i in range(pred.shape[0]):
    confusion_matrix[pred[i], actual[i]] += 1
print(confusion_matrix)
##########################

import string

plt.imshow(confusion_matrix, interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36), string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36), string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()
