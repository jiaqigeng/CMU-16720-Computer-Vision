import numpy as np
import scipy.io
from nn import *
from collections import Counter
from util import *


train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']


max_iters = 100
# pick a batch size, learning rate
batch_size = 36
learning_rate = 3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x, np.ones((train_x.shape[0], 1)), batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
##########################
initialize_weights(1024, hidden_size, params, 'layer1')
initialize_weights(hidden_size, hidden_size, params, 'layer2')
initialize_weights(hidden_size, hidden_size, params, 'layer3')
initialize_weights(hidden_size, 1024, params, 'output')

params['m_layer1'] = np.zeros((1024, hidden_size))
params['m_layer2'] = np.zeros((hidden_size, hidden_size))
params['m_layer3'] = np.zeros((hidden_size, hidden_size))
params['m_output'] = np.zeros((hidden_size, 1024))
##########################

train_loss_history = []

# should look like your previous training loops
for itr in range(max_iters):
    total_loss = 0
    for xb, _ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

        ##########################
        h1 = forward(xb, params, 'layer1', relu)
        h2 = forward(h1, params, 'layer2', relu)
        h3 = forward(h2, params, 'layer3', relu)
        output = forward(h3, params, 'output', sigmoid)
        loss = np.sum((xb - output) ** 2)
        total_loss += loss / train_x.shape[0]

        delta1 = 2 * (output - xb)
        delta2 = backwards(delta1, params, 'output', sigmoid_deriv)
        delta3 = backwards(delta2, params, 'layer3', relu_deriv)
        delta4 = backwards(delta3, params, 'layer2', relu_deriv)
        backwards(delta3, params, 'layer1', relu_deriv)

        params['m_layer1'] = 0.9 * params['m_layer1'] - learning_rate * params['grad_Wlayer1']
        params['m_layer2'] = 0.9 * params['m_layer2'] - learning_rate * params['grad_Wlayer2']
        params['m_layer3'] = 0.9 * params['m_layer3'] - learning_rate * params['grad_Wlayer3']
        params['m_output'] = 0.9 * params['m_output'] - learning_rate * params['grad_Woutput']

        params['Wlayer1'] += params['m_layer1']
        params['blayer1'] -= learning_rate * params['grad_blayer1']
        params['Wlayer2'] += params['m_layer2']
        params['blayer2'] -= learning_rate * params['grad_blayer2']
        params['Wlayer3'] += params['m_layer3']
        params['blayer3'] -= learning_rate * params['grad_blayer3']
        params['Woutput'] += params['m_output']
        params['boutput'] -= learning_rate * params['grad_boutput']

        '''
        params['Wlayer1'] -= learning_rate * params['grad_Wlayer1']
        params['blayer1'] -= learning_rate * params['grad_blayer1']
        params['Wlayer2'] -= learning_rate * params['grad_Wlayer2']
        params['blayer2'] -= learning_rate * params['grad_blayer2']
        params['Wlayer3'] -= learning_rate * params['grad_Wlayer3']
        params['blayer3'] -= learning_rate * params['grad_blayer3']
        params['Woutput'] -= learning_rate * params['grad_Woutput']
        params['boutput'] -= learning_rate * params['grad_boutput']
        '''
        ##########################

    train_loss_history.append(total_loss)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr, total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9
        
# Q5.3.1
import matplotlib.pyplot as plt
# visualize some results
##########################
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(train_loss_history, label='Training Loss')
plt.legend()
plt.show()

valid_y = valid_data['valid_labels']
selected = {}
for i in range(5):
    selected[i] = (np.where([valid_y[:, i] == 1])[1][:2])

for label in selected:
    for i in range(2):
        h1 = forward(valid_x[selected[label][i]], params, 'layer1', relu)
        h2 = forward(h1, params, 'layer2', relu)
        h3 = forward(h2, params, 'layer3', relu)
        output = forward(h3, params, 'output', sigmoid)

        plt.imshow(valid_x[selected[label][i]].reshape(32, 32).T)
        plt.show()
        plt.imshow(output.reshape(32, 32).T)
        plt.show()
##########################


# Q5.3.2
from skimage.measure import compare_psnr as psnr
# evaluate PSNR
##########################
##### your code here #####
##########################
