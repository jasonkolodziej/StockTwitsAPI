# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# import torchvision.datasets as dsets
# train_dataset = dsets.MNIST(root='./data', 
#                             train=True, 
#                             transform=transforms.ToTensor(),
#                             download=True)

# test_dataset = dsets.MNIST(root='./data', 
#                            train=False, 
#                            transform=transforms.ToTensor())
# train_dataset.
# print(train_dataset.train_data.size())
# print(train_dataset.targets.size())
# print(train_dataset.targets)
# exit(8)

###############################################################################
#######################  1. LOAD THE TRAINING TEXT  ###########################
###############################################################################
import csv
shuffled_ts = 'data/full_shuffled_ts.csv'
with open(shuffled_ts, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    # get header from first row
    # headers = next(reader)
    # get all the rows as a list
    data = list(reader)
################################################################################
###################### 1a. Convert the CSV into import #########################
################################################################################
all_reviews = list()
all_labels = list()
for d in data:
    all_reviews.append(d[0])
    all_labels.append(d[1])

LABEL_MAPPING = {"0":0,"2":1,"4":2}
all_labels = [LABEL_MAPPING[label] for label in all_labels]

###############################################################################
##########################  2. TEXT PRE-PROCESSING  ###########################
###############################################################################
import re
import nltk
import numpy as np
nltk.download('punkt')
# remove all symbols from each review
symbol_remover = lambda X : re.sub(r'[^\w]', ' ', X)
all_reviews = list(map(symbol_remover, all_reviews))


###############################################################################
##################  3. CREATE DICTIONARIES & ENCODE REVIEWS  ##################
###############################################################################
from collections import Counter
words = Counter()  # Dictionary that will map a word to the number of times it appeared in all the training sentences
for i, sentence in enumerate(all_reviews):
    # The sentences will be stored as a list of words/tokens
    all_reviews[i] = []
    for word in nltk.word_tokenize(sentence):  # Tokenizing the words
        words.update([word.lower()])  # Converting all the words to lowercase
        all_reviews[i].append(word)
# Removing the words that only appear once
words = {k:v for k,v in words.items() if v>1}
# Sorting the words according to the number of appearances, with the most common word being first
words = sorted(words, key=words.get, reverse=True)
# Adding padding and unknown to our vocabulary so that they will be assigned an index
words = ['_PAD','_UNK'] + words
# Dictionaries to store the word to index mappings and vice versa
word2idx = {o:i for i,o in enumerate(words)}
idx2word = {i:o for i,o in enumerate(words)}
print("Top ten occuring words : ", words[:10])
# for item in vocab_to_int.items():
#     print(item)
# encoded_reviews = [[vocab_to_int[word] for word in review.split()] for review in all_reviews]
for i, sentence in enumerate(all_reviews):
    # Looking up the mapping dictionary and assigning the index to the respective words
    all_reviews[i] = [word2idx[word] if word in word2idx else 0 for word in sentence]

encoded_reviews = all_reviews
encoded_labels = all_labels


print("length of reviews {} length of labels {}".format(len(encoded_reviews), len(encoded_labels)))
###############################################################################
#####################  5. GET RID OF LENGTH-0 REVIEWS   #######################
###############################################################################
import numpy as np
import torch

def pad_text(encoded_reviews, seq_length):
    features = np.zeros((len(encoded_reviews), seq_length), dtype=int)
    for ii, review in enumerate(encoded_reviews):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_length]
    return features

assert len(encoded_reviews) == len(encoded_labels), "There should be the same amount of encoded reviews as labels!"
padded_reviews = pad_text(encoded_reviews, 256)
encoded_labels = np.array(all_labels)
# encoded_labels.resize((encoded_labels.shape[0],1,))
padded_reviews.resize((padded_reviews.shape[0],16,16))
print("Output of the first padded review\n {}".format(padded_reviews[0]))
print("Output of the first encoded labels\n {}".format(encoded_labels[0]))
###############################################################################
##############  7. SPLIT DATA & GET (REVIEW, LABEL) DATALOADER  ###############
###############################################################################
train_ratio = 0.8
valid_ratio = (1 - train_ratio)/2
total = padded_reviews.shape[0]
print("Total [padded reviews shape] {}".format(total))
print("reviews shape {} <==> labels shape {}".format(padded_reviews.shape, encoded_labels.shape))
train_cutoff = int(total * train_ratio)
valid_cutoff = int(total * (1 - valid_ratio))

train_x, train_y = padded_reviews[:train_cutoff], encoded_labels[:train_cutoff]
valid_x, valid_y = padded_reviews[train_cutoff : valid_cutoff], encoded_labels[train_cutoff : valid_cutoff]
test_x, test_y = padded_reviews[valid_cutoff:], encoded_labels[valid_cutoff:]
assert len(train_x) == len(train_y), "lengths of tensors must be the same!"

import torch
from torch.utils.data import TensorDataset, DataLoader
batch_size = 100
n_iters = 130275/2  
# create Tensor Datasets, after importing the numpy arrays into a Tensor `int64`
train_data = TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).long())
valid_data = TensorDataset(torch.from_numpy(valid_x).float(), torch.from_numpy(valid_y).long())
test_data = TensorDataset(torch.from_numpy(test_x).float(), torch.from_numpy(test_y).long())
print("training: reviews tensor size {} <==> labels tensor size {}".format(train_data.tensors[0].size(), train_data.tensors[1].size()))
# assert len(train_x) == len(train_y), "lengths of tensors must be the same!"

train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
valid_loader = DataLoader(valid_data, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = True)

num_epochs = n_iters / (len(train_data) / batch_size)
num_epochs = int(num_epochs)
print("Number of EPOCHs set to {}".format(num_epochs), len(train_data))

###############################################################################
#########################  8. DEFINE THE LSTM MODEL  ##########################
###############################################################################
from torch import nn

'''
STEP 3: CREATE MODEL CLASS
'''

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)

        # One time step
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out

'''
STEP 4: INSTANTIATE MODEL CLASS
'''
input_dim = 16
hidden_dim = 100
layer_dim = 3  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER
output_dim = 3
valid_loss_min = np.Inf
model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)

#######################
#  USE GPU FOR MODEL  #
#######################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

'''
STEP 5: INSTANTIATE LOSS CLASS
'''
criterion = nn.CrossEntropyLoss()

'''
STEP 6: INSTANTIATE OPTIMIZER CLASS
'''
learning_rate = 0.1

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

'''
STEP 7: TRAIN THE MODEL
'''
for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())


# Number of steps to unroll
seq_dim = 16 

itter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Load images as Variable
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        images = images.view(-1, seq_dim, input_dim).requires_grad_().to(device)
        labels = labels.to(device)

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        # outputs.size() --> 100, 10
        outputs = model(images)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        itter += 1

        if itter % 500 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                #######################
                #  USE GPU FOR MODEL  #
                #######################
                images = images.view(-1, seq_dim, input_dim).to(device)
                labels = labels.to(device)

                # Forward pass only to get logits/output
                outputs = model(images)

                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)

                # Total number of labels
                total += labels.size(0)

                # Total correct predictions
                #######################
                #  USE GPU FOR MODEL  #
                #######################
                if torch.cuda.is_available():
                    correct += (predicted.cpu() == labels.cpu()).sum()
                else:
                    correct += (predicted == labels).sum()

            accuracy = 100.00 * correct / total

            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(itter, loss.item(), accuracy))