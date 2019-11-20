###############################################################################
#######################  1. LOAD THE TRAINING TEXT  ###########################
###############################################################################
import csv
shuffled_ts = 'data/cleaned_shuf_ts.csv'
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

all_labels = [1 if int(label) == 4 else 0 for label in all_labels]

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
padded_reviews = pad_text(encoded_reviews, 200)
encoded_labels = np.array(all_labels)
print("Output of the first padded review\n {}".format(padded_reviews[0]))

###############################################################################
##############  7. SPLIT DATA & GET (REVIEW, LABEL) DATALOADER  ###############
###############################################################################
train_ratio = 0.8
valid_ratio = (1 - train_ratio)/2
total = padded_reviews.shape[0]
print("Total [padded reviews shape] {}".format(total))
train_cutoff = int(total * train_ratio)
valid_cutoff = int(total * (1 - valid_ratio))

train_x, train_y = padded_reviews[:train_cutoff], encoded_labels[:train_cutoff]
valid_x, valid_y = padded_reviews[train_cutoff : valid_cutoff], encoded_labels[train_cutoff : valid_cutoff]
test_x, test_y = padded_reviews[valid_cutoff:], encoded_labels[valid_cutoff:]
assert len(train_x) == len(train_y), "lengths of tensors must be the same!"

import torch
from torch.utils.data import TensorDataset, DataLoader
# create Tensor Datasets, after importing the numpy arrays into a Tensor `int64`
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

batch_size = 50
train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
valid_loader = DataLoader(valid_data, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = True)

###############################################################################
#########################  8. DEFINE THE LSTM MODEL  ##########################
###############################################################################
from torch import nn

class SentimentNNet(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        super(SentimentNNet, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)
        
        out = out.view(batch_size, -1)
        out = out[:,-1]
        return out, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden


###############################################################################
################  9. INSTANTIATE THE MODEL W/ HYPERPARAMETERS #################
###############################################################################
vocab_size = len(word2idx) + 1
output_size = 1
embedding_dim = 400
hidden_dim = 512
n_layers = 2

model = SentimentNNet(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)


###############################################################################
#######################  10. DEFINE LOSS & OPTIMIZER  #########################
###############################################################################
from torch import optim
import tqdm

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.005)

step_loader = lambda x,y,z='iterations' : tqdm.tqdm(iter(x), total=y, unit=z)

###############################################################################
##########################  11. TRAIN THE NETWORK!  ###########################
############################################################################### 
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
epochs = 2
counter = 0
print_every = 1000
clip = 5 # for gradient clip to prevent exploding gradient problem in LSTM/RNN
valid_loss_min = np.Inf
model.to(device)
model.train()
for i in range(epochs):
    h = model.init_hidden(batch_size)
    
    for inputs, labels in step_loader(train_loader, len(train_x)):
        counter += 1
        h = tuple([e.data for e in h])
        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()
        output, h = model(inputs, h)
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        if counter%print_every == 0:
            val_h = model.init_hidden(batch_size)
            val_losses = []
            model.eval()
            for inp, lab in step_loader(valid_loader, len(valid_x)):
                val_h = tuple([each.data for each in val_h])
                inp, lab = inp.to(device), lab.to(device)
                out, val_h = model(inp, val_h)
                val_loss = criterion(out.squeeze(), lab.float())
                val_losses.append(val_loss.item())
                
            model.train()
            print("Epoch: {}/{}...".format(i+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))
            if np.mean(val_losses) <= valid_loss_min:
                torch.save(model.state_dict(), './state_dict.pt')
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(val_losses)))
                valid_loss_min = np.mean(val_losses)

###############################################################################
################  12. TEST THE TRAINED MODEL ON THE TEST SET  #################
###############################################################################
# Loading the best model
model.load_state_dict(torch.load('./state_dict.pt'))

test_losses = []
num_correct = 0
h = model.init_hidden(batch_size)

model.eval()
for inputs, labels in test_loader:
    h = tuple([each.data for each in h])
    inputs, labels = inputs.to(device), labels.to(device)
    output, h = model(inputs, h)
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())
    pred = torch.round(output.squeeze())  # Rounds the output to 0/1
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)

print("Test loss: {:.3f}".format(np.mean(test_losses)))
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}%".format(test_acc*100))


###############################################################################
############  13. TEST THE TRAINED MODEL ON A RANDOM SINGLE REVIEW ############
###############################################################################
def predict(net, review, seq_length = 200):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    words = preprocess(review)
    encoded_words = [vocab_to_int[word] for word in words]
    padded_words = pad_text([encoded_words], seq_length)
    padded_words = torch.from_numpy(padded_words).to(device)
    
    if(len(padded_words) == 0):
        "Your review must contain at least 1 word!"
        return None
    
    net.eval()
    h = net.init_hidden(1)
    output, h = net(padded_words, h)
    pred = torch.round(output.squeeze())
    msg = "This is a positive review." if pred == 0 else "This is a negative review."
    
    return msg


review1 = "It made me cry."
review2 = "It was so good it made me cry."
review3 = "It's ok."
review4 = "This movie had the best acting and the dialogue was so good. I loved it."
review5 = "Garbage"
                       ### OUTPUT ###
predict(net, review1)  ## negative ##
predict(net, review2)  ## positive ##
predict(net, review3)  ## negative ##
predict(net, review4)  ## positive ##
predict(net, review5)  ## negative ##
    