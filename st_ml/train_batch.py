import torch
import torch.nn as nn
from torch import optim
import time, random
import os
from tqdm import tqdm
from lstm import LSTMSentiment
from bilstm import BiLSTMSentiment
from torchtext import data
import numpy as np
import argparse

torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)
progression = lambda itr, epoch : tqdm(itr, desc='Train epoch '+str(epoch+1), unit='iter.', dynamic_ncols=True)
load_file_prog = lambda rnge, desc, units : tqdm(range(rnge), desc=desc, unit=units, dynamic_ncols=True)
eval_prog = lambda itr, dsc : tqdm(itr, desc=dsc, unit='iter.', dynamic_ncols=True)


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in load_file_prog(vocab_size, 'Loading Neg. Vectors', 'lines'):
            word = []
            while True:
                ch = f.read(1).decode('latin-1')
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
               word_vecs[word] = np.frombuffer(f.read(binary_len), dtype='float32') # np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs


def get_accuracy(truth, pred):
    assert len(truth) == len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i] == pred[i]:
            right += 1.0
    return right / len(truth)


def train_epoch_progress(model, train_iter, loss_function, optimizer, text_field, label_field, epoch, device):
    model.train()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    count = 0
    for batch in progression(train_iter, epoch):
        # Get data
        # if device:
        sent, label = batch.text.to(device), batch.label.to(device)
        # else:
            # sent, label = batch.text, batch.label
        optimizer.zero_grad() # zero gradient for forward pass
        # Apply forward pass
        label.data.sub_(1)
        truth_res += list(label.data)
        model.batch_size = len(label.data)
        model.hidden = model.init_hidden()
        pred = model(sent)
        # if device:
        pred_label = pred.data.max(1)[1].numpy() if device == "cpu" else pred.data.max(1)[1].detach().cpu().numpy()
        # else:
        #     pred_label = pred.data.max(1)[1].numpy()
        pred_res += [x for x in pred_label]
        # model.zero_grad()
        loss = loss_function(pred, label)
        avg_loss += loss.data # [0]
        count += 1

        # backward pass and optimizer step change
        loss.backward()
        optimizer.step()
    avg_loss /= len(train_iter)
    acc = get_accuracy(truth_res, pred_res)
    return avg_loss, acc


# def train_epoch(model, train_iter, loss_function, optimizer):
#     model.train()
#     avg_loss = 0.0
#     truth_res = []
#     pred_res = []
#     count = 0
#     for batch in train_iter:
#         sent, label = batch.text, batch.label
#         label.data.sub_(1)
#         truth_res += list(label.data)
#         model.batch_size = len(label.data)
#         model.hidden = model.init_hidden()
#         pred = model(sent)
#         pred_label = pred.data.max(1)[1].numpy()
#         pred_res += [x for x in pred_label]
#         model.zero_grad()
#         loss = loss_function(pred, label)
#         avg_loss += loss.data # [0]
#         count += 1
#         loss.backward()
#         optimizer.step()
#     avg_loss /= len(train_iter)
#     acc = get_accuracy(truth_res, pred_res)
#     return avg_loss, acc


def evaluate(model, data, loss_function, name, device):
    model.eval()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    for batch in eval_prog(data, 'Evaluating'):
        # if device:
        sent, label = batch.text.to(device), batch.label.to(device)
        # else:
        #     sent, label = batch.text, batch.label
        label.data.sub_(1)
        truth_res += list(label.data)
        model.batch_size = len(label.data)
        model.hidden = model.init_hidden()
        pred = model(sent)
        # if device:
        pred_label = pred.data.max(1)[1].numpy() if device == "cpu" else pred.data.max(1)[1].detach().cpu().numpy()
        # else:
        #     pred_label = pred.data.max(1)[1].numpy()
        pred_res += [x for x in pred_label]
        loss = loss_function(pred, label)
        avg_loss += loss.data # [0]
    avg_loss /= len(data)
    acc = get_accuracy(truth_res, pred_res)
    print(name + ': loss %.2f acc %.1f' % (avg_loss, acc*100))
    return acc


def load_sst(text_field, label_field, batch_size, device):
    # train, dev, test = data.TabularDataset.splits(path='./data/SST2/', train='train.tsv',
    #                                               validation='dev.tsv', test='test.tsv', format='tsv',
    #                                               fields=[('text', text_field), ('label', label_field)])
    train, dev, test = data.TabularDataset.splits(path='./training_data/', train='train.csv',
                                                  validation='valid.csv', test='test.csv', format='csv',
                                                  skip_header=True,
                                                  fields=[('text', text_field), ('label', label_field)])
    # Construct the Vocab object for nesting field and combine it with this field’s vocab.
    text_field.build_vocab(train, dev, test)
    label_field.build_vocab(train, dev, test)
    # Defines an iterator that batches examples of similar lengths together.
    # && Create Iterator objects for multiple splits of a dataset.
    train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test),
                batch_sizes=(batch_size, batch_size, batch_size), sort_key=lambda x: len(x.text), repeat=False, device=device)
    ## for GPU run
#     train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test),
#                 batch_sizes=(batch_size, len(dev), len(test)), sort_key=lambda x: len(x.text), repeat=False, device=None)
    return train_iter, dev_iter, test_iter


# def adjust_learning_rate(learning_rate, optimizer, epoch):
#     lr = learning_rate * (0.1 ** (epoch // 10))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     return optimizer


args = argparse.ArgumentParser()
args.add_argument('--m', dest='model', default='lstm', help='specify the mode to use (default: lstm)')
args = args.parse_args()

EPOCHS = 20
USE_GPU = torch.cuda.is_available()
DEVICE = "cuda:0" if USE_GPU else "cpu" # None
print("Using CUDA GPUs ==> ",USE_GPU, DEVICE)
EMBEDDING_DIM = 300
HIDDEN_DIM = 150

BATCH_SIZE = 50
timestamp = str(int(time.time()))
best_dev_acc = 0.0

print("Loading data for training, validating, and testing purposes...")
text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)
train_iter, dev_iter, test_iter = load_sst(text_field, label_field, BATCH_SIZE, DEVICE)
print("text_field size : ", len(text_field.vocab), "\tlabel_field size : ", len(label_field.vocab)-1)
print()


if not args.model or args.model == 'lstm':
    model = LSTMSentiment(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, vocab_size=len(text_field.vocab), label_size=len(label_field.vocab)-1,\
                          use_gpu=USE_GPU, batch_size=BATCH_SIZE)
    print('model is lstm')

if args.model == 'bilstm':
    model = BiLSTMSentiment(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, vocab_size=len(text_field.vocab), label_size=len(label_field.vocab)-1,\
                          use_gpu=USE_GPU, batch_size=BATCH_SIZE)
    print('model is bilstm')

if USE_GPU:
    model = model.cuda()


print('Load word embeddings...')
# # glove
# text_field.vocab.load_vectors('glove.6B.100d')

# word2vector
word_to_idx = text_field.vocab.stoi
pretrained_embeddings = np.random.uniform(-0.25, 0.25, (len(text_field.vocab), 300))
pretrained_embeddings[0] = 0
# load Google News vector weights
word2vec = load_bin_vec('./data/GoogleNews-vectors-negative300.bin', word_to_idx)
for word, vector in word2vec.items():
    pretrained_embeddings[word_to_idx[word]-1] = vector

# text_field.vocab.load_vectors(wv_type='', wv_dim=300)
model.embeddings.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
# model.embeddings.weight.data = text_field.vocab.vectors
# model.embeddings.embed.weight.requires_grad = False


best_model = model
optimizer = optim.SGD(model.parameters(), lr=1e-5)
loss_function = nn.NLLLoss()

print('Training...')
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
print("Writing to {}\n".format(out_dir))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
for epoch in range(EPOCHS):
    avg_loss, acc = train_epoch_progress(model, train_iter, loss_function, optimizer, text_field, label_field, epoch, DEVICE)
    tqdm.write('Train: loss %.2f acc %.1f' % (avg_loss, acc*100))
    dev_acc = evaluate(model, dev_iter, loss_function, 'Dev', DEVICE)
    if dev_acc > best_dev_acc:
        if best_dev_acc > 0:
            os.system('rm '+ out_dir + '/best_model' + '.pth')
        best_dev_acc = dev_acc
        best_model = model
        torch.save(best_model.state_dict(), out_dir + '/best_model' + '.pth')
        # evaluate on test with the best dev performance model
        test_acc = evaluate(best_model, test_iter, loss_function, 'Test', DEVICE)
test_acc = evaluate(best_model, test_iter, loss_function, 'Final Test', DEVICE)

