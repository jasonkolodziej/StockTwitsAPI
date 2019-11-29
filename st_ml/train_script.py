import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm, trange
tqdm.pandas(desc='Progress')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from custom_ml import ConcatPoolingGRUAdaptive, SentimentDataset, collate_fn

import ignite
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.contrib.handlers import ProgressBar

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# import warnings
# warnings.filterwarnings('ignore')

# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity='all'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print('Python version:',sys.version)
print('Pandas version:',pd.__version__)
print('Pytorch version:', torch.__version__)
print('Ignite version:', ignite.__version__)

data_root = pathlib.Path('./data')

# load csv in pandas dataframe
df = pd.read_csv(data_root / 'cleaned_tweets.csv', error_bad_lines=False)


# split the data into train and validation
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[['Sentiment']])
train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)

print(train_df.shape, val_df.shape)

# ******************** #
# * Training Dataset * #
# ******************** #
vocab_size = 100000
train_ds = SentimentDataset(train_df, max_vocab_size=vocab_size)
train_ds.save_vocabulary()

# ********************** #
# * Validation dataset * #
# ********************** #

val_ds = SentimentDataset(val_df, word2idx=train_ds.word2idx, idx2word=train_ds.idx2word)

batch_size = 1024

# *************** #
# * Dataloaders * #
# *************** #

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dl = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate_fn)


#####################
# Set up the models #
#####################

# (vocab_size + 2) is because of pad and unk added to the vocab
model_vocab_size = vocab_size + 2
embedding_dim = 100
rnn_hidden = 256
n_out = 2

model = ConcatPoolingGRUAdaptive(model_vocab_size, embedding_dim, rnn_hidden, n_out).to(device) 
optimizer = optim.Adam(model.parameters(), 1e-3)
loss_fn = F.nll_loss

# ****************************************************** #
# *              Ignite training Callbacks             * #
# Ignite is all about callbacks.                         #
# Training and evaluation is defined separately.         #
# You can define your single custom training             #
# and evaluator loop and add them to Engine.             #
# Add loss and accuracy to the trainer and evaluator.    #
# Finally define early stopping and modelcheckpoint      #
# ****************************************************** #
##############################################
# Define single training and validation loop #
##############################################

def process_function(engine, batch):
    """Single training loop to be attached to trainer Engine
    """
    model.train()
    optimizer.zero_grad()
    x, y, lens = batch
    x, y = x.to(device), y.to(device)
    y_pred = model(x, lens)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()
    return loss.item(), torch.max(y_pred, dim=1)[1], y


def eval_function(engine, batch):
    """Single evaluator loop to be attached to trainer and evaluator Engine
    """
    model.eval()
    with torch.no_grad():
        x, y, lens = batch
        x, y = x.to(device), y.to(device)
        y_pred = model(x, lens)
        return y_pred, y
    
trainer = Engine(process_function)
train_evaluator = Engine(eval_function)
validation_evaluator = Engine(eval_function)

################################################################
# Add metrics (Loss and Accuracy) to the trainer and evaluator #
################################################################
def max_output_transform(output):
    """It convers the predicted ouput probabilties to indexes for accuracy calculation
    """
    y_pred, y = output
    return torch.max(y_pred, dim=1)[1], y

# attach running loss (will be displayed in progess bar)
RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'loss')

# attach running accuracy (will be displayed in progess bar)
RunningAverage(Accuracy(output_transform=lambda x: [x[1], x[2]])).attach(trainer, 'acc')

# attach accuracy and loss to train_evaluator
Accuracy(output_transform=max_output_transform).attach(train_evaluator, 'accuracy')
Loss(loss_fn).attach(train_evaluator, 'bce')

# attach accuracy and loss to validation_evaluator
Accuracy(output_transform=max_output_transform).attach(validation_evaluator, 'accuracy')
Loss(loss_fn).attach(validation_evaluator, 'bce')
#############################################
# Report progress through tqdm progress bar #
#############################################
pbar = ProgressBar(persist=True, bar_format="")
pbar.attach(trainer, ['loss', 'acc'])

# Log after each EPOCH
@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
    """This function will run after each epoch and 
       report the training loss and accuracy (defined above)
    """
    train_evaluator.run(train_dl)
    metrics = train_evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_bce = metrics['bce']
    pbar.log_message(
        f'Training Results - Epoch: {engine.state.epoch}  Avg accuracy: {avg_accuracy:.4f} Avg loss: {avg_bce:.4f}')
    
@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(engine):
    """This function will run after each epoch and 
       report the validation loss and accuracy (defined above)
    """
    validation_evaluator.run(val_dl)
    metrics = validation_evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_bce = metrics['bce']
    pbar.log_message(
        f'Validation Results - Epoch: {engine.state.epoch}  Avg accuracy: {avg_accuracy:.4f} Avg loss: {avg_bce:.4f}')
    pbar.n = pbar.last_print_n = 0

###################################################
# Callback for Early stopping and ModelCheckpoint #
###################################################
def score_function(engine):
    """EarlyStopping will call this function to check if score improved
    """
    val_loss = engine.state.metrics['bce']
    return -val_loss


early_stopping = EarlyStopping(patience=3, score_function=score_function, trainer=trainer)
validation_evaluator.add_event_handler(Events.COMPLETED, early_stopping)

checkpointer = ModelCheckpoint(
    './models', 
    'text_gru_concat', 
    save_interval=1, 
    n_saved=1, 
    create_dir=True, 
    save_as_state_dict=True)

trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'sentiment': model})

trainer.run(train_dl, max_epochs=10)
