import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import spacy
from torch.utils.data import Dataset, DataLoader
import re
import subprocess as call
import json
import pandas as pd
from collections import Counter
from tqdm.auto import tqdm, trange
tqdm.pandas(desc='Progress')

# ******************************************* #
# * Make batches through PyTorch Dataloader * #
# ******************************************* #

def collate_fn(data):
    """This function will be used to pad the tweets to max length
       in the batch and transpose the batch from 
       batch_size x max_seq_len to max_seq_len x batch_size.
       It will return padded vectors, labels and lengths of each tweets (before padding)
       It will be used in the Dataloader
    """
    data.sort(key=lambda x: len(x[0]), reverse=True)
    lens = [len(sent) for sent, label in data]
    labels = []
    padded_sents = torch.zeros(len(data), max(lens)).long()
    for i, (sent, label) in enumerate(data):
        padded_sents[i,:lens[i]] = torch.LongTensor(sent)
        labels.append(label)
    
    padded_sents = padded_sents.transpose(0,1)
    return padded_sents, torch.tensor(labels).long(), lens

# ************************ #
# * Custom class for data  #
# ************************ #
PAD = 0
UNK = 1

print('Pytorch version:', torch.__version__)
print('Spacy version:', spacy.__version__)

class SentimentDataset(Dataset):
    """Define the pytorch Dataset to process the tweets
       This class can be used for both training and validation dataset
       Run it for training data and pass the word2idx and idx2word when running
       for validation data
    """
    
    def __init__(self, df=None, word2idx=None, idx2word=None, max_vocab_size=50000):
        print('Processing Data')
        self.df = df
        if(self.df is not None):
            print('Removing white space...')
            self.df.SentimentText = self.df.SentimentText.progress_apply(lambda x: x.strip())
        try:
            self.nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])
        except IOError:
            call.check_output("python -m spacy download en", shell=True)
            self.nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])
        finally:
            pass
        if word2idx is None and df is not None:
            print('Building Counter...')
            word_counter = self.build_counter()
            print('Building Vocab...')
            self.word2idx, self.idx2word = self.build_vocab(word_counter, max_vocab_size)
        else:
            self.word2idx, self.idx2word = word2idx, idx2word
        print('*'*100)
        print('Dataset info:')
        if df is not None:
            print(f'Number of Tweets: {self.df.shape[0]}')
            print(f'Vocab Size: {len(self.word2idx)}')
        print('*'*100)
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        sent = self.df.SentimentText[idx]
        tokens = [w.text.lower() for w in self.nlp(self.tweet_clean(sent))]
        vec = self.vectorize(tokens, self.word2idx)
        return vec, self.df.Sentiment[idx]
    
    def tweet_clean(self, text):
        """Very basic text cleaning. This function can be built upon for
           better preprocessing
        """
        text = re.sub(r'[\s]+', ' ', text) # replace multiple white spaces with single space
#         text = re.sub(r'@[A-Za-z0-9]+', ' ', text) # remove @ mentions
        text = re.sub(r'https?:/\/\S+', ' ', text) # remove links
        text = re.sub(r'[^A-Za-z0-9]+', ' ', text) # remove non alphanumeric character
        return text.strip()
    
    def build_counter(self):
        """Tokenize the tweets using spacy and build vocabulary
        """
        words_counter = Counter()
        for sent in tqdm(self.df.SentimentText.values):
            words_counter.update(w.text.lower() for w in self.nlp(self.tweet_clean(sent)))
        return words_counter
    
    def build_vocab(self, words_counter, max_vocab_size):
        """Add pad and unk tokens and build word2idx and idx2word dictionaries
        """
        word2idx = {'<PAD>': PAD, '<UNK>': UNK}
        word2idx.update({word:i+2 for i, (word, count) in tqdm(enumerate(words_counter.most_common(max_vocab_size)))})
        idx2word = {idx: word for word, idx in tqdm(word2idx.items())}
        return word2idx, idx2word
    
    def vectorize(self, tokens, word2idx):
        """Convert tweet to vector
        """
        vec = [word2idx.get(token, UNK) for token in tokens]
        return vec
    
    def save_vocabulary(self, PATH='./models/sd', filenames=('w2i','i2w')):
        """ Save the word2idx & idx2word dicts
        """
        f = open(PATH+filenames[0]+".json","w")
        f.write(json.dumps(self.word2idx))
        f.close()
        print("wrote word2idx to {}".format(PATH+filenames[0]+".json"))
        f = open(PATH+filenames[1]+".json","w")
        f.write(json.dumps(self.idx2word))
        f.close()
        print("wrote word2idx to {}".format(PATH+filenames[1]+".json"))
        return

    def load_vocabulary(self, PATH='./models/sd', filenames=('w2i','i2w')):
        f = open(PATH+filenames[0]+".json","r")
        self.word2idx = json.load(f)
        f.close()
        f = open(PATH+filenames[1]+".json","r")
        self.idx2word = json.load(f)
        f.close()
        print('!'*100)
        print(f'Updated Vocab Sizes (word2idx, idx2word): {len(self.word2idx)}, {len(self.idx2word)}')
        print('!'*100)
        return
    
    def load_json_as_df(self, json_obj):
        assert 'SentimentText' in json.loads(json_obj), "key `SentimentText` should appear in hash [dict] argument"
        assert 'Sentiment' in json.loads(json_obj), "key `Sentiment` should appear in hash [dict] argument"
        self.df = pd.read_json(json_obj)
        self.df.SentimentText.progress_apply(lambda x: x.strip())
        print('!'*100)
        print('Notice Updated Dataset info:')
        print(f'Number of %Tweets% / data entries: {self.df.shape[0]}')
        print('!'*100)
        return
    @staticmethod
    def get_pdl(ds, collator=collate_fn):
        """Get the Pytorch DataLoader for a SentimentDataset
            used for predictions
        """
        return DataLoader(ds, collate_fn=collator)


# *********************************** #
# * Max Pooling and Average Pooling * #
#   with Concat pooling GRU model     #
# *********************************** #

resolve_model_output = lambda a : 0 if a[0] == 0 and a[1] != 0 else (1 if a[0] != 0 and a[1] == 0 else None)
class ConcatPoolingGRUAdaptive(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_hidden, n_out):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.emb = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.emb_drop = nn.Dropout(0.3)
        self.gru = nn.GRU(self.embedding_dim, self.n_hidden, dropout=0.3)
        self.out = nn.Linear(self.n_hidden*3, self.n_out)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, seq, lengths):
        self.h = self.init_hidden(seq.size(1))
        embs = self.emb_drop(self.emb(seq))
        embs = pack_padded_sequence(embs, lengths)
        gru_out, self.h = self.gru(embs, self.h)
        gru_out, lengths = pad_packed_sequence(gru_out)        
        
        avg_pool = F.adaptive_avg_pool1d(gru_out.permute(1,2,0),1).view(seq.size(1),-1)
        max_pool = F.adaptive_max_pool1d(gru_out.permute(1,2,0),1).view(seq.size(1),-1)

        outp = self.out(torch.cat([self.h[-1],avg_pool,max_pool],dim=1))             
        return F.log_softmax(outp, dim=-1) # it will return log of softmax
    
    def init_hidden(self, batch_size):
        return torch.zeros((1, batch_size,self.n_hidden), requires_grad=True).to(self.device)

    def import_dict(self, PATH='./models/text_gru_concat_sentiment_10.pth'):
        """# Load the saved state into the model
        """
        self.load_state_dict(torch.load(PATH))

    @staticmethod
    def prediction(model, dl):
        """ static method used on `ConcatPoolingGRUAdaptive` class to
            predict sentiment
        """
        sentiment = []
        model.eval() # run if you only want to use it for inference
        with torch.no_grad():
            for x, y, lens in dl:
                x = x.to(model.device) # , y.to(model.device)
                y_pred = model(x, lens)
                pred = torch.round(y_pred.squeeze()).tolist()
                print(pred)
                sentiment.append(resolve_model_output(pred))
            return sentiment
