import csv
from collections import Counter
from string import punctuation
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from lstm_nn import SentimentalLSTM
import torch.nn as nn
from nltk.corpus import stopwords
appos = {
    "aren't" : "are not",
    "can't" : "cannot",
    "couldn't" : "could not",
    "didn't" : "did not",
    "doesn't" : "does not",
    "don't" : "do not",
    "hadn't" : "had not",
    "hasn't" : "has not",
    "haven't" : "have not",
    "he'd" : "he would",
    "he'll" : "he will",
    "he's" : "he is",
    "i'd" : "I would",
    "i'd" : "I had",
    "i'll" : "I will",
    "i'm" : "I am",
    "isn't" : "is not",
    "it's" : "it is",
    "it'll":"it will",
    "i've" : "I have",
    "let's" : "let us",
    "mightn't" : "might not",
    "mustn't" : "must not",
    "shan't" : "shall not",
    "she'd" : "she would",
    "she'll" : "she will",
    "she's" : "she is",
    "shouldn't" : "should not",
    "that's" : "that is",
    "there's" : "there is",
    "they'd" : "they would",
    "they'll" : "they will",
    "they're" : "they are",
    "they've" : "they have",
    "we'd" : "we would",
    "we're" : "we are",
    "weren't" : "were not",
    "we've" : "we have",
    "what'll" : "what will",
    "what're" : "what are",
    "what's" : "what is",
    "what've" : "what have",
    "where's" : "where is",
    "who'd" : "who would",
    "who'll" : "who will",
    "who're" : "who are",
    "who's" : "who is",
    "who've" : "who have",
    "won't" : "will not",
    "wouldn't" : "would not",
    "you'd" : "you would",
    "you'll" : "you will",
    "you're" : "you are",
    "you've" : "you have",
    "'re": " are",
    "wasn't": "was not",
    "we'll":" will",
    "didn't": "did not"
}
stop_words = stopwords.words('english')

def review_formatting(reviews):
    all_reviews=list()
    for text in reviews:
        lower_case = text.lower()
        words = lower_case.split()
        reformed = [appos[word] if word in appos else word for word in words]
        reformed_test=list()
        for word in reformed:
            if word not in stop_words:
                reformed_test.append(word)
        reformed = " ".join(reformed_test) 
        punct_text = "".join([ch for ch in reformed if ch not in punctuation])
        all_reviews.append(punct_text)
    all_text = " ".join(all_reviews)
    all_words = all_text.split()
    return all_reviews, all_words

def encode_reviews(reviews):
    """
    encode_reviews function will encodes review in to array of numbers
    """
    all_reviews=list()
    for text in reviews:
        text = text.lower()
        text = "".join([ch for ch in text if ch not in punctuation])
        all_reviews.append(text)
    encoded_reviews=list()
    for review in all_reviews:
        encoded_review=list()
        for word in review.split():
            if word not in vocab_to_int.keys():
                encoded_review.append(0)
            else:
                encoded_review.append(vocab_to_int[word])
        encoded_reviews.append(encoded_review)
    return encoded_reviews

def pad_sequences(encoded_reviews, sequence_length=250):
    """ 
    Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.
    """
    features=np.zeros((len(encoded_reviews), sequence_length), dtype=int)
    
    for i, review in enumerate(encoded_reviews):
        review_len=len(review)
        if (review_len<=sequence_length):
            zeros=list(np.zeros(sequence_length-review_len))
            new=zeros+review
        else:
            new=review[:sequence_length]
        features[i,:]=np.array(new)
    return features

def preprocess(reviews):
    """
    This Function will tranform reviews in to model readable form
    """
    formated_reviews, all_words = review_formatting(reviews)
    encoded_reviews=encode_reviews(formated_reviews)
    features=pad_sequences(encoded_reviews, 250)
    return features

def test_model(input_test):
    output_list=list()
    batch_size=50   
    net.eval()
    with torch.no_grad():
        test_review=preprocess(input_test)
        for review in test_review:
            # convert to tensor to pass into your model
            feature_tensor = torch.from_numpy(review).view(1,-1)
            if(train_on_gpu):
                feature_tensor= feature_tensor.cuda()
            batch_size = feature_tensor.size(0)
            # initialize hidden state
            h = net.init_hidden(batch_size)
            # get the output from the model
            output, h = net(feature_tensor, h)
            pred = torch.round(output.squeeze()) 
            output_list.append(pred)
        labels=[int(i.data.cpu().numpy()) for i in output_list]
        return labels

if __name__ == "__main__":
    shuffled_ts = 'data_gathered/full_shuffled_ts.csv'
    with open(shuffled_ts, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        # get header from first row
        headers = next(reader)
        # get all the rows as a list
        data = list(reader)
        # transform data into numpy array
        # data = np.array(data)
    tweets = []
    labels = []
    # separate the csv file
    for i in data:
        tweets.append(i[0])
        labels.append(i[1])
    all_reviews, all_words = review_formatting(tweets)
    # all_reviews = list()
    # # Remove Punctuation and get all the words from review dataset
    # for tweet in tweets:
    #     tweet = tweet.lower()
    #     tweet = "".join([ch for ch in tweet if ch not in punctuation])
    #     all_reviews.append(tweet)
    # all_text = " ".join(all_reviews)
    # all_words = all_text.split()
    # Count all the words and sort it based on counts
    count_words = Counter(all_words)
    total_words=len(all_words)
    sorted_words=count_words.most_common(total_words)
    vocab_to_int={w:i+1 for i,(w,c) in enumerate(sorted_words)}
    print("Top ten occuring words : ", sorted_words[:10])
    # Create a dictionary to convert words to Integers based on the number of occurrence of the word
    # vocab_to_int={w:i+1 for i,(w,c) in enumerate(sorted_words)}
    # # print(vocab_to_int)
    # # Encode review in to list of Integer by using above dictionary
    # encoded_reviews=list()
    # for review in all_reviews:
    #     encoded_review=list()
    #     for word in review.split():
    #         if word not in vocab_to_int.keys():
    #             #if word is not available in vocab_to_int put 0 in that place
    #             encoded_review.append(0)
    #         else:
    #             encoded_review.append(vocab_to_int[word])
    #     encoded_reviews.append(encoded_review)
    # #  make all the encoded_review of the same length
    # sequence_length=250
    # features=np.zeros((len(encoded_reviews), sequence_length), dtype=int)
    # for i, review in enumerate(encoded_reviews):
    #     review_len=len(review)
    #     if (review_len<=sequence_length):
    #         zeros=list(np.zeros(sequence_length-review_len))
    #         new=zeros+review
    #     else:
    #         new=review[:sequence_length]
    # features[i,:]=np.array(new)
    # convert labels into ndarray
    labels = np.array(labels, dtype=int)
    # Split this feature data into Training and Validation set
    # split_dataset into 90% training and 10% Validation Dataset
    features=preprocess(tweets)
    train_x=features[:int(0.90*len(features))]
    train_y=labels[:int(0.90*len(features))]
    valid_x=features[int(0.90*len(features)):]
    valid_y=labels[int(0.90*len(features)):]
    print(len(train_y), len(valid_y))
    #create Tensor Dataset
    train_data=TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    valid_data=TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y))
    #dataloader
    batch_size=50
    train_loader=DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader=DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    # obtain one batch of training data
    dataiter = iter(train_loader)
    sample_x, sample_y = dataiter.next()
    print('Sample input size: ', sample_x.size()) # batch_size, seq_length
    print('Sample input: \n', sample_x)
    print()
    print('Sample label size: ', sample_y.size()) # batch_size
    print('Sample label: \n', sample_y)
    # Step13: Initialize the model
    # Instantiate the model w/ hyperparams
    vocab_size = len(vocab_to_int)+1 # +1 for the 0 padding
    print("Vocabulary Size: ", vocab_size)
    output_size = 2
    embedding_dim = 400
    hidden_dim = 256
    n_layers = 2

    net = SentimentalLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
    print(net)
    # Step14: Train the model
    # loss and optimization functions
    lr=0.001
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # training params
    epochs = 3 # 3-4 is approx where I noticed the validation loss stop decreasing
    counter = 0
    print_every = 100
    clip=5 # gradient clipping

    # move model to GPU, if available
    if(net.train_on_gpu):
        net.cuda()

    net.train()
    # train for some number of epochs
    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)
        # batch loop
        for inputs, labels in train_loader:
            counter += 1
            if(net.train_on_gpu):
                inputs=inputs.cuda()
                labels=labels.cuda()
            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            output, h = net(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for inputs, labels in valid_loader:

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])
                    if net.train_on_gpu:
                        inputs, labels = inputs.cuda(), labels.cuda()  
                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output.squeeze(), labels.float())
                    val_losses.append(val_loss.item())

                net.train()
                print("Epoch: {}/{}...".format(e+1, epochs),
                    "Step: {}...".format(counter),
                    "Loss: {:.6f}...".format(loss.item()),
                    "Val Loss: {:.6f}".format(np.mean(val_losses)))