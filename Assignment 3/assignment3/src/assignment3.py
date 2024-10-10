# -*- coding: utf-8 -*-
"""
   Assignment 3: Sentiment Classification on a Feed-Forward Neural Network using Pretrained Embeddings
   Original code by Hande Celikkanat & Miikka Silfverberg.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gensim
import os
from data_semeval import *
from paths import data_dir, model_dir
from nltk.corpus import stopwords
#nltk.download("stopwords")
stopwords_set = set(stopwords.words("english"))
from matplotlib import pyplot as plt



# GoogleNews-vectors-negative300.bin (from https://code.google.com/archive/p/word2vec/)
embeddings_file = 'GoogleNews-pruned2tweets.bin'


n_classes = len(LABEL_INDICES)
n_epochs = 30 #changing this didn't help either
learning_rate = 0.001 #getting this higher actually made it worse
report_every = 1
verbose = False


# To convert string label to pytorch format:
def label_to_idx(label):
    return torch.LongTensor([LABEL_INDICES[label]])


class FFNN(nn.Module):
    
    def __init__(self, pretrained_embeds, n_classes):
        super(FFNN, self).__init__()
        hidden_neurons = 300
        (num_embeddings,embeddings_dim) = pretrained_embeds.shape
        self.hidden_layer = nn.Linear(embeddings_dim,hidden_neurons)
        self.hidden_layer2 = nn.Linear(hidden_neurons,hidden_neurons)
        self.output_layer = nn.Linear(hidden_neurons,n_classes)
        self.activation = F.sigmoid
        self.softmax = F.log_softmax


    def forward(self, x):
        tweet_embeddings = embeddings(x)
        hidden_output = self.hidden_layer(tweet_embeddings)
        hidden_output = self.activation(hidden_output)

        hidden_output2 = self.hidden_layer2(hidden_output)
        hidden_output2 = self.activation(hidden_output2)

        output = self.output_layer(hidden_output2)
        output = self.softmax(output,dim=0) 
        
        return output
    
#this didn't help not surprising
#def remove_stopwords(tweet):
 #   no_stopwords = []
  #  for word in  tweet['BODY']:
   #     if word.lower() not in stopwords_set:
    #        no_stopwords.append(word)
    #return no_stopwords

def embeddings(tweet):
    '''
    Converts a tweet into an embedding vector representing the tweet.

    Converts the tweet by first collecting all the word embedding of the tokens
    stacking them together and then summing them.

    Args :
        tweet: A dictionary representing a tweet having it's ID, SENTIMENT AND BODY
    
    Returns : 
        torch.Tensor : A tensor representing the embedding of the enitre tweet.
    '''
    embeddings = []
    indices_tweet = []
    tweet_embeddings = []
    #tokenized = remove_stopwords(tweet)
    #for token in tokenized:
    for token in tweet['BODY']:
        if token in word_to_idx:
            index = word_to_idx[token]
            indices_tweet.append(index)
    for index in indices_tweet:
        embedding= torch.tensor(pretrained_embeds[index])
        embeddings.append(embedding)
    stacked = torch.stack(embeddings,dim = 0)
    tweet_embeddings = torch.sum(stacked,dim=0)
    return tweet_embeddings

#I followed a tutorial to make this work but I don't really see the results(the plots are empty)
#maybe something is wrong with anaconda on my laptop
#there might be something off that I don't understand as I am pretty new at plotting.

def plot_most_info_embed(embeddings,labels,n_classes,top_n):

    '''
    Plots the most informative tweet embeddings for each class.

    Args:
        embeddings(list of torch.Tensor) : A list containing embedding vectors for each tweet.
        labels : A list containing the labels for each tweet.
        n_classes : The number of classes.
        top_n : The number of top informative embeddings to display for each class.
    '''

    fig,axs = plt.subplots(n_classes,figsize= (10,6*n_classes))

    for class_index in range(n_classes):
        class_embeddings = []
        for embedding,label in zip(embeddings,labels):
            if label == class_index:
                class_embeddings.append(embedding)

        importance = []
        for embedding in class_embeddings:
            importance.append(torch.norm(embedding))

        sorted_indices = sorted(range(len(importance)), key=lambda i: importance[i],reverse=True)

        top_embeddings = []
        for i in sorted_indices[:top_n]:
            top_embeddings.append(class_embeddings[i])

        axs[class_index].scatter([embedding[0] for embedding,label in zip(top_embeddings,labels) if label == class_index],
                                 [embedding[1] for embedding,label in zip(top_embeddings,labels) if label == class_index],
                                 label = f"Class {class_index}")
        axs[class_index].legend()
        axs[class_index].set_title(f"Top {top_n} Most informative for class {class_index}")
        axs[class_index].set_xlabel("Dimension 1")
        axs[class_index].set_ylabel("Dimension 2")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    
    # --- data loading ---
    data = read_semeval_datasets(data_dir)
    gensim_embeds = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(model_dir, embeddings_file),
                                                                    binary=True)
    pretrained_embeds = gensim_embeds.vectors
    
    # To convert words in the input tweet to indices of the embeddings matrix:
    word_to_idx = {word: i for i, word in enumerate(gensim_embeds.index_to_key)}
    
    # --- set up ---
    model = FFNN(pretrained_embeds,n_classes)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(),lr=learning_rate,weight_decay=0.002)
    
    # --- training --- 
    for epoch in range(n_epochs):
        total_loss = 0
        for tweet in  data['training']:
            gold_class = torch.LongTensor([label_to_idx(tweet['SENTIMENT'])])
            optimizer.zero_grad()
            outputs = model(tweet)
            loss = loss_function(outputs.unsqueeze(0),gold_class) 
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()

        if ((epoch + 1) % report_every) == 0:
            print(f"epoch: {epoch}, loss: {round(total_loss * 100 / len(data['training']), 4)}")

    # --- test ---
    predicted_labels_dev = []
    correct = 0
    with torch.no_grad():
        for tweet in data['dev.gold']:
            gold_class = label_to_idx(tweet['SENTIMENT'])
            outputs = model(tweet)
            _,predicted = torch.max(outputs.unsqueeze(0),1)

            correct += torch.eq(predicted, gold_class).item()

            predicted_labels_dev.append(predicted.item())
            
            if verbose:
                print('DEV DATA: %s, OUTPUT: %s, GOLD LABEL: %d' %
                      (tweet['BODY'], tweet['SENTIMENT'], predicted))

        print(f"dev accuracy: {round(100 * correct / len(data['dev.gold']), 2)}")

    predicted_labels_test = []
    correct = 0
    with torch.no_grad():
        for tweet in data['test.gold']:
            gold_class = label_to_idx(tweet['SENTIMENT'])
            outputs = model(tweet)
            _,predicted = torch.max(outputs.unsqueeze(0),1)

            correct += torch.eq(predicted, gold_class).item()

            predicted_labels_test.append(predicted.item())
            
            if verbose:
                print('DEV DATA: %s, OUTPUT: %s, GOLD LABEL: %d' %
                      (tweet['BODY'], tweet['SENTIMENT'], predicted))

        print(f"test accuracy: {round(100 * correct / len(data['test.gold']), 2)}")

    idx_to_label = {}
    for label,idx in LABEL_INDICES.items():
        idx_to_label[idx] = label

    predicted_labels_words = []
    for prediction in predicted_labels_test:
        sentiment = idx_to_label[prediction]
        predicted_labels_words.append(sentiment)
    

    with open(r'C:\Users\User\MLT\Machine learning\Assignment 3\assignment3\data\test.input.txt', "w", encoding="utf-8") as file: 
        write_semeval(data['test.input'],predicted_labels_words,file) #I am having trouble using relative paths so I sticked to the absolute


    tweet_embeddings = []
    for tweet in data['test.gold']:
        tweet_embeddings.append(embeddings(tweet))
    
    tweet_labels = []
    for tweet in data['test.gold']:
        tweet_labels.append(tweet['SENTIMENT'])

    plot_most_info_embed(tweet_embeddings,tweet_labels,n_classes,top_n=10)