import nltk
from nltk.corpus import brown 
import random
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np
import gensim.downloader as api
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from collections import Counter


embeddings_model = api.load("glove-wiki-gigaword-100")

n_epochs = 20
learning_rate = 0.001 
report_every = 1


def embeddings(data) : 
    '''
    Creates trigram embeddings from a glove pre-trained model of embeddings.

    Args:
      data : a list of tuples containing word and tag.

    Returns:
       - A list of concatenated trigram embeddings for the data.
       - A nested list of tags of each sentence from the data.
    '''
    sentence_words = []
    sentence_tags = []
    for sentence in data:
        words = []
        tags = []
        for word in sentence:
            words.append(word[0])
            tags.append(word[1])
        sentence_words.append(words)
        sentence_tags.append(tags)

    trigrams_sentence = []
    for sentence in sentence_words:
        trigrams = []
        for i in range(len(sentence)):
            n_2 = sentence[max(i -2,0)] if i>=2 else "<START>"
            n_1 = sentence[max(i -1, 0)] if i>=1 else "<START>"

            word = sentence[i]

            trigram = (n_2,n_1,word)
            trigrams.append(trigram)
        trigrams_sentence.append(trigrams)
          
    unk_embedding = torch.randn(embedding_dim) #I couldn't figure out how to deal with unk tokens without deleting them
    #I would loved to create embeddings from scratch but i don't know how to run the code in the server and it would take super long in my laptop
    #if you could help with that to know for future work

    trigram_embeddings = []
    for sentence in trigrams_sentence:
        for trigram in sentence:
            trigram_embed = []
            for word in trigram:
                if word.lower() in embeddings_model:
                    word_embedding = torch.tensor(embeddings_model[word.lower()])
                else:
                    word_embedding = unk_embedding
                trigram_embed.append(word_embedding)
            
            trigram_embed = torch.cat(trigram_embed,dim=0)
            trigram_embeddings.append(trigram_embed)

    return trigram_embeddings, sentence_tags

class FFNN(nn.Module):

    def __init__(self, input_size, hidden_size1,hidden_size2,output_size):
        super(FFNN,self).__init__()
        self.hidden_layer = nn.Linear(input_size,hidden_size1)
        self.hidden_layer2 = nn.Linear(hidden_size1,hidden_size2)
        self.output_layer = nn.Linear(hidden_size2,output_size)
        self.activation = f.relu
        self.softmax = f.log_softmax
    
    def forward(self,x):
        hidden_output = self.hidden_layer(x)
        hidden_output = self.activation(hidden_output)

        hidden_output2 = self.hidden_layer2(hidden_output)
        hidden_output2 = self.activation(hidden_output2)

        output = self.output_layer(hidden_output2)
        output = self.softmax(output,dim=0)

        return output


if __name__ == '__main__':
    #print(brown.categories())
    tag_counts = Counter()
    words = 0
    for sentence in brown.tagged_sents(tagset='universal'):
        for _, tag in sentence:
            tag_counts[tag]+=1
            words +=1
    most_common_tag = tag_counts.most_common(1)
    
    baseline = 275558/words
    print(f'Baseline: {baseline}')

    corpus = list(brown.tagged_sents(tagset='universal'))
    random.shuffle(corpus)
    pos_tags = []
    for sentence in corpus:
        for _,tag in sentence :
            pos_tags.append(tag)
    
    unique_pos_tags = set(pos_tags)
    pos_to_idx = {tag : idx for idx, tag in enumerate(unique_pos_tags)}
    size_corpus = len(corpus)
    size_train = int(0.8*size_corpus)
    size_dev = int(0.1 * size_corpus)
    size_test = int(0.1* size_corpus)
    train_corpus = corpus[:size_train]
    dev_corpus = corpus[size_train:size_train+size_dev]
    test_corpus = corpus[size_train+size_dev:]
    
    n_classes = len(pos_to_idx)
    embedding_dim = embeddings_model.vector_size
    input_size = embedding_dim * 3
    hidden_size1 = 200
    hidden_size2 = 100
    model = FFNN(input_size,hidden_size1,hidden_size2,n_classes)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters())

    trigram_embeddings,sentence_tags = embeddings(train_corpus)

    for epoch in range(n_epochs):
        total_loss = 0
        for embedding,sentence in zip(trigram_embeddings,sentence_tags):
            for tag in sentence:
                gold_class = torch.tensor([pos_to_idx[tag]])
            
                optimizer.zero_grad()
                output = model(embedding)
                loss = loss_function(output.unsqueeze(0),gold_class)
                loss.backward()
                optimizer.step()
                total_loss+=loss.item()

        if ((epoch + 1) % report_every) == 0:
            print(f"epoch: {epoch}, loss: {round(total_loss * 100 / words, 4)}") #should this be len of train corpus??

    trigram_embeddings,sentence_tags = embeddings(dev_corpus)
    predicted_labels_dev = []
    correct = 0
    tags= []
    for sentence in sentence_tags:
        for tag in sentence:
            tags.append([pos_to_idx[tag]])
    gold_class_array = np.array(tags)

    with torch.no_grad():
        for embedding,sentence in zip(trigram_embeddings,sentence_tags):
            for tag in sentence:
                gold_class = torch.tensor([pos_to_idx[tag]])
                output = model(embedding)
                _,predicted = torch.max(output.unsqueeze(0),1)
                correct+=int(torch.eq(predicted,gold_class).item())
                predicted_labels_dev.append(predicted.item())
    predicted_labels_dev_array = np.array(predicted_labels_dev)
    accuracy_dev = accuracy_score(gold_class_array,predicted_labels_dev_array)
    precision_dev = precision_score(gold_class_array,predicted_labels_dev_array,average='macro')
    recall_dev = recall_score(gold_class_array,predicted_labels_dev_array,average = 'macro')
    f1_dev = f1_score(gold_class_array,predicted_labels_dev_array,average='macro')

    print(f'Dev Accuracy : {accuracy_dev}')
    print(f'Dev Precision : {precision_dev}')
    print(f'Dev Recall : {recall_dev}')
    print(f'Dev F1 : {f1_dev}')

    trigram_embeddings,sentence_tags = embeddings(test_corpus)
    predicted_labels_test = []
    correct = 0
    tags= []
    for sentence in sentence_tags:
        for tag in sentence:
            tags.append([pos_to_idx[tag]])
    gold_class_array = np.array(tags)

    with torch.no_grad():
        for embedding,sentence in zip(trigram_embeddings,sentence_tags):
            for tag in sentence:
                gold_class = torch.tensor([pos_to_idx[tag]])
                output = model(embedding)
                _,predicted = torch.max(output.unsqueeze(0),1)
                correct+=int(torch.eq(predicted,gold_class).item())
                predicted_labels_test.append(predicted.item())
    predicted_labels_test_array = np.array(predicted_labels_test)
    accuracy_test = accuracy_score(gold_class_array,predicted_labels_test_array)
    precision_test = precision_score(gold_class_array,predicted_labels_test_array,average='macro')
    recall_test = recall_score(gold_class_array,predicted_labels_test_array,average = 'macro')
    f1_test = f1_score(gold_class_array,predicted_labels_test_array,average='macro')

    print(f'Test Accuracy : {accuracy_test}')
    print(f'Test Precision : {precision_test}')
    print(f'Test Recall : {recall_test}')
    print(f'Test F1 : {f1_test}')


    #The model is performing higher than the baseline but in general I would say that the model is performing quite poorly with very low scores,
    #and tuning hyperparameters did't really help.