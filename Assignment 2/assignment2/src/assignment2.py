#! /usr/bin/env python3
# -*- encoding: utf-8 -*-

"""
In this assignment you will experiment with sentiment classification.

You will be working with datasets from the 2014 SemEval shared task ``Sentiment analysis in Twitter''.
If you are interested in background information or want to compare your own system to systems which participated
in the competition, please have a look at:

S. Rosenthal, A. Ritter, P. Nakov and V. Stoyanov. SemEval-2014 Task 9: Sentiment Analysis in Twitter. SemEval 2014.
"""


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score,KFold

def load_data(file):
    '''Loads data into pandas dataframes.
  
  Args: 
    The file or file_path of the data that are needed to be read.
  
  Returns:
	    A pandas dataframe with three categories the ID of the tweets, their sentiment if known and the tweet itself.
	'''
    return pd.read_csv(file,delimiter='\t',header=None,names=["ID", "Sentiment","Tweet"])

def construct_label_vector(labels):
    ''' Constructs vectors of the labels.
    
    Args : 
        column of pandas dataframes with the normalised hashable labels.

    Returns :
         Numerical labels.
    '''
    label_encoder = LabelEncoder()
    return label_encoder,label_encoder.fit_transform(labels)

def construct_feature_vectors(train_data,dev_data,test_data):
    ''' Creates a matrix of TF-IDF features for the training,development and testing. 

    Args : 
        train_data : list or array object with training text data.
        dev_data : list or array object with development text data.
        test_data : list or array object with test text data. 

    Returns : 
            Three vectorized matrixes of TF-IDF features one for the training data, one 
            for the development and one for testing. Development and testing matrixes are based on the
            training features. 

    '''

    data_vectorizer = TfidfVectorizer() #surprisingly not having stop words removed gave a higher accuracy so ultimately I decided to keep them
    tfidf_matrix_train = data_vectorizer.fit_transform(train_data)
    tfidf_matrix_dev = data_vectorizer.transform(dev_data)
    tfidf_matrix_test = data_vectorizer.transform(test_data)
    #feature_names = data_vectorizer.get_feature_names_out()

    return tfidf_matrix_train,tfidf_matrix_dev,tfidf_matrix_test

def build_and_train_model(train_data,train_labels):
    ''' Trains a logistic regression model with balanced weight and iterations of 120.
    
    Args:
        train_data : a feature matrix of train data.
        train_labels : a vector of numerical labels.

    Returns :
            A trained model of logistic regression equipped to deal mutliclass classification.
    '''
    model = LogisticRegression(class_weight='balanced',max_iter=120) #saw a big difference when added the class weight,  
    return model.fit(train_data,train_labels)                #the precision and recall of a class got 3 time better
                                                                     #I figured there is some imbalance in the data,
                                                                     #added more iteration as I was getting a warning

def evaluate_model(model,dev_matrix,dev_gold_labels):
    '''  Predicts the development classes and evaluates the model.

    Args:
        model : machine learning model.
        dev_matrix : feature matrix of the development data.
        dev_gold_labels : the true labels of the development data.

    Returns:
        Accuracy score, precision score, recall, f1 score (weighted)
    '''
    predict_dev = model.predict(dev_matrix)
    accuracy = accuracy_score(dev_gold_labels,predict_dev)
    precision = precision_score(dev_gold_labels,predict_dev,average='weighted')
    recall = recall_score(dev_gold_labels,predict_dev,average='weighted')
    F1_score = f1_score(dev_gold_labels,predict_dev,average='weighted')
    return accuracy,precision,recall,F1_score
    
def cross_validation(train_data,train_labels,model,num_folds,scoring='accuracy'):
    ''' Performs k-fold cross-validation on the model and training data.

        Args :
            train_data : a feature matrix of train data.
            train_labels : a vector of numerical labels.
            model : machine learning model.
            num_folds : Number of folds for cross-validation.
            scoring : scoring metric for evaluation. Accuracy is the default.
        
        Returns :
            An array of cross-validation scores.
    '''
    kf = KFold(n_splits=num_folds,shuffle=True,random_state=42) 
    scores = cross_val_score(model,train_data,train_labels,cv=kf,scoring=scoring)
    return scores

def predict_test(model,test_matrix):
    ''' Predicts the labels of the test data.

        Args:
            model : machine learning model.
            test_data: a feature matrix of test data.

        Returns : 
            An array of vectorized encoded test labels.
    '''
    predict_test_labels = model.predict(test_matrix)
    return predict_test_labels


def main():
    #filepaths of my computer so I can load my data
    filepath_train = r'C:\Users\User\MLT\Machine learning\Assignment 2\assignment2\data\training.txt'
    filepath_dev_input = r'C:\Users\User\MLT\Machine learning\Assignment 2\assignment2\data\development.input.txt'
    filepath_dev_golf = r'C:\Users\User\MLT\Machine learning\Assignment 2\assignment2\data\development.gold.txt'
    filepath_test_input = r'C:\Users\User\MLT\Machine learning\Assignment 2\assignment2\data\test.labels.logistic.regression.txt'
    filepath_test_gold = r'C:\Users\User\MLT\Machine learning\Assignment 2\assignment2\data\test.gold.txt'

    #loading data
    twitter_train = load_data(filepath_train)
    twitter_dev_input = load_data(filepath_dev_input)
    twitter_dev_gold = load_data(filepath_dev_golf)
    twitter_test_input = load_data(filepath_test_input)
    twitter_test_gold = load_data(filepath_test_gold)

    #getting the classes/labels
    labels_train = twitter_train.loc[:,'Sentiment']
    labels_dev = twitter_dev_gold.loc[:,'Sentiment']
    labels_test = twitter_test_gold.loc[:,'Sentiment']

    #vectorizing labels
    encoder,vectorized_labels_train = construct_label_vector(labels_train) #2-positve,0-negative,1-neutral
    encoder,vectorized_labels_dev = construct_label_vector(labels_dev)
    encoder,vectorized_labels_test = construct_label_vector(labels_test)

    #getting the tweets
    tweets_train = twitter_train.loc[:,'Tweet']
    tweets_dev = twitter_dev_input.loc[:,'Tweet']
    tweets_test = twitter_test_input.loc[:,'Tweet']

    #Shuffling my data so the model doesn't learn based on order
    twitter_train_shuffled = twitter_train.sample(frac=1,random_state=42).reset_index(drop=True)
    labels_train_shuffled = twitter_train_shuffled.loc[:,'Sentiment']
    encoder,vectorized_labels_train_shuffled = construct_label_vector(labels_train_shuffled)
    tweets_train_shuffled = twitter_train_shuffled.loc[:,'Tweet']

    #construction the features
    matrix__train,matrix_dev,matrix_test = construct_feature_vectors(tweets_train_shuffled,tweets_dev,tweets_test)
    #trying dimensionality reduction 
    #I ended up not using it as the evaluation metrics went down
    svd = TruncatedSVD(n_components=1000)
    matrix__train_reduced = svd.fit_transform(matrix__train)
    matrix_dev_reduced = svd.transform(matrix_dev)
    #training and evaluation
    trained_model = build_and_train_model(matrix__train,vectorized_labels_train_shuffled)
    accuracy,precision,recall,F1_score = evaluate_model(trained_model,matrix_dev,vectorized_labels_dev)
    print(f'The accuracy of the model is {accuracy}')
    print(f'The precision of the model is {precision}')
    print(f'The recall of the model is {recall}')
    print(f'The f1 score of the model is {F1_score}')

    accuracy_test,precision_test,recall_test,F1_score_test = evaluate_model(trained_model,matrix_test,vectorized_labels_test)
    print(f'The accuracy of the model in test is {accuracy_test}')
    print(f'The precision of the model is {precision_test}')
    print(f'The recall of the model is {recall_test}')
    print(f'The f1 score of the model is {F1_score_test}')

    #cross-validation bonus:
    scores =cross_validation(matrix__train,vectorized_labels_train_shuffled,trained_model,num_folds=5)
    print(scores)
    
    #test_labels
    encoded_test_labels = predict_test(trained_model,matrix_test)
    test_labels= encoder.inverse_transform(encoded_test_labels)
    #changing the labels in the file
    twitter_test_input['Sentiment'] = test_labels   
    twitter_test_input.to_csv(filepath_test_input, sep='\t', index=False, header=False)


if __name__ == "__main__":
    main()


