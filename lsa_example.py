#! /usr/bin/env python3
# -*- encoding: utf-8 -*-

"""
recommended: https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#sphx-glr-auto-examples-text-plot-document-classification-20newsgroups-py

also recommended but watch out for errors (I've corrected them here) !!
https://towardsdatascience.com/latent-semantic-analysis-intuition-math-implementation-a194aff870f8
"""

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns # for plotting
import matplotlib.pyplot as plt # for plotting
import pandas as pd
import numpy as np
import re
import string


np.random.seed(500)

X_train, y_train = fetch_20newsgroups(subset='train', return_X_y=True)
X_test, y_test = fetch_20newsgroups(subset='test', return_X_y=True)

tokenizer = RegexpTokenizer(r'\b\w{3,}\b')
stop_words = list(set(stopwords.words("english")))
stop_words += list(string.punctuation)
stop_words += ['__', '___']

def rmv_emails_websites(string):
    """Function removes emails, websites and numbers"""
    new_str = re.sub(r"\S+@\S+", '', string)
    new_str = re.sub(r"\S+.co\S+", '', new_str)
    new_str = re.sub(r"\S+.ed\S+", '', new_str)
    new_str = re.sub(r"[0-9]+", '', new_str)
    return new_str

X_train = list(map(rmv_emails_websites, X_train))
X_test  = list(map(rmv_emails_websites, X_test))

# take raw counts (e.g., CountVectorizer) or "better" counts
# check out working with text data: https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#extracting-features-from-text-files

tfidf = TfidfVectorizer(lowercase=True,
                        stop_words=stop_words,
                        tokenizer=tokenizer.tokenize,
                        max_df=0.2,
                        min_df=0.02
                       )
tfidf_train_sparse = tfidf.fit_transform(X_train)

# convert to dataframe for ease of visualization
tfidf_train_df = pd.DataFrame(tfidf_train_sparse.toarray(),
                        columns=tfidf.get_feature_names_out())
tfidf_train_df.head()
#print(tfidf_train_df.head())

# following https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html#sklearn.decomposition.TruncatedSVD

lsa = TruncatedSVD(n_components=20, n_iter=100, random_state=42) # it's actually called truncated
tfidf_lsa_data = lsa.fit_transform(tfidf_train_df)
# attributes of the lsa_obj
U = lsa.explained_variance_
Sigma = lsa.singular_values_
# V = lsa.components_
V_T = lsa.components_.T

#print(Sigma)

# plot 1: how much each concept represents the data
#sns.barplot(x=list(range(len(Sigma))), y = Sigma)
#plt.show()
#plt.savefig('plot1.png')

# plot 2: V_T
# first put into dataframe
n_topics = 20 #n_components
term_topic_matrix = pd.DataFrame(data=V_T,
                index=tfidf.get_feature_names_out(),
                columns=[f'Latent_concept_{r}' for r in range(n_topics)])

print(term_topic_matrix.head())
#
# data = term_topic_matrix[f'Latent_concept_1']
# data = data.sort_values(ascending=False)
# top_10 = data[:10]
# plt.title('Top terms along the axis of Latent concept 1')
# sns.barplot(x= top_10.values, y=top_10.index)
# plt.show()
# plt.savefig('plot1.png')
