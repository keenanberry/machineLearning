#!/usr/bin/env python3
from __future__ import print_function
import sys, os, re
import scipy
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import string
import csv
import statistics
import random
from operator import itemgetter
from collections import defaultdict
# scikit learn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# keras imports
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional, Activation, Flatten

###############################################################################

"""IrishNLP.py: This program implements and tests the performance of various 
   machine learning algorithms in modeling the use of diacritics in the Irish 
   language.
   The models used in this script include Unigram (baseline), Logistic Regression, 
   SVM, and LSTM.
"""

__author__  = "Keenan Berry"
__date__    = "05/06/2019"

###############################################################################

# Set directory and file names
DIR = './kaggleS19/'
TRAIN_FILE = 'train.txt'
TEST_FILE = 'test.txt'

# Regular expression operations
RE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
RE_PUNCT = str.maketrans('', '', string.punctuation)
RE_DIGITS = str.maketrans('', '', string.digits)

# Vocabulary dictionary for number of lines in train.txt.
VOCAB_TRAIN = {'a':24200,'ais':24200,'aisti':2890,'ait':24200,'ar':24200,
               'arsa':24200,'ban':7269,'cead':24200,'chas':24200,'chuig':24200,
               'dar':24200,'do':24200,'gaire':6068,'i':24200,'inar':24200,
               'leacht':3396,'leas':24200,'mo':24200,'na':24200,'os':24200,
               're':12105,'scor':11497,'te':16563,'teann':5049,'thoir':4534}

# Vocabulary dictionary for accented complements.
VOCAB_ACCENT = {'a':'á','ais':'áis','aisti':'aistí','ait':'áit','ar':'ár',
                'arsa':'ársa','ban':'bán','cead':'céad','chas':'chás',
                'chuig':'chúig','dar':'dár','do':'dó','gaire':'gáire','i':'í',
                'inar':'inár','leacht':'léacht','leas':'léas','mo':'mó','na':'ná',
                'os':'ós','re':'ré','scor':'scór','te':'té','teann':'téann',
                'thoir':'thóir'}

###############################################################################
# PROCESSING
# The following functions are used to load and process the data

def load_train(filepath):
    text = list()

    with open(filepath, 'r') as f:
        for line in f:
            line = line.rstrip("\n")
            text.append(line)
    # remove digits, punctuation, and change text to lower case
    stripped = [re.sub('\W+',' ', l).translate(RE_DIGITS).lower().strip() for l in text]
    return stripped


def load_test(filepath):
    text = list()
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.rstrip("\n")
            text.append(line)
    # note: test text is not processed in this function
    return text


# create dictionary of training sentences for each word
def get_sentence_dict(train):
    training_dict = defaultdict(list)
    for word, accent in VOCAB_ACCENT.items():
        for sentence in train:
            if word in sentence.split() or accent in sentence.split():
                training_dict[word].append(sentence)
    return training_dict


# function to return full training data set (including duplicates)
def get_train_data(train, word):
    training_sentences = list()
    training_labels = list()
    duplicates = list()
    
    for sentence in train:
        if word in sentence.split() and VOCAB_ACCENT[word] in sentence.split():
            duplicates.append(sentence)
            new = re.sub(r'\b{}\b'.format(word), '$t@r', sentence)
            training_sentences.append(new)
            training_labels.append(1)
    
        elif word in sentence.split():
            new = re.sub(r'\b{}\b'.format(word), '$t@r', sentence)
            training_sentences.append(new)
            training_labels.append(1)
        
        elif VOCAB_ACCENT[word] in sentence.split():
            new = re.sub(r'\b{}\b'.format(VOCAB_ACCENT[word]), '$t@r', sentence)
            training_sentences.append(new)
            training_labels.append(0)
    
    for dup in duplicates:
        new_dup = re.sub(r'\b{}\b'.format(VOCAB_ACCENT[word]), '$t@r', sentence)
        training_sentences.append(new_dup)
        training_labels.append(0)
    
    zipper = list(zip(training_sentences, training_labels))
    random.shuffle(zipper)
    
    train_text, train_y = zip(*zipper)
    
    return train_text, train_y


# process test text
def get_test_data(test, word):
    testing_sentences = list()
    indices = list()
    
    # find test sentences for 'word'
    for i in range(len(test)):
        new = ""
        match ='{%s|%s}' % (word, VOCAB_ACCENT[word]) 
        if match in test[i]:
            # split sentence
            left, right = test[i].split('{}'.format(match))
            # process split sentence
            new = new + re.sub('\W+',' ', left).translate(RE_DIGITS).lower().strip() + ' $t@r ' + re.sub('\W+',' ', right).translate(RE_DIGITS).lower().strip()
            # append the newly generated sentence to list of test sentences
            testing_sentences.append(new)
            indices.append(i+1)
    
    # handle sentences that have more than one occurence of 'word'/accent
    # replace occurences with unique identifier symbol '$t@r'
    # this is really only a concern for a/á pair
    for i in range(len(testing_sentences)):
        new = re.sub(r'\b{}\b'.format(word), '$t@r', testing_sentences[i])
        new = re.sub(r'\b{}\b'.format(VOCAB_ACCENT[word]), '$t@r', new)
        testing_sentences[i] = new
            
    zipper = list(zip(testing_sentences, indices))
    random.shuffle(zipper)
    
    test_text, test_indices = zip(*zipper)
    
    return test_text, test_indices


# function for the 'baseline' probability classifier model (Unigram)
def get_test_base(test, word):
    indices = list()
    new_text = list()
    for i in range(len(test)):
        if '{%s|%s}' % (word, VOCAB_ACCENT[word]) in test[i]:
            indices.append(i+1)
            new_text.append(re.sub('\W+',' ', test[i]).translate(RE_DIGITS).lower().strip())
    return new_text, indices


# function to find the number of words above a specified frequency threshold
def filter_low_freq(train, word, freq):
    cutoff = len(train) * freq
    word_freq = dict()
    
    for sentence in train:
        for item in sentence.split():
            word_freq[item] = word_freq.get(item, 0) + 1
    
    low_words = 0
    for k, v in word_freq.items():
        if word_freq[k] < cutoff:
            low_words += 1
    
    num_words = len(word_freq) - low_words
    
    return num_words


###############################################################################
# MODELS

# Simple probability classifier (Unigram)
def probability_classifier(train, test, word):
    
    test_text, test_ids = get_test_base(test, word)  # process data before feeding to model
    
    count = 0
    accent_count = 0
    
    for sentence in train:
        for w in sentence.split():
            if w == word:
                count += 1
            if w == VOCAB_ACCENT[word]:
                accent_count += 1
    
    total = count + accent_count
    proba = count / total
    
    prob_list = [proba] * len(test_ids)
    
    return prob_list, test_ids


# logistic regression classifier
def logistic_regression(train, test, word):
    train_text, train_y = get_train_data(train, word)
    test_text, test_ids = get_test_data(test, word)
    
    # vectorizer
    tfidf = TfidfVectorizer(binary=True) 
    
    # get train and test vectorized inputs
    X = tfidf.fit_transform(train_text)
    X_test = tfidf.transform(test_text)
    
    model = LogisticRegression().fit(X,train_y)
    proba = model.predict_proba(X_test)[:,1]
    proba = proba.tolist()
    
    # return id and probability estimates
    return proba, test_ids


# used for hyperparameter tuning
def svc_param_selection(x, y, nfolds):
    Cs = [0.01, 0.1, 1, 10]
    gammas = ['auto', 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(x, y)
    
    return grid_search.best_params_  # returns dictionary


def SVM(train, test, word):
    train_text, train_y = get_train_data(train, word)
    test_text, test_ids = get_test_data(test, word)
    
    tfidf = TfidfVectorizer(binary=True) 
    
    # get train and test vectorized inputs
    X = tfidf.fit_transform(train_text)
    X_test = tfidf.transform(test_text)
    
    # find best parameters
    print('Finding best hyperparameters for "%s" SVM model...' % word)
    best_params_dict = svc_param_selection(X, train_y, nfolds=5)
    print('best params:', best_params_dict)

    # fit best model and make predictions
    print('Evaluating SVM model for "%s"...' % word)
    model = svm.SVC(kernel='rbf', C=best_params_dict['C'], gamma=best_params_dict['gamma'], probability=True)
    model.fit(X, train_y)
    proba = model.predict_proba(X_test)[:,1]
    proba = proba.tolist()
    
    return proba, test_ids


# function to reduce data dimensions
# never used by the program
def reduce_dim(X):
    pca = PCA()
    pca.fit(X)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d = np.argmax(cumsum >= 0.99) + 1
    return d


def lstm(train, test, word):
    train_text, train_y = get_train_data(train, word)
    test_text, test_ids = get_test_data(test, word)
    
    print('\nNow learning the use of "%s" in the Irish Gaelic language' % word)
    
    num_words = 6000  # arbitrary assignment
    tokenizer = Tokenizer(num_words=num_words, filters='!"#%&()*+,-./:;<=>?[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(train_text)
    X_train = tokenizer.texts_to_sequences(train_text)
    X_train = pad_sequences(X_train)
    
    train_samples = X_train.shape[0]
    seq_length = X_train.shape[1]
    #d = reduce_dim(X_train)
    
    X_test = tokenizer.texts_to_sequences(test_text)
    X_test = pad_sequences(X_test, maxlen=seq_length)
    test_samples = X_test.shape[0]

    y_train = to_categorical(train_y, 2)
    
    # reshape data for non-embedding LSTM
    #X_train = X_train.reshape(train_samples, seq_length, 1)
    #X_test = X_test.reshape(test_samples, seq_length, 1)
    
    model = Sequential()
    model.add(Embedding(len(tokenizer.word_counts)+1, 300, input_length=seq_length))
    #model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2))) # input_shape=(seq_length,1)))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    earlyStop = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.0001, verbose=1, patience=1)
    print(model.summary())
    
    num_epochs = 100
    batch_size = 128

    model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_split=0.1, callbacks=[earlyStop])
    test_proba = model.predict(X_test, batch_size=batch_size)
    proba = test_proba[:,1].tolist()
    return proba, test_ids


# PREDICTION
def get_proba(train, test, classifier):
    ids = list()
    proba = list()
    training_dict = dict()
    
    index = 0
    for word, val in VOCAB_TRAIN.items():
        training_dict[word] = train[index:index+val]
        index += val
        
    complete_training_dict = get_sentence_dict(train)
    print('Learning probabilities from %s model! \nThis may take some time...' % classifier)
    for word, text in complete_training_dict.items():
        if len(text) > 100000:
            text = training_dict[word]
            
        if classifier == 'base':
            p, rowid = probability_classifier(text, test, word)
            proba.extend(p)
            ids.extend(rowid)
        elif classifier == 'logReg':
            p, rowid = logistic_regression(text, test, word)
            proba.extend(p)
            ids.extend(rowid)
        elif classifier == 'svm':
            p, rowid = SVM(text, test, word)
            proba.extend(p)
            ids.extend(rowid)
        elif classifier == 'lstm':
            p, rowid = lstm(text, test, word)
            proba.extend(p)
            ids.extend(rowid)
    
    results = tuple(zip(ids,proba))
    results = sorted(results,key=itemgetter(0))
    
    return results


###############################################################################


def main():
    train = load_train(DIR+TRAIN_FILE)
    test = load_test(DIR+TEST_FILE)

    classifiers = ['base', 'logReg', 'svm', 'lstm']

    for clf in classifiers:
        print('Running the %s classifier!' % clf)
        # get probabilities
        predictions = get_proba(train, test, clf)
        # write csv files
        dirname = 'submissions/'
        filename = clf + '_predictions.csv'
        with open(DIR+dirname+filename, 'w') as f:
            writer = csv.writer(f , lineterminator='\n')
            writer.writerow(['Id','Expected'])
            for tup in predictions:
                writer.writerow(tup)
    
    sys.exit()

if __name__ == '__main__':
    main()