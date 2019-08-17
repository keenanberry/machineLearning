#!/usr/bin/env python3
from __future__ import print_function

"""skinMNIST.py: Skin cancer image classification models.
   Program is designed to compare classification accuracy of various ML
   models on the HAM100000 dataset, an MNIST image collection of 7 types
   of skin cancers. See Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 
   dataset, a large collection of multi-source dermatoscopic images of common 
   pigmented skin lesions. Sci. Data 5, 180161 (2018). 
   doi: 10.1038/sdata.2018.161 for more information.

   #TODO in future add predict proba option and log loss scoring metric
"""

__author__  = "Keenan Berry"
__date__    = "05/06/2019"

###############################################################################

import os, sys, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import sklearn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, log_loss

import keras
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau

INPUT_DIR = os.path.join('.', 'data')
# os.mkdir('results')
OUTPUT_DIR = os.path.join('.', 'results')

CANCER_ID = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

def process_data(meta_df, rbg_df):
    meta_df['cell_type'] = meta_df['dx'].map(CANCER_ID.get) 
    # fill NA values with column mean
    for col in meta_df.columns[meta_df.isna().any()].tolist():
        meta_df[col].fillna((meta_df[col].mean()), inplace=True)
    
    # match labels by value counts
    cell_type_counts = meta_df['cell_type'].value_counts().to_dict()
    cell_type_code = {y:x for x,y in rbg_df['label'].value_counts().to_dict().items()}
    meta_df['cell_type_id'] = meta_df['cell_type'].map(
        lambda x: cell_type_code[cell_type_counts[x]])
    
    # reshape pixel data: (height x width x channel) = (28 x 28 x 3)
    pixel_data = rbg_df.drop(['label'], axis=1).values
    pixel_reshape = pixel_data.reshape(pixel_data.shape[0], 28, 28, 3)
    #print(pixel_reshape)

    return meta_df, pixel_reshape
    

def split_data(x, y, ts, rs):
    # train test split with stratified sampling
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=ts,random_state=rs, stratify=y)
    
    return x_train, x_test, y_train, y_test


def standardize_data(x):
    x_mean = np.mean(x)
    x_std = np.std(x)
    x = (x - x_mean)/x_std

    return x


# function to save plot of sklearn.metrics confusion_matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          filename='confusion_matrix.png'):
    """
    This function creates and saves the confusion matrix plot.
    Normalization can be applied by setting `normalize=True`.
    Output file name can be selected by defining `filename=...`.
    """
    print('plotting confusion matrix...')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    print('plot saved!') 
    plt.close()  # clear plot


# function to save plot of model's validation loss and validation accuracy
def plot_model_history(model_history, filename='model_history.png'):
    print('plotting model history...')
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    print('plot saved!')
    plt.close()  # clear plot


def evaluate_model(y_true, y_pred, metric='Accuracy', model_name=''): 
    score = None
    if (metric == 'Accuracy'):
        # get classification report and confusion matrix
        cr = classification_report(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        # print classification report
        print(model_name, 'classification report:\n', cr)
        # save confusion matrix heatmap
        plot_confusion_matrix(cm, classes=range(7), filename=model_name+'_confusionMatrix.png')

        # get score
        score = accuracy_score(y_true, y_pred)
        print(model_name, 'score:', score)
    elif (metric == 'Log Loss'):
        score = log_loss(y_true, y_pred)
        
    return score


def plot_scores(scores, metric='Accuracy'):
    filename = None
    classifiers = list(scores.keys())
    scores = [x*100 for x in list(scores.values())]
    x_pos = np.arange(len(scores))
    # plot fig
    #plt.figure(figsize=(4,3))
    plt.bar(x_pos, scores, align='center', alpha=0.8)
    plt.xticks(x_pos, classifiers)
    if (metric == 'Accuracy'):
        filename = 'accuracy_scores.png'
        plt.ylabel(metric+' (%)')
    elif (metric == 'Log Loss'):
        filename = 'logloss_scores.png'
        plt.ylabel(metric)
    plt.title('Classifier Score Report')
    # include score labels
    for i in range(len(classifiers)):
        plt.text(x = x_pos[i] , y = scores[i]-5, s = round(scores[i], 2), size = 15, 
                 horizontalalignment='center', verticalalignment='center', color='white')
    
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    print('plot saved!')
    plt.close()  # clear plot   


def svc_param_selection(x, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = ['auto', 0.001, 0.01, 0.1, 1.0]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(x, y)
    
    return grid_search.best_params_  # returns dictionary


def SVM(x, y):
    # train test split
    x_train, x_test, y_train, y_test = split_data(x, y, 0.3, 42)
    # standard scaler
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    
    # find best parameters
    print('Finding best hyperparameters for SVM model...')
    best_params_dict = svc_param_selection(x_train, y_train, nfolds=3)
    print('best params:', best_params_dict)

    # fit best model and make predictions
    print('Evaluating SVM model...')
    model = svm.SVC(kernel='rbf', C=best_params_dict['C'], gamma=best_params_dict['gamma'])
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = evaluate_model(y_test, y_pred, model_name='svm')
    
    return score


def knn(x, y):
    # train test split
    x_train, x_test, y_train, y_test = split_data(x, y, 0.3, 42)

    # use cross validation to determine optimal k
    n = list(range(3, 85, 2))
    model_scores = []
    print('Finding optimal k value for kNN model...')
    for k in n:
        model = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=10)
        model_scores.append(scores.mean())
    mse = [1 - x for x in model_scores]
    optimal_k = n[mse.index(min(mse))]
    print('optimal k =', optimal_k)

    # fit optimal model and predict
    print('Evaluating optimal kNN model...')
    model = KNeighborsClassifier(n_neighbors=optimal_k)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = evaluate_model(y_test, y_pred, model_name='knn')

    return score


def rf_param_search(x, y, nfolds, nIter):
    
    n_estimators = [int(x) for x in np.linspace(start = 4, stop = 200, num = 7)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(2, 34, num = 8)]
    min_samples_split = [int(x) for x in np.linspace(2, 10, 5)]
    min_samples_leaf = [int(x) for x in np.linspace(1, 5, 5)]
    bootstrap = [True, False]

    # create random grid dictionary
    random_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}

    rf = RandomForestClassifier()
    # Random search of parameters 
    random_search = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, 
                                       n_iter=nIter, cv=nfolds)
    # fit the random search model
    random_search.fit(x, y)
    
    return random_search.best_params_  # returns dictionary


def randomForest(x, y):
    # train test split
    x_train, x_test, y_train, y_test = split_data(x, y, 0.3, 42)

    # use the random grid to search for best hyperparameters
    print('Finding hyperparameters for random forest model...')
    best_random_params = rf_param_search(x_train, y_train, nfolds=3, nIter=20)
    print('params:', best_random_params)

    # fit best model and make predictions
    model = RandomForestClassifier(n_estimators=best_random_params['n_estimators'],
                                   max_depth=best_random_params['max_depth'],
                                   max_features=best_random_params['max_features'],
                                   min_samples_split=best_random_params['min_samples_split'],
                                   min_samples_leaf=best_random_params['min_samples_leaf'],
                                   bootstrap=best_random_params['bootstrap'])
    print('Evaluating random forest model...')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = evaluate_model(y_test, y_pred, model_name='rf')

    return score


def cnn(x, y, num_epochs, batch_size):
    # train test split
    x_train, x_test, y_train, y_test = split_data(x, y, 0.2, 42)

    # standardize pixel data
    x_train = standardize_data(x_train)
    x_test = standardize_data(x_test)
    # print('train data shape after standardization:', x_train.shape)
    # train validate split
    x_train, x_validate, y_train, y_validate = split_data(x_train, y_train, 0.1, 2)

    """
    The CNN model architecture used in this analysis is adapted from a Kaggle 
    kernel for the Skin Cancer MNIST: HAM10000 dataset. The code and reasoning 
    behind each layer selection can be found at the following link:
    https://www.kaggle.com/sid321axn/step-wise-approach-cnn-model-77-0344-accuracy
    """

    # set the CNN model
    input_shape = (28, 28, 3)
    num_classes = 7

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding ='Same', input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding ='Same',))
    model.add(MaxPool2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu', padding ='Same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding ='Same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.40))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    # define the optimizer
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # compile the model
    model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
    # set a learning rate annealer
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                                patience=3, 
                                                verbose=1, 
                                                factor=0.5, 
                                                min_lr=0.00001)

    # augment pixel data to prevent overfitting 
    datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=10,
            zoom_range = 0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=False,
            vertical_flip=False)

    datagen.fit(x_train)

    # fit the model
    history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                                epochs = num_epochs, validation_data = (x_validate,y_validate),
                                verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size, 
                                callbacks=[learning_rate_reduction])
    
    # evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    loss_v, accuracy_v = model.evaluate(x_validate, y_validate, verbose=1)
    print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))
    print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))
    plot_model_history(history)

    # predict y_test
    y_pred = model.predict(x_test)
    y_true = np.argmax(y_test, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    score = evaluate_model(y_true, y_pred_classes, model_name='cnn')
    
    return max(accuracy,score)

###############################################################################

def main():
    # load metadata
    meta_df = pd.read_csv(os.path.join(INPUT_DIR, 'HAM10000_metadata.csv'))
    # load (28x28) pixel rbg-value data
    rbg_df = pd.read_csv(os.path.join(INPUT_DIR, 'hmnist_28_28_RGB.csv'))

    # process data
    meta_df, cnn_pixel_data  = process_data(meta_df, rbg_df)
    
    # create label dictionary
    type_label_dict = pd.Series(meta_df.dx.values,index=meta_df.cell_type_id).to_dict()
    labels = meta_df['cell_type_id']
    # one-hot-encode labels
    one_hot_labels = to_categorical(labels, num_classes=7)

    # prepare data for "traiditonal" image classification models
    rgb_features = rbg_df.drop(columns=['label'],axis=1)
    # max normalization of features
    norm_features = rgb_features / 255.0
    
    # SVM model
    svm_score = SVM(rgb_features, labels)
    # kNN model
    knn_score = knn(norm_features, labels)
    # random forest model
    rf_score = randomForest(rgb_features, labels)

    # prepare data for CNN model
    meta_features = meta_df.drop(columns=['cell_type_id'],axis=1)
    
    # CNN model using Keras
    cnn_score = cnn(cnn_pixel_data, one_hot_labels, num_epochs=40, batch_size=64)
    
    print('Testing is complete!')
    scores_dict = {
        'SVM': svm_score,
        'kNN': knn_score,
        'Random Forest': rf_score,
        'CNN': cnn_score 
    }
    print('The model with the best score is %s!' % max(scores_dict, key=scores_dict.get))
    
    plot_scores(scores_dict)

    sys.exit()

###############################################################################

if __name__ == '__main__':
    main()