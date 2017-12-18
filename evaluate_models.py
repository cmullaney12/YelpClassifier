""" This file is used to evaluate model performance on left-out testing sets.
This script calculates class-averaged mean asbolute error and outputs a confusion matrix
"""

from sklearn.metrics import confusion_matrix
from scoring_functions import class_averaged_mean_absolute_error

import matplotlib.pyplot as plt
import configparser
import itertools
import numpy as np
import pandas as pd

config = configparser.ConfigParser()
config.read('config.ini')
params = config['DEFAULT']
model_name = params['name']
target_col = params['target_col']

### List of model types to evaluate
model_types = ['knn','ordr']

### Mapping of target class labels based on target variable
target_val_labels = {'sentiment':['negative', 'neutral', 'positive'],
                     'Rating':['1', '2', '3', '4', '5']}

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    This code was taken from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

for model_type in model_types:

    ### Load predictions for this model type
    predictions = pd.read_pickle('{}/{}_pred_{}.pickle'.format(model_name, model_type, target_col))

    ### Calculate the error for the test set
    error = round(class_averaged_mean_absolute_error(predictions['Actual'], predictions['Pred']), 3)
    
    ### Compute confusion matrix
    cnf_matrix = confusion_matrix(predictions['Actual'], predictions['Pred'])
    np.set_printoptions(precision=2)

    ### Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=target_val_labels[target_col], normalize=True,
                          title='{} Normalized confusion matrix, Error = {}'.format(model_type, error))

    ### Save figure
    plt.savefig("{}/confusion_matrix_{}_{}.png".format(model_name, model_type, target_col))