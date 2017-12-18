"""This script performs out of sample testing over the various
categories of data collected for this project, and the models
generated for each category.

Simply specify the datasets for each model, as well as the model types
that you would like to evaluate. The config file is used to determine
the target value you are predicting.
"""

import pandas as pd
import pickle
import numpy
import configparser

from run_model import predict
from scoring_functions import class_averaged_mean_absolute_error as camae


### Dictionary mapping model name to the dataset that was used for training
category_data_dict = {'burgers':pd.read_pickle('burgers_reviews.pickle'),
					  'breweries':pd.read_pickle('breweries_reviews.pickle'),
					  'brunch':pd.read_pickle('breakfast_brunch_reviews.pickle')}


### Model types to evaluate
model_types = ['svc', 'ordr']

### List of models to use for evaluation
model_list = list(category_data_dict.keys())

config = configparser.ConfigParser()
config.read('config.ini')
params = config['DEFAULT']

review_col = params['review_col']
target_col = params['target_col']


for mod_type in model_types:

	### Write results out to txt file
	outfile = open("out_of_sample_results_{}_{}.txt".format(mod_type, target_col), 'w')
	outfile.write("Trained, Tested, Error\n")

	for train_cat in model_list:
		
		for test_cat in model_list:

			### Get actual target values for test category
			actual = category_data_dict[test_cat][target_col]

			### Predict target values using model trained on train category
			pred = predict(category_data_dict[test_cat][review_col], model_name=train_cat, model_type=model_type, target_val=target_col)

			### Calculate class-averaged mean absolute deviation
			error = camae(actual, pred)

			### Write out metrics
			outfile.write('{}, {}, {}\n'.format(train_cat, test_cat, round(error, 3)))

	### Close file
	outfile.close()