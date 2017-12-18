"""This file defines a single function that is used to make a prediction from a Series of text data.
This can be imported and run to make use of the models that have been trained for this project."""

import pickle
from get_features import generate_features

def predict(text_data, model_name='burgers', model_type='svc', target_val='Rating'):
	""" This function takes a Series of text data, as well as a model_name, model_type
	and target_val, and returns a set of predictions

	model_name: the unique identifying name of the model iteration that you want to use
	model_type: the type of machine learning model to use (options: 'svc', 'rf', 'knn', 'ordr')
	target_val: the target variable to predict. 

	NOTE: a model of type 'model_type' predicting the 'target_val' variable must have already
		  been trained and saved in order for this function to work"""

	### Generate the features using the tfidf and SVD that were trained for model_name
	feats = generate_features(text_data, model_name, train=False)

	try:
		### Attempt to load the model, if it exists
		with open("{}/best_{}_{}.pickle".format(model_name, model_type, target_val), 'rb') as f:
			model = pickle.load(f)
	except:
		print("Cannot find specified model: {}, {}, {}".format(model_name, model_type, target_val))

	### Make a prediction using that model
	pred = model.predict(feats)

	return pred

