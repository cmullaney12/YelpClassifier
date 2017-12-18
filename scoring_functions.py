"""
This file defines the custom scoring functions that are used for evaluating
the performance of our models
"""

import numpy as np
import pandas as pd

def class_averaged_mean_absolute_error(y, y_pred):
	""" This function uses label predictions and calculates
	average mean absolute deviation values across all classes
	of the target variable
	"""
	error = abs(y - y_pred)
	df = pd.DataFrame({'y':y, 'error':error})

	class_avg_errors = df.groupby('y')['error'].mean()

	return class_avg_errors.mean()

def review_weighted_entropy(y, y_prob_preds):
	""" This calculates the 'weighted entropy' for a single
	review given the probability predictions for each target class.
	The error is weighted by the absolute deviation from the 'true' label.
	"""
	review_vals = np.arange(1, 6)
	abs_diff = abs(review_vals - y)

	return sum(abs_diff * np.array(y_prob_preds))


def multiclass_weighted_entropy(y, y_prob_preds):
	""" This functions uses the predicted target class probabilities
	to calculate the weighted entropy across all classes of the target variable
	"""

	entropy = map(review_weighted_entropy, y, y_prob_preds)
	df = pd.DataFrame({'y':y, 'error':list(entropy)})

	class_avg_errors = df.groupby('y')['error'].mean()

	return class_avg_errors.mean()

