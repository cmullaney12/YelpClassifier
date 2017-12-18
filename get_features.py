""" This file is used to generate features from our gathered reviews.
It can either be run directly from the command line or imported 
to make use of the 'generate_features' function.
"""

import configparser
import pandas as pd
import pickle
import string
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.stem.snowball import SnowballStemmer

import os


def general_sentiment(rating):
	"""
	This function maps a rating (on the scale of 1-5)
	to a general sentiment class. This is only useful if you want
	to train a new classifier for general sentiment.

	Our mapping is that 3 is neutral sentiment, 4 or 5 is positive, and 1 or 2 is negative.
	These mappings can easily be changed by altering this function
	"""

	if (rating > 3):
		return 'positive'
	elif (rating < 3):
		return 'negative'
	else:
		return 'neutral'

def percent_capitalization(review):
	"""
	This function returns the proportion of a given textual review that is capitalized
	"""

	num_uppercase = sum(1 for char in review if char.isupper())

	return num_uppercase / len(review)


def count_punctuation(review):
	"""
	This function returns normalized frequency counts for a 
	number of common punctuation marks
	"""

	marks = ['!', '.', ',', '?', '$', '#']

	return pd.Series({m: review.count(m) / len(review) for m in marks})


def clean_and_stem_text(review):
	"""
	This function removes all non-alphanumeric symbols from the text,
	converts it to lowercase, and stems each term in the document
	"""

	no_symbols = re.sub(r'[^\w]', ' ', review.lower())

	stemmer = SnowballStemmer("english")

	stemmed_words = [stemmer.stem(word) for word in review.split()]

	return ' '.join(stemmed_words)


def vectorize(text, save_file, fit=True):
	"""
	This function vectorizes some given text, applying a TFIDF vectorizer and SVD transformer to it.

	If fit=True, a new TFIDF and SVD are fitted and saved using the given text data.
	If fit=False, existing TFIDF and SVD are loaded from save_file and used to transform the text.
	"""

	if fit:

		### Create and fit a new TFIDF vectorizer
		tfidf = TfidfVectorizer(strip_accents='unicode', analyzer='word', lowercase=True, max_df=0.8, min_df=0.01, smooth_idf=True)

		train_tfidf = tfidf.fit_transform(text).toarray()

		### Create and fit a new Truncated SVD
		svd = TruncatedSVD(n_components = min(200, len(tfidf.idf_)))

		train_features = svd.fit_transform(train_tfidf)

		### Save the TFIDF and SVD for future use
		with open('{}/tfidf.pickle'.format(save_file), 'wb') as f:
			pickle.dump(tfidf, f)

		with open('{}/svd.pickle'.format(save_file), 'wb') as f:
			pickle.dump(svd, f)

		return train_features

	else:

		### Load the existing TFIDF and SVD
		with open('{}/tfidf.pickle'.format(save_file), 'rb') as f:
			tfidf = pickle.load(f)

		with open('{}/svd.pickle'.format(save_file), 'rb') as f:
			svd = pickle.load(f)

		### Transform the data
		test_tfidf = tfidf.transform(text).toarray()
		test_features = svd.transform(test_tfidf)

		return test_features

def generate_features(text_data, save_file, train=True):
	""" This function generates and outputs all the needed features for
	some given textual data (in Series format). save_file indicates
	where to save or load the tfidf and SVD objects.
	Use train=True if you want to retrain your tfidf and SVD objects.
	Use train=False if they have already been trained.
	"""
	percent_cap = text_data.apply(percent_capitalization)
	punc_counts = text_data.apply(count_punctuation)

	cleaned_text = text_data.apply(clean_and_stem_text)

	tfidf_feats = vectorize(cleaned_text, save_file, fit=train)

	features = percent_cap.to_frame('percent_cap').join(punc_counts).join(pd.DataFrame(tfidf_feats, index=text_data.index))

	return features


### If script is run directly, call the following code
if __name__ == "__main__":

	config = configparser.ConfigParser()
	config.read('config.ini')
	params = config['DEFAULT']

	model_name = params['name']
	dataset = pd.read_pickle(params['dataset'])
	train_set_size = float(params['train_size'])
	review_col = params['review_col']
	target_col = params['target_col']

	### Create directory for this model, if it does not exist
	if not os.path.exists(model_name + '/'):
		os.makedirs(model_name + '/')

	### Generate the training and testing sets
	train_indices = dataset.sample(frac=train_set_size).index
	test_indices = [ind for ind in dataset.index if ind not in train_indices]

	train_set = dataset.loc[train_indices]
	test_set = dataset.loc[test_indices]

	print('{} Training data points, {} Testing data points'.format(train_set.shape[0], test_set.shape[0]))

	### Calculate the general sentiment
	train_set['sentiment'] = train_set[target_col].apply(general_sentiment)
	test_set['sentiment'] = test_set[target_col].apply(general_sentiment)

	### Generate numerical features for the text
	train_feats = generate_features(train_set[review_col], save_file=model_name, train=True)
	test_feats = generate_features(test_set[review_col], save_file=model_name, train=False)

	### Combine all features into dataframes and output to pickle
	train_set = train_set.join(train_feats)
	test_set = test_set.join(test_feats)

	train_set.to_pickle('{}/train_set.pickle'.format(model_name))
	test_set.to_pickle('{}/test_set.pickle'.format(model_name))

	print('Subsets saved')
