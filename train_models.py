"""This file is for training various machine learning models, as well as outputting their predictions"""

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from scoring_functions import class_averaged_mean_absolute_error, multiclass_weighted_entropy

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from mord import OrdinalRidge

import pickle
import configparser
import pandas as pd
import time


### Select which models to train. Options are 'svc', 'rf', 'knn', 'ordr'
models_to_train = ['knn','ordr']

### Here we create sklearn scorers from the two custom error functions we had defined
### NOTE: by setting 'greater_is_better' to false, the metrics returned by
###       these scorers will be negative. Don't be alarmed by getting a negative mean absolute error
mae = make_scorer(class_averaged_mean_absolute_error, greater_is_better=False, needs_proba=False)

mwe = make_scorer(multiclass_weighted_entropy, greater_is_better=False, needs_proba=True)


config = configparser.ConfigParser()
config.read('config.ini')
params = config['DEFAULT']

model_name = params['name']
target_col = params['target_col']

exclude_cols = params['exclude_cols'].split(',') + [target_col]

# Load training and testing data

train_set = pd.read_pickle('{}/train_set.pickle'.format(model_name))
test_set = pd.read_pickle('{}/test_set.pickle'.format(model_name))

X_train = train_set[[c for c in train_set.columns if c not in exclude_cols]]
X_test = test_set[[c for c in test_set.columns if c not in exclude_cols]] 

Y_train = train_set[target_col]
Y_test = test_set[target_col]


outfile = open('{}/grid_search_log_{}.txt'.format(model_name, target_col), 'w')

# Support Vector Classifier
# -------------------------
# The two main parameters for a SVC are C and gamma
# C trades off accuracy for simplicity of the model. A low C favors a simpler decision surface, while a high C favors high accuracy (could overfit)
# Gamma determines the 'radius' of influence for the support vectors. A low value means a larger radius, while a high value means a smaller radius.

if 'svc' in models_to_train:
	svc_params = {'C':[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], 'gamma':[0.01, 0.1, 1.0, 10.0]}

	### If predicting sentiment, use f1_macro otherwise use mae
	svc_scorer = 'f1_macro' if target_col == "sentiment" else mae

	svc = SVC(probability=False)

	### Perform GridSearch
	start = time.time()
	print("Starting SVC Grid search")
	svc_grid_search = GridSearchCV(estimator=svc, param_grid=svc_params, scoring=svc_scorer, cv=5, refit=True)

	svc_grid_search.fit(X_train, Y_train)
	print("Grid search completed in {} minutes!".format((time.time() - start) / 60))

	### Make predictions on left-out test set
	Y_pred = svc_grid_search.predict(X_test)
	pd.DataFrame({'Actual':Y_test, 'Pred':Y_pred}).to_pickle('{}/svc_pred_{}.pickle'.format(model_name, target_col))

	### Output best model
	pickle.dump(svc_grid_search.best_estimator_, open('{}/best_svc_{}.pickle'.format(model_name, target_col), 'wb'))

	outfile.write("Best SVC params:{}\n".format(svc_grid_search.best_params_))
	outfile.write("Best SVC score:{}\n".format(svc_grid_search.best_score_))

# Random Forest Classifier
# ------------------------
# The main parameters for a Random Forest Classifier are the number of trees to create, and 
# other parameters regarding the growing of the trees, such as: max features per split, min samples per split, etc

if 'rf' in models_to_train:
	### If predicting sentiment, use f1_macro otherwise use mwe
	rfc_scorer = 'f1_macro' if target_col == "sentiment" else mwe

	rfc_params = {'n_estimators':[10, 50, 100], 'max_features':[0.05, 0.15, "auto"], 'min_samples_split':[2, 0.01]}
	rfc = RandomForestClassifier()

	### Perform GridSearch
	start = time.time()
	print("Starting RandomForest Grid search")
	rfc_grid_search = GridSearchCV(estimator=rfc, param_grid=rfc_params, scoring=rfc_scorer, cv=5, refit=True)

	rfc_grid_search.fit(X_train, Y_train)
	print("Grid search completed in {} minutes!".format((time.time() - start) / 60))

	Y_pred = rfc_grid_search.predict(X_test)
	pd.DataFrame({'Actual':Y_test, 'Pred':Y_pred}).to_pickle('{}/rf_pred_{}.pickle'.format(model_name, target_col))

	pickle.dump(rfc_grid_search.best_estimator_, open('{}/best_rf_{}.pickle'.format(model_name, target_col), 'wb'))

	outfile.write("Best RandomForest params:{}\n".format(rfc_grid_search.best_params_))
	outfile.write("Best RandomForest score:{}\n".format(rfc_grid_search.best_score_))

# KNeighbors Classifier 
# ---------------------
# The main parameters for a K nearest neighbors classifier are the number of neighbors and 
# the weighting of the neighbors (either uniform or based on distance)

if 'knn' in models_to_train:
	### If predicting sentiment, use f1_macro otherwise use mwe
	knn_scorer = 'f1_macro' if target_col == "sentiment" else mwe

	knn_params = {'n_neighbors':[5, 10, 50, 100], 'weights':['uniform', 'distance']}
	knn = KNeighborsClassifier()

	### Perform GridSearch
	start = time.time()
	print("Starting KNN Grid search")
	knn_grid_search = GridSearchCV(estimator=knn, param_grid=knn_params, scoring=knn_scorer, cv=5, refit=True)

	knn_grid_search.fit(X_train, Y_train)
	print("Grid search completed in {} minutes!".format((time.time() - start) / 60))

	Y_pred = knn_grid_search.predict(X_test)
	pd.DataFrame({'Actual':Y_test, 'Pred':Y_pred}).to_pickle('{}/knn_pred_{}.pickle'.format(model_name, target_col))

	pickle.dump(knn_grid_search.best_estimator_, open('{}/best_knn_{}.pickle'.format(model_name, target_col), 'wb'))

	outfile.write("Best KNN params:{}\n".format(knn_grid_search.best_params_))
	outfile.write("Best KNN score:{}\n".format(knn_grid_search.best_score_))

# Ordinal Ridge Regression
# --------------------------------------
# This model is part of the 'mord' package, which was built using the scikit-learn API
# The main parameter to tune is 'alpha', which is the l2 regularization parameter (with 0 being no regularization)

if 'ordr' in models_to_train:
	### If predicting sentiment, use f1_macro otherwise use mae
	ordr_scorer = 'f1_macro' if target_col == "sentiment" else mae

	ordr_params = {'alpha':[0.1, 0.5, 1.0, 5.0, 10.0]}
	ordr = OrdinalRidge()

	### Perform GridSearch
	start = time.time()
	print("Starting Ordinal Ridge Grid search")
	ordr_grid_search = GridSearchCV(estimator=ordr, param_grid=ordr_params, scoring=ordr_scorer, cv=5, refit=True)

	ordr_grid_search.fit(X_train, Y_train)
	print("Grid search completed in {} minutes!".format((time.time() - start) / 60))

	Y_pred = ordr_grid_search.predict(X_test)
	pd.DataFrame({'Actual':Y_test, 'Pred':Y_pred}).to_pickle('{}/ordr_pred_{}.pickle'.format(model_name, target_col))

	pickle.dump(ordr_grid_search.best_estimator_, open('{}/best_ordr_{}.pickle'.format(model_name, target_col), 'wb'))

	outfile.write("Best OrdinalRidge params:{}\n".format(ordr_grid_search.best_params_))
	outfile.write("Best OrdinalRidge score:{}\n".format(ordr_grid_search.best_score_))

outfile.close()