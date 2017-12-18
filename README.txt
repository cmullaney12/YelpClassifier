README for my Yelp Classification project:

The goal of this project is to gather reviews from Yelp and use
machine learning techniques to build supervised models that
can accurately predict each review's rating (from 1 to 5 stars).

Before running any code, be sure to install all required packages
found in the 'requirements.txt' file by running:

pip install -r requirements.txt

After doing that, you can begin the process of scraping reviews,
generating features, training models, evaluating test sets,
and performing out of sample testing.

Below I will list, in order, each script that you must run:

1. gather_data.py:
   This script queries the Yelp API to search for businesses that fall into
   specified categories. Next, it scrapes Yelp's website to get up to 15 of the
   most recent reviews for each of those businesses.

   OUTPUTS: pickled Pandas dataframes containing Yelp reviews and ratings.
   One pickle file is saved for each category ({category_name}_reviews.pickle),
   as well as a file containing an aggregation of all reviews (all_reviews.pickle).

   BEFORE RUNNING:
   	- Edit the 'CATEGORIES' variable to include any desired yelp categories to scrape,
   	  or stick with the default options that I have selected

   This script can take a long time to scrape all of the Yelp reviews, so I would suggest
   using the datasets that I've included (breweries_reviews.pickle, etc.) and skipping this step

BEFORE CONTINUING: Open and edit the 'config.ini' file. This file defines some overarching
parameters for the iteration of the project you are about to build.

	Mainly specify:
	- name: the unique name/ID of the project you are creating (this is used for creating a subfolder to store results)
			   (Ex. 'first_trial' or 'brunch_pt2', etc)
	- dataset: the location of the pickled DataFrame to use for training (this file should be a direct output 
			   from gather_data.py). If you want to train on a single Yelp category, specify that dataset here.
			   (Ex. breweries_reviews.pickle or all_reviews.pickle)
	- train_size : a proportion representing the amount of data to use for the training set
				   (Ex. 0.75 or 0.8)
	- review_col: the name of the column containing the textual review (should leave this as Text)
	- target_col: the name of the target variable to predict 
				  (should be Rating, but could use sentiment if that's what you wanted to predict)
	- exclude_cols: all extra columns in the training dataframe that you don't want to be included in the analysis
					(should leave these as I have defined, unless you edited the {}_reviews.pickle files at all)


2. get_features.py:
   This script uses the information found in the config file to generate a subfolder for your project,
   create training and testing splits of your data, and generate features for both of those subsets.
   In the process, it also fits and saves a TFIDFVectorizer and a TruncatedSVD object. Both of these 
   will be used later when making predictions.

   OUTPUTS: a training feature set (train_set.pickle), a testing feature set (test_set.pickle),
   			a fitted TFIDFVectorizer (tfidf.pickle), and a fitted SVD object (svd.pickle).


3. train_models.py:
   This script uses the saved features from 'get_features.py', as well as the target_col defined in the config file
   to perform grid search for multiple types of machine learning models.

   OUTPUTS: a log file containing grid search results (grid_search_log_{target_variable}.txt), as well as 
   			test set predictions ({model_type}_pred_{target_col}.pickle) and the best grid-searched
   			estimator (best_{model_type}_{target_col}.pickle) for each model_type specified

   BEFORE RUNNING:
    - Edit the 'models_to_train' variable to contain any of the models that you wish to train
    	OPTIONS: 'svc', 'rf', 'knn', 'ordr' (suggested to stick with svc and ordr since they performed the best)

   NOTE: This script can also take a while to complete because I am performing grid search AND cross validation


4. evaluate_models.py:
   This script evaluates the test performance for all of the models that you have created for the given iteration
   of this project (specified in config file). 

   OUTPUTS: a confusion matrix (confusion_matrix_{model_type}_{target_col}.png) for each specified model type

   BEFORE RUNNING:
    - Edit the 'model_types' variable to include any models that you wish to evaluate
    	OPTIONS: 'svc', 'rf', 'knn', 'ordr' (these should match the models chosen in step 3)

OPTIONAL:
5. out_of_sample.py:
   This script calculates out of sample metrics for all of the subdomains of Yelp data that you specify

   OUTPUTS: a log file (out_of_sample_results_{model_type}_{target_col}.txt)
   			for each model type ('svc', etc) that contains the out of sample metrics

   BEFORE RUNNING:
    - Edit the 'category_data_dict' dictionary to include a mapping from each specific project name 
    to the pickle Dataframe (output from gather_data.py) that was used for training that model
    - Edit the 'model_types' variable to include any  models that you wish to evaluate
    	OPTIONS: 'svc', 'rf', 'knn', 'ordr' (these should match the models chosen in step 3 and 4)

   NOTE: out-of-sample testing will be performed for every ordered pairing of project 
   		 included in the 'category_data_dict'

###################################################################################
If after training all of the models in step 3, you want to use them, you can simply
import the function 'predict' from 'run_model.py'.

The predict function takes a Series of text data, as well as parameters specifying the
project name (ex. brunch), model type (ex. svc), and target variable (ex. Rating).
Those 3 parameters identify which model to use.

This function may take some time because it needs to fully generate the features for the given text,
as well as load the specified machine learning model and calculate predictions.
###################################################################################

Steps 2-5 listed above can be repeated as many times as desired to test out different
training data sets. Before rerunning each of the scripts in the order described, just
edit any relevant information in the 'config.ini' file before repeating the steps.

ADVICE: Tyically I would create a separate project for each category gathered from
Yelp (edit the project name and the location of the training dataset in the config file),
as well as a project that utilized all of the gathered data (dataset=all_reviews.pickle).

The entire lifecycle of this project can take a fairly long time to run, because
of the large amount of data that needs to be scraped and the number of models that
need to be created. I suggest starting out with 'breweries' since it was the smallest
of all the datasets.
