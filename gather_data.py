"""This script gathers and cleans all of the data for my project.
It makes use of the yelp API to search for businesses by certain terms, and 
then scrapes the most recent reviews from Yelp's website."""

from yelp_api import obtain_access_token, search
from yelp_api import API_HOST, TOKEN_PATH, SEARCH_PATH, SEARCH_LIMIT

from yelp_scraper import get_reviews

import time
import pickle
import numpy as np
import pandas as pd

# Specific categories to search for on Yelp
# All supported categories available at: https://www.yelp.com/developers/documentation/v3/all_category_list
# CATEGORIES = ['breweries', 'burgers', 'breakfast_brunch']
CATEGORIES = ['asianfusion']
DEFAULT_LOCATION = 'US'

# Get Access Token

access_token = obtain_access_token(API_HOST, TOKEN_PATH)

cat_dfs = []

# Loop over all categories and gather reviews

for cat in CATEGORIES:

	cat_businesses = []

	# Yelp API only returns ~20 businesses per search, so you need to search multiple times and offset the results
	for offset in np.arange(10):
		response = search(access_token, cat, DEFAULT_LOCATION, offset * SEARCH_LIMIT)
		cat_businesses.extend(response['businesses'])
		print("{} found!".format(len(response['businesses'])))
	
	cat_reviews = []
	print("Found {0} results for {1}".format(len(cat_businesses), cat))

	count = 1
	start = time.time()

	# For all business IDs gathered for this category
	for b in cat_businesses:
		if count % 10 == 0:
			print("Scraping business {0} out of {1}".format(count, len(cat_businesses)))
		try:
			# Scrape the reviews from Yelp's site
			revs = get_reviews(b['id'])
			cat_reviews.append(revs)
		except Exception as e:
			print("Error scraping reviews for {}".format(b['id'].encode("ascii", "ignore").decode('ascii')))
			print(e)
		
		count += 1
		time.sleep(2)

	print("{0} completed in {1} minutes".format(cat, (time.time() - start) / 60))
	cat_df = pd.concat(cat_reviews)
	cat_df['Category'] = cat

	cat_df.reset_index(inplace=True)

	# Output specific category's reviews
	cat_df.to_pickle("{0}_reviews.pickle".format(cat))

	cat_dfs.append(cat_df)

final_df = pd.concat(cat_dfs)

# Concatenate reviews from all categories and save to a pickle file
final_df.reset_index().to_pickle('all_reviews.pickle')


