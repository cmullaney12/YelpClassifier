"""This file provides functions for scraping data from YELP"""

from bs4 import BeautifulSoup
import urllib.request as ur
import pandas as pd
import re

# Method for sorting reviews on Yelp's site
SORT_BY = "date_desc"

# The URL used for scraping
YELP_BASE_URL = "https://www.yelp.com/biz/{0}?sort_by={1}"
REVIEW_LIMIT = 15


def get_reviews(business_id):
	""" This function uses a given business ID to scrape recent reviews for a Yelp business"""

	# Open the webpage for this specific business, and parse the HTML
	site = ur.urlopen(YELP_BASE_URL.format(business_id, SORT_BY))
	soup = BeautifulSoup(site, 'html.parser')

	# Find the review list, and get the first REVIEW_LIMIT number of reviews
	review_list = soup.find('ul', class_='reviews')
	top_reviews = review_list.findAll('div', class_='review-content', limit=REVIEW_LIMIT)

	r_list = []

	# For each review, scrape the rating and the text content and accumulate a list
	for review in top_reviews:
		raw_rating = review.find('div', class_='i-stars')['title']
		cleaned_rating = int(re.search(r'\d+', raw_rating).group()) if raw_rating is not None else None
		r_list.append({'Business ID':business_id, 'Rating':cleaned_rating, 'Text':review.find('p', attrs={'lang':'en'}).text})

	# Append together a dataframe of reviews for this business
	df = pd.DataFrame(r_list)

	return df

