'''
Author: Kristopher Buote
Date: 9/15/2018

This script scrapes blogs hosted on medium and saves the data into csv.

The scrape function from medium_scraper.py and links gathered with get_links.py are utilized.

Workflow: scrape function collects data with beautiful soup and returns a dictionary loaded with data.
Data is pulled from dictionary and added to pandas dataframe.
Dataframe is saved periodically and finally at the end.

# TODO: More sophisticated error handling
# TODO: More efficient workflow?
'''

from medium_scraper import scrape
import pandas as pd
import numpy as np

# Load up the urls to be scraped!
base = 'links'
modifier = '_Jan_July_2017'
urls_df = pd.read_csv(base + modifier + '.csv', usecols=['url'])
urls = np.squeeze(urls_df.values)

# Here's a sample URL and response
data = scrape(urls[0])
print('Sample URL:', urls[0])
print('Sample data: ', data)

# Construct empty data frame with data dictionary keys
df = pd.DataFrame(columns=list(data.keys()))

# Scrape the rest of the urls and add to Dataframe. Save periodically.

temp_save_path = 'medium_data_temp' + modifier
final_save_path = 'medium_data' + modifier

for i in range(len(urls)):
    try:
        data = scrape(urls[i])
        df.loc[i] = list(data.values())

    except Exception as e:
        print('Error from URL # ', i)
        print(e)

    # Save periodically in case you get bounced!
    if i % 100 == 0:
        df.to_csv('../data/' + temp_save_path + '.csv')
        print("{0:.1f}% complete. Saving current progress as {1}.csv".format((i/len(urls))*100.0, temp_save_path))

# Check out data summary
print(df.head())
print(df.describe(), '\n')
print("{0}/{1} urls successfully scraped.".format(len(df),len(urls)))

df.to_csv('../data/' + final_save_path + '.csv')
