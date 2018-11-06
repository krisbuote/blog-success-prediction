'''
Created by Moses Hassan
August 2018 Chiang Mai, Thailand

This simple script will collects links to posts on Medium for a specified tag

NOTE:
Script is still under development.  It does not handle errors and requires more commenting and documentation.

Update 9/15/2018 by Kristopher Buote: changed the final pd.to_csv to include the '.csv' file format.
Also included date of blog and tag used to find it for future reference.

This script collects 'membership' articles (starred articles on medium) behind the paywall.
'''

# Import Libraries

import time
import requests
import random

import pandas as pd

from calendar import monthrange
from datetime import datetime

from bs4 import BeautifulSoup as BSoup

print('Packages loaded..........')

# Set Variables

BASE_URL = 'https://medium.com/'
TAGS = ['Travel']		# Specify tags to search
YEARS = [2017]		# Specify years to search through
MONTHS = range(12,0, -1)

all_links = []			# This will collect all links
link_dates = []
search_tags = []

for tag in TAGS:
    for year in YEARS:
        for month in MONTHS:
            current_month = datetime.now().month
            # Start from 2 months before current month
            if month < current_month-1:
                nums_days = monthrange(year, month)[1]	# tuple holds number of days in index=1
                for day in range(nums_days, 0, -1):
                    parameter_url = 'tag/{tag}/archive/{year}/{month:0>2}/{day:0>2}'.format(
                        tag=tag,
                        year=year,
                        month=month,
                        day=day)
                    full_url = BASE_URL + parameter_url
                    print('Sleeping........')
                    time.sleep(random.random()*1)	# Space out requests randomly
                    print('Requesting {0}'.format(full_url))
                    page = requests.get(full_url)
                    print('Response: {0}'.format(page.status_code))
                    soup = BSoup(page.text, 'html.parser')
                    print('Soup ready...........')
                    posts = soup.findAll('div', {'class': 'postArticle-readMore'})
                    print('Found {0} posts'.format(len(posts)))
                    for post in posts:
                        link = post.a.get('href')
                        all_links.append(link)
                        date = str(year) + '-' + str(month) + '-' + str(day)
                        link_dates.append(date)
                        search_tags.append(tag)
                    print('Links from {}/{}: {}'.format(day, month, len(all_links)))

d = {'dateYMD':link_dates, 'search_tag':search_tags, 'url': all_links, }
df = pd.DataFrame(data=d)
df.to_csv('links.csv')
