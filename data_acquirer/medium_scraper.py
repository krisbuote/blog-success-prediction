'''
Author: Kristopher Buote
Date: 9/15/2018
Work in Progress

This scripts scrapes the heck out blogs/articles hosted on medium.com.
A dictionary is returned containing:

    data = {'title': title,
            'author_bio': bio,
            'claps': claps,
            'num_images': numImages,
            'num_links': numLinks,
            'num_tags': numTags,
            'num_lists': numLists,
            'num_h3': numH3,
            'num_h4': numH4,
            'num_blockquotes': numBlockQuotes,
            'num_pullquotes': numPullQuotes,
            'num_sections': numSections,
            'text_content': text_content,
            'tag_words': tags,
            'title_subBlog_mainBlog': title_subBlog_Blog}
'''

import requests
from bs4 import BeautifulSoup
import time

def scrape(_url):
    # parse the html using beautiful soup and store in variable `soup`
    page = requests.get(_url)
    data = page.text
    soup = BeautifulSoup(data, 'html.parser')

    '''EXTRACT THE TITLE / SUBBLOG / BLOG STRING
    note: this string contains 'title' (hyphen) 'subblog' (hyphen) 'blog'. The sublog and blog elements are optional'''
    # TODO split the title_subBlog_Blog string if appropriate. Based on hosting domain medium.com vs. towardsdatascience etc

    title_subBlog_Blog = str(soup.title.string)

    ''' EXTRACT THE DATE'''
    blog_date = soup.find('time').text

    '''EXTRACT THE NUMBER OF LINKS and IMAGES and TAGS and OTHER'''
    # TODO Possible inclusions: Tweets/soundcloud/other embedded media, % bold/italic content, other markdown

    article_full = soup.find('div', class_="postArticle-content js-postField js-notesSource js-trackedPost")

    numImages = str(article_full).count('<img ') # Number of images
    numLinks = str(article_full).count(' href=') # Number of links
    numH3 = str(article_full).count('<h3 ') # Number of h3 headings
    numH4 = str(article_full).count('<h4 ') # Number of h4 headings
    numBlockQuotes = str(article_full).count('<blockquote ') # Number of blockquotes
    numPullQuotes = len(soup.find_all('em', class_='markup--em markup--pullquote-em')) # Number of pullquotes
    numTags = len(soup.find('ul', class_="tags tags--postTags tags--borderless"))  # Number of tags
    numLists = len(soup.find_all('ul', class_="postList")) # Number of lists (bullet points)
    numSections = len(soup.find_all("div", class_="section-content")) # Number of sections in text

    # If author bio exists, grab it. Else return ''
    if soup.find('div', class_='ui-caption ui-xs-clamp2 postMetaInline') is not None:
        bio = str(soup.find('div', class_='ui-caption ui-xs-clamp2 postMetaInline').text)
    else:
        bio = 'none'

    # Check for title heading, otherwise title will be in title_subBlog_Blog
    if article_full.find('h1') is not None:
        title = article_full.find('h1').text
    else:
        title = 'none'

    '''EXTRACT THE TAG WORDS'''
    tags = str()
    for tag in soup.find('ul', class_="tags tags--postTags tags--borderless"):
        tags += str(tag.text) + ', '

    '''EXTRACT THE CLAPS ! There's probably a more elegant way to handle claps...'''
    claps = ''

    # This for loop will return empty string if there are no claps yet
    for button in soup.find_all("button"):
        if button.get('data-action-value'): # If the button element contains the # of claps
            claps = button.text
            # TODO break out of this when claps found. It returns empty when I break?

    if claps == '':
        claps = '0'

    # Convert to int
    if 'K' in claps:
        claps = int(float(claps.split('K')[0])*1000)
    else:
        claps = int(claps)

    '''EXTRACT THE TEXT ! This can return the full text_content or the separated sections'''
    ''' Method 1: Takes full text. Downside: jams together paragraph ends so you end up with merged words like finishStart'''
    # text_content = str()
    # sc_time_start = time.time()
    # for sc_div in soup.find_all("div", class_="section-content"):
    #     section_content = sc_div.text
    #     text_content +=  section_content +' \n '
    # sc_end_time = time.time()
    # print('SC time:', sc_end_time - sc_time_start)
    #

    ''' Method 2: look inside all possible text containing tags and combine the text inside. SLower, captures all'''
    # TODO ensure these are all the possible tags containing text
    look_in = ['html', 'head','body','title','p','h1','h2', 'h3','h4','pre','blockquote','i','b','tt',
               'em','strong','cite','ol','ul','li','dl','dt','dd','a','table','tr','td']

    text_content = str()
    for htmltag in article_full.find_all(look_in):
        text_content += str(htmltag.text) + ' '


    '''COLLECT THE DATA INTO A NICE DICTIONARY'''
    data = {'blog_date': blog_date,
            'title': title,
            'author_bio': bio,
            'claps': claps,
            'num_images': numImages,
            'num_links': numLinks,
            'num_tags': numTags,
            'num_lists': numLists,
            'num_h3': numH3,
            'num_h4': numH4,
            'num_blockquotes': numBlockQuotes,
            'num_pullquotes': numPullQuotes,
            'num_sections': numSections,
            'title_subBlog_mainBlog': title_subBlog_Blog,
            'tag_words': tags,
            'text_content': text_content
            }

    return data


''' Test it out '''
# data = scrape('https://medium.com/tensorist/classifying-yelp-reviews-using-nltk-and-scikit-learn-c58e71e962d9')
# print('data:', data)






