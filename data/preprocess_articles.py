import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
'''
This script cleans the articles scraped from medium. It converts them to lowercase and removes non alphabetical chars
'''

filename = 'medium_data_Jan_to_August_2018'
data = pd.read_csv(filename + '.csv')

# Collect the textual information
titles = np.squeeze(data['title'].values)
blogs = np.squeeze(data['text_content'].values)
bio = np.squeeze(data['author_bio'].values)
tag_words = np.squeeze(data['tag_words'].values)

# Add the number of words per blog into dataframe
numWords = [len(x.split()) for x in data['text_content'].values]
data['num_words'] = numWords


text_to_clean = [titles, blogs, bio, tag_words]

# turn data into clean tokens
def textCleaner(text):
    no_punct = re.sub(r"[,.;@#?!&$]+\ *", " ", text) # Replace punctuation with spaces
    clean_string = re.sub("[^a-zA-Z]",  # Remove Anything except a..z and A..Z
           " ",  # replaced with a space
           no_punct)  # in this string
    tokens = clean_string.split()
    tokens = [word.lower() for word in tokens]
    return tokens

cleaned_titles = []
cleaned_blogs = []
cleaned_bio = []
cleaned_tag_words = []

cleaned_texts = [cleaned_titles, cleaned_blogs, cleaned_bio, cleaned_tag_words]

num_blogs = len(blogs)

print('Cleaning text...\n')

for j in range(len(text_to_clean)):
    unclean_text = text_to_clean[j]
    clean_text = cleaned_texts[j]

    for i in range(num_blogs):
        # clean the text
        cleaned_tokens = textCleaner(unclean_text[i])
        cleaned_string = ' '.join(cleaned_tokens)
        clean_text.append(cleaned_string)


print('Text is scrubbed! Here is a sample: \n ')
rando = np.random.randint(num_blogs)
print('Random title: ', cleaned_titles[rando])
print('Random blog: ', cleaned_blogs[rando])


# Update DF and save
data['text_content'] = cleaned_blogs
data['title'] = cleaned_titles
data['author_bio'] = cleaned_bio
data['tag_words'] = cleaned_tag_words

data.to_csv('cleaned_' + filename + '.csv', index=False)
