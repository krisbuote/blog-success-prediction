# README #

# CLAPPIE #

### What is this repository for? ###
+ Scraping data from medium.com
+ Predicting blog/article success, as measured by "claps" on Medium.com

### How do I gather data? ###
+ Open /data_acquirer/
+ Run get_links.py with the parameters you're interested in (tags and dates)
+ Using the links.csv that is created, scrape them with scrape_n_save.py
+ You will now have a data .csv file inside /data/

### How do I prepare data? ###
+ Open /data/
+ Run preprocess_articles.py to clean the data (i.e. make all lowercase, remove punctuation, etc.)

### How do I make predictions? ###
+ Open /model/ 
+ *Warning: The models are all half-baked and very messy (11/6/2018)*
+ Run one of the models, or make your own. TODO suggestion: use facebook's [fastText](https://fasttext.cc/)
