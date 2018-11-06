import pandas as pd
import numpy as np
august_data = pd.read_csv('medium_data_august2018.csv')
new_data = pd.read_csv('medium_data_jan_may_2018.csv')

full_data = pd.concat([new_data, august_data])

print('aug:', august_data.describe())
print('new', new_data.describe())
print('full:, ', full_data.describe())

blogs = full_data['text_content'].values
blog = blogs[2]
blogWords = blog.split()
print(blogWords)
blogNum = print('blog num', len(blogWords))

numWords = [len(x.split()) for x in full_data['text_content'].values]
print(numWords)
print(int(np.mean(numWords)))


full_data.to_csv('medium_data_Jan_to_August_2018.csv')
