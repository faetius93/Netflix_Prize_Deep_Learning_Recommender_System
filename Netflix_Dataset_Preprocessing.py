import os
import pandas as pd
import numpy as np
import math
import re
import matplotlib.pyplot as plt

# useful CONSTANTS
BASE_DIR = '.'
DATASET_DIR = BASE_DIR + '/utility/'

DATA_1_TXT = 'combined_data_1.txt'
DATA_2_TXT = 'combined_data_2.txt'
DATA_3_TXT = 'combined_data_3.txt'
DATA_4_TXT = 'combined_data_4.txt'

DATA_CSV = 'trainingRatings.csv'

# Loading Data Files
df1 = pd.read_csv(
				os.path.join(DATASET_DIR, DATA_1_TXT),
				header=None,
				names=['customer_id', 'rating'],
				usecols=[0,1]
				)
				
df2 = pd.read_csv(
				os.path.join(DATASET_DIR, DATA_2_TXT),
				header=None,
				names=['customer_id', 'rating'],
				usecols=[0,1]
				)
				
df3 = pd.read_csv(
				os.path.join(DATASET_DIR, DATA_3_TXT),
				header=None,
				names=['customer_id', 'rating'],
				usecols=[0,1]
				)
				
df4 = pd.read_csv(
				os.path.join(DATASET_DIR, DATA_4_TXT),
				header=None,
				names=['customer_id', 'rating'],
				usecols=[0,1]
				)
				
df1['rating'] = df1['rating'].astype('float')
df2['rating'] = df2['rating'].astype('float')
df3['rating'] = df3['rating'].astype('float')
df4['rating'] = df4['rating'].astype('float')

# Print useful info about the Data
print('Dataset 1 shape: {}'.format(df1.shape))
print('Dataset 2 shape: {}'.format(df2.shape))
print('Dataset 3 shape: {}'.format(df3.shape))
print('Dataset 4 shape: {}'.format(df4.shape))

# Join the Data
df = df1
df = df.append(df2)
df = df.append(df3)
df = df.append(df4)

df.index = np.arange(0, len(df))

print('Full Dataset shape: {}'.format(df.shape))
print('\n --- Complete Dataset example ---')
print(df.iloc[::5000000, :],'\n')

# Data Viewing

p = df.groupby('rating')['rating'].agg(['count'])

movie_count = df.isnull().sum()[1]
customer_count = df['customer_id'].nunique() - movie_count
rating_count = df['customer_id'].count() - movie_count

ax = p.plot(kind = 'barh', legend = False, figsize = (15,10))
plt.title('Total pool: {:,} Movies, {:,} Customers, {:,} Ratings given'.format(movie_count, customer_count, rating_count), fontsize=20)
plt.axis('off')

for i in range(1,6):
    ax.text(p.iloc[i-1][0]/4, i-1, 'Rating {}: {:.0f}%'.format(i, p.iloc[i-1][0]*100 / p.sum()[0]), color = 'white', weight = 'bold')

plt.show()

# Data Cleaning

df_nan = pd.DataFrame(pd.isnull(df.rating))
df_nan = df_nan[df_nan['rating'] == True]
df_nan = df_nan.reset_index()

movie_np = []
movie_id = 1

for i,j in zip(df_nan['index'][1:], df_nan['index'][:-1]):
    temp = np.full((1,i-j-1), movie_id)
    movie_np = np.append(movie_np, temp)
    movie_id += 1

last_record = np.full((1,len(df) - df_nan.iloc[-1, 0] - 1),movie_id)
movie_np = np.append(movie_np, last_record)

print('Movie numpy: {}'.format(movie_np))
print('Length: {}'.format(len(movie_np)))

# remove those Movie ID rows
df = df[pd.notnull(df['rating'])]

df['movie_id'] = movie_np.astype(int)
df['customer_id'] = df['customer_id'].astype(int)

print('\n --- New Dataset examples ---')
print(df.iloc[::5000000, :])

# Delete the Outliers -> Better Performance

f = ['count','mean']

df_movie_summary = df.groupby('movie_id')['rating'].agg(f)
df_movie_summary.index = df_movie_summary.index.map(int)
movie_benchmark = round(df_movie_summary['count'].quantile(0.8), 0)
drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index

df_cust_summary = df.groupby('customer_id')['rating'].agg(f)
df_cust_summary.index = df_cust_summary.index.map(int)
customer_benchmark = round(df_cust_summary['count'].quantile(0.8), 0)
drop_customer_list = df_cust_summary[df_cust_summary['count'] < customer_benchmark].index

print('Movie minimum times of review: {}'.format(movie_benchmark))
print('Customer minimum times of review: {}'.format(customer_benchmark))

print('Original Shape: {}'.format(df.shape))

df = df[~df['movie_id'].isin(drop_movie_list)]
df = df[~df['customer_id'].isin(drop_customer_list)]

print('After Trim Shape: {}'.format(df.shape))

df['customer_emb_id'] = df['customer_id'] - 1
df['movie_emb_id'] = df['movie_id'] - 1

print('\n --- After Trim Data Examples ---')
print(df.iloc[::5000000, :])

print('\nStarting .csv file')

df.to_csv(
		DATA_CSV,
		sep=',',
		header=True,
		index=False,
		columns=['customer_id', 'movie_id', 'rating', 'customer_emb_id', 'movie_emb_id']
		)

print('Saved to ', DATA_CSV)