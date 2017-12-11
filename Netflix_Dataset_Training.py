import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from Netflix_Recommender_System_DL import MovielensDNN
from Netflix_Recommender_System_DL import NetflixDNN

RATINGS_CSV = 'trainingRatingsSmall.csv'
MODEL_WEIGHTS_FILE = 'netflix_weights.h5'
K_FACTORS = 120
RNG_SEED = 1446557

ratings = pd.read_csv(
					RATINGS_CSV, 
					sep=',', 
					usecols=['customer_id', 'movie_id', 'rating', 'customer_emb_id', 'movie_emb_id']
					)
max_customer_id = ratings['customer_id'].drop_duplicates().max()
max_movie_id = ratings['movie_id'].drop_duplicates().max()

print(len(ratings), ' ratings loaded.')

shuffled_ratings = ratings.sample(frac=1., random_state=RNG_SEED)

Customers = shuffled_ratings['customer_emb_id'].values
Movies = shuffled_ratings['movie_emb_id'].values
Ratings = shuffled_ratings['rating'].values

print('Customers: ', Customers, ' , shape = ', Customers.shape)
print('Movies: ', Movies, ' , shape = ', Movies.shape)
print('Ratings: ', Ratings, ' , shape = ', Ratings.shape)

model = NetflixDNN(max_customer_id, max_movie_id, K_FACTORS)
model.compile(loss='mse', optimizer='sgd')

callbacks = [EarlyStopping('val_loss', patience=2), ModelCheckpoint(MODEL_WEIGHTS_FILE, save_best_only=True)]
history = model.fit([Customers, Movies], Ratings, epochs=10, validation_split=.1, verbose=2, callbacks=callbacks)

dataUtil = {
			'epoch': [ i + 1 for i in history.epoch ],
			'training': [ math.sqrt(loss) for loss in history.history['loss'] ],
			'validation': [ math.sqrt(loss) for loss in history.history['val_loss'] ]
			}

loss = pd.DataFrame(data=dataUtil)

print('loss dataframe: ', loss)

# summarize history for loss
plt.plot(loss.training)
plt.plot(loss.validation)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

min_val_loss, idx = min((val, idx) for (idx, val) in enumerate(history.history['val_loss']))

print('Minimum RMSE at epoch', '{:d}'.format(idx+1), '=', '{:.4f}'.format(math.sqrt(min_val_loss)))