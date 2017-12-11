import numpy as np
from keras.layers import Embedding, Reshape, Merge, Dropout, Dense, Concatenate
from keras.models import Sequential
		
class MovielensDNN(Sequential):
	
	def __init__(self, n_users, m_items, k_factors, p_dropout=0.1, **kwargs):
		
		L = Sequential()
		L.add(Embedding(n_users, k_factors, input_length=1))
		L.add(Reshape((k_factors,)))
		
		R = Sequential()
		R.add(Embedding(m_items, k_factors, input_length=1))
		R.add(Reshape((k_factors,)))
		
		super(MovielensDNN, self).__init__(**kwargs)
		self.add(Merge([L, R], mode='concat'))
		self.add(Dropout(p_dropout))
		# Hidden Layer:
		self.add(Dense(k_factors, activation='relu'))
		self.add(Dropout(p_dropout))
		self.add(Dense(k_factors, activation='relu'))
		self.add(Dropout(p_dropout))
		# Output:
		self.add(Dense(1, activation='linear'))
		
	def rate(self, user_id, item_id):
		return self.predict([np.array([user_id]), np.array([item_id])])[0][0]

class NetflixDNN(Sequential):
	
	def __init__(self, n_users, m_items, k_factors, p_dropout=0.1, **kwargs):
		
		L = Sequential()
		L.add(Embedding(n_users, k_factors, input_length=1))
		L.add(Reshape((k_factors,)))
		
		R = Sequential()
		R.add(Embedding(m_items, k_factors, input_length=1))
		R.add(Reshape((k_factors,)))
		
		super(NetflixDNN, self).__init__(**kwargs)
		self.add(Merge([L, R], mode='concat'))
		self.add(Dropout(p_dropout))
		# Hidden Layer:
		self.add(Dense(k_factors, activation='relu'))
		self.add(Dropout(p_dropout))
		self.add(Dense(k_factors, activation='sigmoid'))
		self.add(Dropout(p_dropout))
		# Output:
		self.add(Dense(1, activation='linear'))
		
	def rate(self, user_id, item_id):
		return self.predict([np.array([user_id]), np.array([item_id])])[0][0]