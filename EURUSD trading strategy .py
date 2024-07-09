#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Grab Data
import yfinance as yf

#Usual Suspects
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns


# In[2]:


#EUR/USD price levels for the past 10 years
symbol = 'EURUSD=X'
raw = yf.download(symbol, start="2010-01-01", end="2024-06-28")['Adj Close']

data = pd.DataFrame(raw)


# In[3]:


data.head()


# In[4]:


#Calculate the returns and add it to the DataFrame
data['return'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))

#When the market direction is greater than 0 --> classify 1, less than 0 --> 0
data['direction'] = np.where(data['return'] > 0, 1, 0)

data.head()


# In[5]:


#Create 5 columns for each lag representing previous day's return
lags = 5

cols = []
for lag in range(1, lags + 1):
  col = f'lag_{lag}'
  data[col] = data['return'].shift(lag)
  cols.append(col)

data.dropna(inplace=True)


# In[6]:


data.round(4).tail()


# In[7]:


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop
import random


# In[8]:


pip install tensorflow


# In[9]:


optimizer = Adam(learning_rate=0.0001)

def set_seeds(seed=100):
  random.seed(seed)
  np.random.seed(seed)
  tf.random.set_seed(100)


# In[10]:


set_seeds()
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(lags,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[11]:


train_set, test_set = np.split(data, [int(.70 *len(data))])


# In[12]:


mu, std = train_set.mean(), train_set.std()


# In[13]:


# Normalizes the features data by Gaussian normalization
training_data_ = (train_set - mu) / std
test_data_ = (test_set - mu) / std


# In[14]:


get_ipython().run_cell_magic('time', '', "model.fit(train_set[cols],\n           train_set['direction'],\n           epochs=50, verbose=False,\n           validation_split=0.2, shuffle=False)\n")


# In[15]:


res = pd.DataFrame(model.history.history)


# In[16]:


# Accuracy of the model for training and validation in the training set
res[['accuracy', 'val_accuracy']].plot(figsize=(10,6), style='--')


# In[17]:


model.evaluate(training_data_[cols], train_set['direction'])


# In[18]:


# Creating Prediction of market direction
pred = np.where(model.predict(training_data_[cols]) > 0.5, 1, 0)


# In[19]:


pred


# In[20]:


pred[:30].flatten()


# In[21]:


# Transforming the predictions to long-short position; 1,-1
train_set['prediction'] = np.where(pred > 0, 1, -1)


# In[22]:


# Calculate strategy returns given the positions
train_set['strategy'] = (train_set['prediction'] * train_set['return'])


# In[23]:


train_set[['return', 'strategy']].sum().apply(np.exp)


# In[24]:


# Plots the strategy returns to the benchmark performance for the sample
train_set[['return', 'strategy']].cumsum().apply(np.exp).plot(figsize=(10,6))


# In[25]:


model.evaluate(test_data_[cols], test_set['direction'])


# In[26]:


pred = np.where(model.predict(test_data_[cols]) > 0.5, 1, 0)


# In[27]:


test_set['prediction'] = np.where(pred > 0, 1, -1)


# In[28]:


test_set['prediction'].value_counts()


# In[29]:


test_set['strategy'] = (test_set['prediction'] * test_set['return'])


# In[30]:


test_set[['return', 'strategy']].sum().apply(np.exp)


# In[31]:


test_set[['return', 'strategy']].cumsum().apply(np.exp).plot(figsize=(10,6))


# In[32]:


#Add momentum, volatitlity, & distance to the data_frame
data['momentum'] = data['return'].rolling(5).mean().shift(1)
data['volatility'] = data['return'].rolling(20).std().shift(1)
data['distance'] = (data['Adj Close'] - data['Adj Close'].rolling(50).mean()).shift(1)


# In[33]:


data.dropna(inplace=True)


# In[34]:


cols.extend(['momentum', 'volatility', 'distance'])


# In[35]:


# New DataFrame
data.round(4).tail()


# In[36]:


# Refit and train/test
train_data, test_data = np.split(data, [int(.70 *len(data))])
mu, std = train_data.mean(), train_data.std()


# In[37]:


training_data_ = (train_data - mu) / std
test_data_ = (test_data - mu) / std


# In[38]:


# Update Dense Layers to 32
set_seeds()
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(len(cols),)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[39]:


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input
import numpy as np

# Define the model
model = Sequential()
model.add(Input(shape=(784,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model with an optimizer
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Create some dummy data for demonstration purposes
X_train = np.random.random((1000, 784)).astype(np.float32)
y_train = np.random.randint(10, size=(1000,)).astype(np.int32)

# Fit the model
model.fit(X_train, y_train, epochs=10)


# In[40]:


print(X_train.shape)  # Should print (number_of_samples, 784)
print(y_train.shape)  # Should print (number_of_samples,)


# In[41]:


import matplotlib.pyplot as plt

# Assuming `history` is the returned History object from model.fit()
history = model.fit(X_train, y_train, epochs=10)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()


# In[42]:


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input

# Function to set seeds for reproducibility
def set_seeds(seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)

set_seeds()

# Define the model
model = Sequential()
model.add(Input(shape=(len(cols),)))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model with an optimizer
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

# To print the model summary
model.summary()


# In[43]:


get_ipython().run_cell_magic('time', '', "model.fit(training_data_[cols],\n           train_data['direction'],\n           epochs=25, verbose=False,\n           validation_split=0.2, shuffle=False)\n")


# In[44]:


model.evaluate(training_data_[cols], train_data['direction'])


# In[45]:


pred = np.where(model.predict(training_data_[cols]) > 0.5, 1, 0)


# In[46]:


train_data['prediction'] = np.where(pred > 0, 1, -1)


# In[47]:


train_data['strategy'] = (train_data['prediction'] * train_data['return'])


# In[48]:


train_data[['return', 'strategy']].sum().apply(np.exp)


# In[49]:


train_data[['return', 'strategy']].cumsum().apply(np.exp).plot(figsize=(10,6))


# In[50]:


model.evaluate(test_data_[cols], test_data['direction'])


# In[51]:


pred = np.where(model.predict(test_data_[cols]) > 0.5, 1, 0)


# In[52]:


test_data['prediction'] = np.where(pred > 0, 1, -1)


# In[53]:


test_set['prediction'].value_counts()


# In[54]:


test_set['strategy'] = (test_set['prediction'] * test_set['return'])


# In[55]:


test_set[['return', 'strategy']].sum().apply(np.exp)


# In[56]:


test_set[['return', 'strategy']].cumsum().apply(np.exp).plot(figsize=(10,6))


# In[ ]:




