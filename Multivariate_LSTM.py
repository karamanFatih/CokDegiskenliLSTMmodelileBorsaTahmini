#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 21:47:27 2020

@author: fatih
"""
import sys
print(sys.version)
#Bağımlılıkları içe aktarma
import numpy as np
np.random.seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras import optimizers
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import datetime as dt
import time
plt.style.use('ggplot')

# veri setini yukleme
url = 'multiLSTMdataset.csv'
df = pd.read_csv(url,parse_dates = True,index_col=0)
df.tail()

# Korelasyon matrisi
df.corr()['Close']
print(df.corr()['Close'])
print(df.describe().Volume) 
df.drop(df[df['Volume']==0].index, inplace = True) #Hacim değeri 0 olan satır bırakma
# Erken bir durma ayarlama
earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=80,  verbose=1, mode='min')
callbacks_list = [earlystop]
#Modeli oluşturulmasi ve eğitilmesi
def fit_model(train,val,timesteps,hl,lr,batch,epochs):
    X_train = []
    Y_train = []
    X_val = []
    Y_val = []
  
    # egitim verileri için döngü
    for i in range(timesteps,train.shape[0]):
        X_train.append(train[i-timesteps:i])
        Y_train.append(train[i][0])
    X_train,Y_train = np.array(X_train),np.array(Y_train)
  
    # gercek verileri için döngü
    for i in range(timesteps,val.shape[0]):
        X_val.append(val[i-timesteps:i])
        Y_val.append(val[i][0])
    X_val,Y_val = np.array(X_val),np.array(Y_val)
    
    # Modele Katman Ekleme
    model = Sequential()
    model.add(LSTM(X_train.shape[2],input_shape = (X_train.shape[1],X_train.shape[2]),return_sequences = True,
                   activation = 'relu'))
    for i in range(len(hl)-1):        
        model.add(LSTM(hl[i],activation = 'relu',return_sequences = True))
    model.add(LSTM(hl[-1],activation = 'relu'))
    model.add(Dense(1))
    model.compile(optimizer = optimizers.Adam(lr = lr), loss = 'mean_squared_error')
    
  
    # Verilerin eğitimi
    history = model.fit(X_train,Y_train,epochs = epochs,batch_size = batch,validation_data = (X_val, Y_val),verbose = 0,
                        shuffle = False, callbacks=callbacks_list)
    model.reset_states()
    return model, history.history['loss'], history.history['val_loss']
# Modelin değerlendirilmesi
def evaluate_model(model,test,timesteps):
    X_test = []
    Y_test = []

    # Verileri test etmek için döngü
    for i in range(timesteps,test.shape[0]):
        X_test.append(test[i-timesteps:i])
        Y_test.append(test[i][0])
    X_test,Y_test = np.array(X_test),np.array(Y_test)
      
    # Tahmin  !!!!
    Y_hat = model.predict(X_test)
    mse = mean_squared_error(Y_test,Y_hat)
    rmse = sqrt(mse)
    r = r2_score(Y_test,Y_hat)
    return mse, rmse, r, Y_test, Y_hat
# Tahminleri çizmek
def plot_data(Y_test,Y_hat):
    plt.plot(Y_test,c = 'r')
    plt.plot(Y_hat,c = 'y')
    plt.xlabel('Gun')
    plt.ylabel('Fiyatlar(Olceklendirilmis haliyle)')
    plt.title('Cok Degiskenli LSTM Model Kullanarak Borsa Tahmini ')
    plt.legend(['Gercek Fiyat','Tahmin Edilen Fiyat'],loc = 'lower right')
    plt.show()    
# Eğitim hatalarını çizme
def plot_error(train_loss,val_loss):
    plt.plot(train_loss,c = 'r')
    plt.plot(val_loss,c = 'b')
    plt.ylabel('Kayip')
    plt.legend(['train','val'],loc = 'upper right')
    plt.show()    
# Serinin çıkarılması
series = df[['Close','High','Volume']] # Picking the series with high correlation
print(series.shape)
print(series.tail())    
# Train Val Test ayirimi
train_start = dt.date(2000,8,16)
train_end = dt.date(2007,12,31)
train_data = series.loc[train_start:train_end]

val_start = dt.date(2008,1,4)
val_end = dt.date(2009,12,31)
val_data = series.loc[val_start:val_end]

test_start = dt.date(2010,1,2)
test_end = dt.date(2011,11,30)
test_data = series.loc[test_start:test_end]

print(train_data.shape,val_data.shape,test_data.shape)
# normalleştirme
sc = MinMaxScaler()
train = sc.fit_transform(train_data)
val = sc.transform(val_data)
test = sc.transform(test_data)
print(train.shape,val.shape,test.shape)
timesteps = 120
hl = [40,35]
lr = 1e-3
batch_size = 64
num_epochs = 50
model,train_error,val_error = fit_model(train,val,timesteps,hl,lr,batch_size,num_epochs)
plot_error(train_error,val_error)
mse, rmse, r2_value,true,predicted = evaluate_model(model,test,timesteps)
print('MSE = {}'.format(mse))
print('RMSE = {}'.format(rmse))
print('R-Squared Score = {}'.format(r2_value))
plot_data(true,predicted)
# Save a model
#model.save('MV3-LSTM_50_[40,35]_1e-3_64.h5')
#del model # Deletes the model
# Load a model
#model = load_model('MV3-LSTM_50_[40,35]_1e-3_64.h5')