import numpy as np
import pandas as pd 
from stockstats import wrap

#OHLC(AC)V
class TimeSeriesDataset:
    def __init__(self,data_raw):
        self.data_raw = data_raw
        self.process_data()

    def sample(self,n_samples,window_size=1):
        self.process_data(n_samples=n_samples,window_size=window_size)
        return self

    def process_data(self,n_samples: int=0, window_size: int=1):
        data = self.data_raw.iloc[:,[0,1,2,3]].copy()
        
        #data_indicators = self.data_raw.iloc[:,[0,1,2,3]].copy()
        stockdf = wrap(self.data_raw.iloc[:,[0,1,2,3]])
        data['21sma'] = stockdf['close_21_sma']
        data['50sma'] = stockdf['close_50_sma']
        data['100sma'] = stockdf['close_100_sma']
        data['200sma'] = stockdf['close_200_sma']
        data['rsi'] = stockdf['rsi'] * self.data_raw['Open']
        data['macd'] = stockdf['macd']
        data['macds'] = stockdf['macds']

        data = data.to_numpy()
        data = data[200:]
        n_samples = min(len(data),n_samples)    
        # shorten dataset
        if n_samples != 0:
            data = data[-n_samples:]

        data_norm = np.zeros(data.shape)
        for i in range(len(data)):
            data_norm[i] = data[i,:]/data[i,0]
        data_sliced = np.zeros((data_norm.shape[0]-window_size,window_size,data_norm.shape[1]))
        for i in range(len(data_norm)-window_size):
            data_sliced[i] = data_norm[i:i+window_size,:]
        # extract specific columns
        # columns_to_extract = range(1,12)
        data_sliced = data_sliced[:,:,1:]
        targets = data_norm[window_size:,3]
        test_index = -4
        targets = np.where(np.abs(targets-1)>=0.01,np.sign(targets-1),0)
        targets_pre = np.sign(targets-1)
        self.targets_pre = targets_pre
        occurences = np.bincount((targets+1).astype(int))
        self.occurences = occurences
        split = 0.90
        split_index = int(len(data_sliced)*split)
        self.y_train, self.y_val = np.split(targets,indices_or_sections=[split_index])
        self.y_train_pre, self.y_val_pre = np.split(targets_pre,indices_or_sections=[split_index])
        data_train, data_val = np.split(data_sliced,indices_or_sections=[split_index])
        self.train = data_train.reshape((data_train.shape[0],data_train.shape[1]*data_train.shape[2]))
        self.val = data_val.reshape((data_val.shape[0],data_val.shape[1]*data_val.shape[2]))