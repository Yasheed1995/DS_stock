import os
import numpy as np
import pandas as pd

class DataManager:
    def __init__(self, stock_num=2330):
        self.data = {}
        self.stock_num = stock_num
    def add_data(self, data_path, with_label=True):
        print ('read data from %s...'%data_path)
        X,  Y = [],  []
        data = pd.read_csv(data_path)
        
        data['Date'] = data['Date'].apply(lambda x: (x[5:7]) if x[4] != '/'else x[5:7])
        data['Date'] = data['Date'].apply(lambda x: x[0] if x[1] == '/' else (x))
        data['Date'] = data['Date'].apply(lambda x: float(x))

        for stock_num in set(data['stock_symbol'].values):
            if stock_num == self.stock_num:
                X.append(data.loc[data['stock_symbol'] == stock_num].values[:-1])
                Y.append(data.loc[data['stock_symbol'] == stock_num]['Open'].values[1:])
        X = np.array(X)
        Y = np.array(Y)
        
        self.data['data'] = X
        self.data['label'] = Y

    def get_data(self, name):
        return self.data[name]

# testing
if __name__ == '__main__': 
    data = DataManager()
    data.add_data('data/data.csv')
    print(data)
    print(data.get_data('data').shape)
    print(data.get_data('label').shape)
    #print ('Shape of array[X, Y], contains 53 stocks, ', data.get_data('test').shape)
    #x = data.get_data('test')[0]
    #y = data.get_data('test')[1]
    #print(x[0])
    #print(y[0])
