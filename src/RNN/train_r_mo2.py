from __future__ import print_function
import numpy as np
from keras.models import Model, load_model, Sequential
from keras.layers import Embedding, LSTM, GRU, Dense, Dropout, Bidirectional, TimeDistributed, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
from keras.utils import np_utils
import pickle
import argparse
import _pickle as pk
import readline
from keras import regularizers
from keras.layers import Input
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
import keras.backend.tensorflow_backend as K
import tensorflow as tf
import pandas as pd
import os
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error, confusion_matrix
from sklearn.preprocessing import StandardScaler
from plot import plot_conf_matrix
import matplotlib.pyplot as plt


os.environ['CUDA_VISIBLE_DEVICE'] = '1'

parser = argparse.ArgumentParser(description='Stock prediction')
parser.add_argument('--model')
parser.add_argument('--model_type', choices=['RT_lstm', 'RT_gru', 'RF_lstm', 'RF_gru'])
parser.add_argument('--action', choices=['train','test','class'])

'''
parser.add_argument('--train_x_path', default='data/trainX_5.npy',type=str)
parser.add_argument('--train_y_path', default='data/trainY_5.npy',type=str)
parser.add_argument('--test_x_path', default= 'data/testX_5.npy' ,type=str)
parser.add_argument('--test_y_path', default= 'data/testY_5.npy' , type=str)
'''
parser.add_argument('--window', default='20', type=int)

parser.add_argument('--save-model_path', default='save/model.h5', type=str)
parser.add_argument('--save_history_path', default='save/history.npy',
        type=str)

# training argument
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--nb_epoch', default=20, type=int)
parser.add_argument('--val_ratio', default=0.1, type=float)
parser.add_argument('--gpu_fraction', default=0.8, type=float)
parser.add_argument('--max_length', default=7,type=int)

# model parameter
parser.add_argument('--loss_function', default='mse')
parser.add_argument('--cell', default='LSTM', choices=['LSTM','GRU'])
parser.add_argument('-hid_siz', '--hidden_size', default=512, type=int)
parser.add_argument('--dropout_rate', default=0.3, type=float)
parser.add_argument('-lr','--learning_rate', default=0.001,type=float)
parser.add_argument('--threshold', default=0.1,type=float)

# for testing
parser.add_argument('--test_y', dest='test_y', type=str, default='npy/1.npy')

# output path for your prediction
parser.add_argument('--result_path', default='result.csv')

# put model in the same directory
parser.add_argument('--load_model', default = None)
parser.add_argument('--save_dir', default = 'model/')

# for class
parser.add_argument('--index', default=0, type=int)
parser.add_argument('--bin_size', default=5, type=int)

args = parser.parse_args()

mode = args.action

def RT_lstm(args):
    model = Sequential()
    model.add(LSTM(args.hidden_size, input_shape=(int(args.window),4), return_sequences=True))
    model.add(LSTM(args.hidden_size, return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.add(Flatten())
    if mode == 'class':
        model.add(Dense(args.bin_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                metrics=['acc'])
        
    else:
        model.add(Dense(5,activation='linear'))
        model.add(Dense(1,activation='linear'))
        model.compile(loss=args.loss_function, optimizer="adam")
    
    return model
    
def RT_gru(args):
    model = Sequential()
    model.add(GRU(args.hidden_size, input_shape=(int(args.window),4), return_sequences=True))
    model.add(Dropout(args.dropout_rate))
    model.add(GRU(args.hidden_size, return_sequences=True))
    model.add(Dropout(args.dropout_rate))
    model.add(TimeDistributed(Dense(1)))
    model.add(Flatten())
    if mode == 'class':
        model.add(Dense(args.bin_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                metrics=['acc'])

    else:
        model.add(Dense(5,activation='linear'))
        model.add(Dense(1,activation='linear'))
        model.compile(loss=args.loss_function, optimizer='adam')
    
    return model
    
def RF_lstm(args):
    model = Sequential()
    model.add(LSTM(args.hidden_size, return_sequences=True,input_shape=(int(args.window),4)))
    model.add(Dropout(args.dropout_rate))
    model.add(LSTM(args.hidden_size))
    model.add(Dropout(args.dropout_rate))
    
    if mode == 'class':
        model.add(Dense(args.bin_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                metrics=['acc'])

    else:
        model.add(Dense(1, activation='linear'))
        model.compile(loss=args.loss_function, optimizer="adam")

    return model
    
def RF_gru(args):
    model = Sequential()
    model.add(GRU(args.hidden_size, return_sequences=True,input_shape=(int(args.window),4)))
    model.add(Dropout(args.dropout_rate))
    model.add(GRU(args.hidden_size))
    model.add(Dropout(args.dropout_rate))
    if mode == 'class':
        model.add(Dense(args.bin_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                metrics=['acc'])
                
    else:
        model.add(Dense(1, activation='linear'))
        model.compile(loss=args.loss_function, optimizer="adam")

    return model
    

def main():

    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    set_session(tf.Session(config=config))

    save_path = os.path.join(args.save_dir,args.model)
    
    if args.load_model is not None:
        load_path = os.path.join(args.save_dir,args.load_model)

    #####read data#####
    #dm = DataManager(args.window, args.index)
    #dm.add_data('data/data.csv')
    #data, y_open, y_close, y_high, y_low = dm.get_data()
    data = np.load('data/x_2330.npy').astype(float)
    y_open = np.load('data/y_open_2330.npy').astype(float).reshape(-1,1)
    y_close = np.load('data/y_close_2330.npy').astype(float).reshape(-1,1)
    y_high = np.load('data/y_high_2330.npy').astype(float).reshape(-1,1)
    y_low = np.load('data/y_low_2330.npy').astype(float).reshape(-1,1)
    y = np.concatenate((y_close, y_high, y_low, y_open), axis = 1)
    print('concate Y shape:', y.shape)
    
    data = data.astype(float)
    y = y.astype(float)
    x_train, x_test, y_train, y_test = train_test_split(data[:-61, :, :], y[:-61],
            test_size=0.01)

    x_last60 = data[-61:-1, :, :]
    y_last60 = y_close[-61:-1]
   
    #####preprocess data#####
    x_train1 = x_train[:, :, 0].reshape(-1, args.window, 1)
    x_train2 = x_train[:, :, 2:5].reshape(-1, args.window, 3)
    x_train1 = np.concatenate((x_train1, x_train2), axis = 2)
        
    mean_x = np.mean(x_train1, axis = 0)
    std_x = np.std(x_train1, axis = 0)
    x_1 = (x_train1 - mean_x) / std_x
    
    x_test1 = x_test[:, :, 0].reshape(-1, args.window, 1)
    x_test2 = x_test[:, :, 2:5].reshape(-1, args.window,3)
    x_test1 = np.concatenate((x_test1, x_test2), axis = 2)
    
    x_last60_1 = x_last60[:, :, 0].reshape(-1, args.window, 1)
    x_last60_2 = x_last60[:, :, 2:5].reshape(-1, args.window, 3)
    x_last60_1 = np.concatenate((x_last60_1, x_last60_2), axis = 2)
    x_60 = (x_last60_1 - mean_x) / std_x  
   
    mean_y = np.mean(y_train, axis = 0)
    std_y = np.std(y_train, axis = 0)
    print('std y shape', std_y.shape)
    y_train_n = (y_train - mean_y) / std_y
    y_test_n = (y_test - mean_y) / std_y
    y_o = y_train_n[:, 0].reshape(-1)
    y_c = y_train_n[:, 1].reshape(-1)
    y_h = y_train_n[:, 2].reshape(-1)
    y_l = y_train_n[:, 3].reshape(-1)    

    # training
    if args.action == 'train':
       
        # initial model
        print ('initial model...')
        if args.model_type == 'RT_lstm':
            model = RT_lstm(args)
        elif args.model_type == 'RT_gru':
            model = RT_gru(args)
        elif args.model_type == 'RF_lstm':
            model = RF_lstm(args)
        elif args.model_type == 'RF_gru':
            model = RF_gru(args)
        print (model.summary())

        earlystopping = EarlyStopping(monitor='val_loss', patience = 3, verbose=1,
                mode='min')

        Y_train = [y_o, y_c, y_h, y_l]
        y_type = ['open', 'close', 'high', 'low']
        for i in range(4):
            save_path1 = os.path.join(save_path,'model_{}.h5'.format(y_type[i]))
            print('=================================')
            print(save_path1)
            print('=================================')
            checkpoint = ModelCheckpoint(filepath=save_path1,
                                         verbose=1,
                                         save_best_only=True,
                                         monitor='val_loss',
                                         mode='min' )
            
            history = model.fit(x_1, Y_train[i],
                                validation_split = 0.1,
                                epochs=args.nb_epoch,
                                batch_size=args.batch_size,
                                callbacks=[checkpoint, earlystopping] )
    
            dict_history = pd.DataFrame(history.history)
            
            print ('saving history in: ' + args.save_history_path + '...')
            his_path = '{}_{}.csv'.format(args.save_history_path, y_type[i])
            dict_history.to_csv(his_path)
   
    # testing
    elif args.action == 'test' :
        print ('testing ...')
 
        x_t = x_60
        y_type = ['open', 'close', 'high', 'low']
        for i in range(4):
            load_path1 = os.path.join(load_path, 'model_{}.h5'.format(y_type[i]))
            model = load_model(load_path1)
    
            p = model.predict(x_t[-61:-1, :, :], batch_size=args.batch_size, verbose=1)
            if i == 0:
                open_p = p
            elif i == 1:
                close_p = p
            elif i == 2:
                high_p = p    
            elif i == 3:
                low_p = p 
                
        predict_all = np.concatenate((close_p, high_p, low_p, open_p), axis = 1)
        y2 = [] #x_60[:20,:,:]
        for i in range(60 - args.window):
            y2.append(predict_all[i:i+args.window])
        predict_seq = np.concatenate((x_60[:args.window,:,:], np.array(y2)), axis = 0)
        print('Y test:', predict_seq.shape)
        
        #predict close by perdicted values
        load_path2 = os.path.join(load_path, 'model_close.h5')
        model = load_model(load_path2)
        test_y = model.predict(predict_seq, batch_size=args.batch_size, verbose=1)
        print (test_y[:10])
        test_y = test_y * std_y[0]
        test_y = test_y + mean_y[0]
        #print (x_60[:10])
        print (y_last60[:10])
        print (test_y[:10])

        np.save(args.test_y, test_y)
           
        #draw fig
        plt.plot(test_y, color='red', label='Prediction')
        plt.plot(y_last60, color='blue', label='Ground Truth')
        plt.legend(loc='upper right')
        plt.xlabel("Time Period")
        plt.ylabel("Stock Price")
        plt.title("{} ({})".format(args.model_type, args.loss_function))
        figdir = 'fig/'
        figpath = os.path.join(figdir,
                'long_term{}-{}-{}-win{}.pdf'.format(args.model_type, args.loss_function, args.hidden_size, args.window))
        plt.savefig(figpath)
        print('fig save!!! ', 'long_term{}-{}-{}-win{}.pdf'.format(args.model_type, args.loss_function, args.hidden_size, args.window))
         

if __name__ == '__main__':
    main()
