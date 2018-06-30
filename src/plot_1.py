import matplotlib as mpl
mpl.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

y_last60 = np.load("npy/y_last60.npy")
rf_gru_all = np.load("npy/2330_RF_gru_300_all.npy")
rt_gru_all = np.load("npy/2330_RT_gru_300_all.npy")
rf_lstm_all = np.load("npy/2330_RF_lstm_300_all.npy")
rt_lstm_all = np.load("npy/2330_RT_lstm_300_all.npy")

rf_gru_one = np.load("npy/2330_RF_gru_200_sing.npy")
rt_gru_one = np.load("npy/2330_RT_gru_200_sing.npy")
rf_lstm_one = np.load("npy/2330_RF_lstm_200_sing.npy")
rt_lstm_one = np.load("npy/2330_RT_lstm_200_sing.npy")

plt.plot(y_last60, label='Ground Truth')
plt.plot(rf_gru_all, label = 'RF_GRU (all)')
plt.plot(rf_lstm_all, label = 'RF_LSTM (all)')
plt.plot(rt_gru_all, label = 'RT_GRU (all)')
plt.plot(rt_lstm_all, label = 'RT_LSTM (all)')

plt.plot(rf_gru_one, label = 'RF_GRU (one)')
plt.plot(rf_lstm_one, label = 'RF_LSTM (one)')
plt.plot(rt_gru_one, label = 'RT_GRU (one)')
plt.plot(rt_lstm_one, label = 'RT_LSTM (one)')

plt.legend(loc='upper right')
plt.xlabel("Time Period")
plt.ylabel("Stock Price")

#figdir = 'last60/'
#figpath = os.path.join(figdir,
#		'{}-{}-dim-{}-sing.pdf'.format(args.index, args.model_type, args.hidden_size))
plt.savefig('last60/all.pdf')
print('fig save!!!')
 