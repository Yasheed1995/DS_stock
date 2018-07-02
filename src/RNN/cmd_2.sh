# all/single
# LSTM/GRU/simpleRNN
# window size
# $$: model model_type hidden_size stock_num 
# model name: (stock)_(m_type)_(hidsiz)_(all/sig)

for stock in 2330
do
	./class.sh ${stock}_RF_lstm_10_class RF_lstm 11 ${stock}
	./class.sh ${stock}_RF_gru_10_class RF_gru 11 ${stock}
	./class.sh ${stock}_RT_lstm_10_class RT_lstm 11 ${stock}
	./class.sh ${stock}_RT_gru_10_class RT_gru 11 ${stock}
	
done
