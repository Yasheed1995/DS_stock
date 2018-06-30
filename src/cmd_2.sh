# all/single
# LSTM/GRU/simpleRNN
# window size
# $$: model model_type hidden_size stock_num 
# model name: (stock)_(m_type)_(hidsiz)_(all/sig)

for stock in 2330
do
	./run.sh ${stock}_RF_lstm_20_class RF_lstm 10 ${stock}
	./run.sh ${stock}_RF_gru_20_class RF_gru 10 ${stock}
	./run.sh ${stock}_RT_lstm_20_class RT_lstm 10 ${stock}
	./run.sh ${stock}_RT_gru_20_class RT_gru 10 ${stock}
	
done
