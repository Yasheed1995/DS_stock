# all/single
# LSTM/GRU/simpleRNN
# window size
# $$: model model_type hidden_size stock_num 
# model name: (stock)_(m_type)_(hidsiz)_(all/sig)

for stock in 2330
do
	./run.sh ${stock}_RF_lstm_200_sing_mae RF_lstm 200 ${stock}
	./run.sh ${stock}_RF_gru_200_sing_mae RF_gru 200 ${stock}
	./run.sh ${stock}_RT_lstm_200_sing_mae RT_lstm 200 ${stock}
	./run.sh ${stock}_RT_gru_200_sing_mae RT_gru 200 ${stock}
	
	./run.sh ${stock}_RF_lstm_300_all_mae RF_lstm 300 ${stock}
	./run.sh ${stock}_RF_gru_300_all_mae RF_gru 300 ${stock}
	./run.sh ${stock}_RT_lstm_300_all_mae RT_lstm 300 ${stock}
	./run.sh ${stock}_RT_gru_300_all_mae RT_gru 300 ${stock}
done
