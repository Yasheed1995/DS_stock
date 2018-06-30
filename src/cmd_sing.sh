# all/single
# LSTM/GRU/simpleRNN
# window size
# $$: model model_type hidden_size stock_num 
# model name: (stock)_(m_type)_(hidsiz)_(all/sig)

for stock in 2330 2317 2409 2454 1301
do
	./run.sh ${stock}_RF_lstm_100_sing RF_lstm 100 ${stock}
	./run.sh ${stock}_RF_gru_100_sing RF_gru 100 ${stock}
	./run.sh ${stock}_RT_lstm_100_sing RT_lstm 100 ${stock}
	./run.sh ${stock}_RT_gru_100_sing RT_gru 100 ${stock}
	
	./run.sh ${stock}_RF_lstm_50_sing RF_lstm 50 ${stock}
	./run.sh ${stock}_RF_gru_50_sing RF_gru 50 ${stock}
	./run.sh ${stock}_RT_lstm_50_sing RT_lstm 50 ${stock}
	./run.sh ${stock}_RT_gru_50_sing RT_gru 50 ${stock}
 
 	./run.sh ${stock}_RF_lstm_20_sing RF_lstm 20 ${stock}
	./run.sh ${stock}_RF_gru_20_sing RF_gru 20 ${stock}
	./run.sh ${stock}_RT_lstm_20_sing RT_lstm 20 ${stock}
	./run.sh ${stock}_RT_gru_20_sing RT_gru 20 ${stock}
	
	./run.sh ${stock}_RF_lstm_200_sing RF_lstm 200 ${stock}
	./run.sh ${stock}_RF_gru_200_sing RF_gru 200 ${stock}
	./run.sh ${stock}_RT_lstm_200_sing RT_lstm 200 ${stock}
	./run.sh ${stock}_RT_gru_200_sing RT_gru 200 ${stock}
	
	./run.sh ${stock}_RF_lstm_300_sing RF_lstm 300 ${stock}
	./run.sh ${stock}_RF_gru_300_sing RF_gru 300 ${stock}
	./run.sh ${stock}_RT_lstm_300_sing RT_lstm 300 ${stock}
	./run.sh ${stock}_RT_gru_300_sing RT_gru 300 ${stock}
done

