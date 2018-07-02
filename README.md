# datascience_final

# RNN-based Regression Command
```
python3 train_r_mo1.py
```

# RNN-based Classification Command
```
python3 train_r_mo.py
```

## RNN based Regression & Classification Command Usage
```
--model name			your model name

--model_type type		choose a model type

--action action			class, train or test

--window size			window size for data

--save_model_path path	        path to save the model

--save_history_path 		path to save the history

--batch_size size		specify the batch size

--nb_epoch epoch		specify the number of epoches

--gpu_fraction num		specify the gpu usage ratio

--loss_function func		specify the loss function

--cell cell			LSTM or GRU

--hidden_size size		hidden size of LSTM(or GRU)

--dropout_rate 			specify the dropout rate for LSTM

--test_y path			path to save the result npy

--result_path path		path to save the result csv

--load_model 			if testing, need to specify the model path

--index i			index of stock

--bin_size size			number of bins for classification	
```
For more commands, please refer to class.sh & run.sh.

# Outputs
The prediction curves and the confusion matrix are saved in fig directory.

You may find the results according to model name.

The accuracy, precision, recall, and f-measure for classification and MSE and MAE for regression are saved in log directory.

The trained models are saved in models directory.


