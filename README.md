# Learning to Make Stock Price Predictions

## RNN-based Regression Command (path: src/RNN)
```
python3 train_r_mo1.py [options as below]
```

## RNN-based Classification Command (path: src/RNN)
```
python3 train_r_mo.py [options as below]
```

## Data
``` Put data.csv under data directory.```

### RNN based Regression & Classification Command Usage
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

## Output
```
Confusion matrix: saved in fig directory.
    
Prediction curves: saved in fig directory.
    
Numerical results: saved in log directory.
    
Trained models: saved in models directory.
    
You may find the corresponding outputs according to the model name.
```


## Other Machine Learning Approaches (path: src/ML)

## Data
``` Put data.csv under data directory.```

## Command
```python run.py [ML Model] [Bin Size]```

## Arguments
#### ML Model
```
LinR: Linean Regression

LogR: Logistic Regression

SVM: Support Vector Machine

D-Tree: Decision Tree

RF: Random Forest

NN: Neural Network (Multiple Layers Perceptron)

KMeans: K-Means Clustering

Bayes: Bayesian Classifier
```

#### Bin Size
```
A positive integer which equals to the number of bins plus 1.
```

## Example Command
```
python run.py LinR
    
python run.py LogR 11 (this will perform 10 classes classification)
    
python run.py SVM 11 (this will perform 10 classes classification)

For more commands, please refer to run.sh.
```

## Output
```
Confusion matrix: saved in fig directory.
    
Prediction curves: saved in fig directory.
    
Numerical results: saved in log directory.
    
Trained models: saved in models directory.
    
You may find the corresponding outputs according to the model name.
```
