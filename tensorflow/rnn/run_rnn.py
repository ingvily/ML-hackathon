from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from model_rnn import lstm_model, load_csvdata
import pandas as pd



TIMESTEPS = 10
BATCH_SIZE = 100
RNN_LAYERS = [{'steps': TIMESTEPS}]
DENSE_LAYERS = [10, 10]
TRAINING_STEPS = 100000
PRINT_STEPS = TRAINING_STEPS / 100
LOG_DIR = './ops_logs'



#rawdata = pd.read_csv("./train_scaled.csv")

#X, y = load_csvdata(rawdata, TIMESTEPS, seperate=False)


dateparse = lambda dates: pd.datetime.strptime(dates, '%d/%m/%Y %H:%M')
rawdata = pd.read_csv("./RealMarketPriceDataPT.csv", 
                   parse_dates={'timeline': ['date', '(UTC)']}, 
                   index_col='timeline', date_parser=dateparse)

X, y = load_csvdata(rawdata, TIMESTEPS, seperate=False)


#print X['test'][0][1]
print y


regressor = learn.TensorFlowEstimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS), 
                                      n_classes=0,
                                      verbose=1,  
                                      steps=TRAINING_STEPS, 
                                      optimizer='Adagrad',
                                      learning_rate=0.03, 
                                      batch_size=BATCH_SIZE)




validation_monitor = learn.monitors.ValidationMonitor(X['val'], y['val'],
                                                      every_n_steps=PRINT_STEPS,
                                                      early_stopping_rounds=1000)
print validation_monitor

regressor.fit(X['train'], y['train'], monitors=[validation_monitor])

predicted = regressor.predict(X['test'])