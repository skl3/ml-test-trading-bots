import matplotlib.pyplot as plt
import graphlab as gl
from __future__ import division
from datetime import datetime
from yahoo_finance import Share

# download historical prices of S&P 500 index
today = datetime.strftime(datetime.today(), "%Y-%m-%d")
stock = Share('^GSPC') # ^GSPC is the Yahoo finance symbol to refer S&P 500 index
# we gather historical quotes from 2001-01-01 up to today
hist_quotes = stock.get_historical('2001-01-01', today)
# here is how a row looks like
hist_quotes[0]
# example response
'''
{'Adj_Close': '2091.580078',
 'Close': '2091.580078',
 'Date': '2016-04-22',
 'High': '2094.320068',
 'Low': '2081.199951',
 'Open': '2091.48999',
 'Symbol': '%5eGSPC',
 'Volume': '3790580000'}
'''

l_date = []
l_open = []
l_high = []
l_low = []
l_close = []
l_volume = []
# reverse the list
hist_quotes.reverse()
for quotes in hist_quotes:
    l_date.append(quotes['Date'])
    l_open.append(float(quotes['Open']))
    l_high.append(float(quotes['High']))
    l_low.append(float(quotes['Low']))
    l_close.append(float(quotes['Close']))
    l_volume.append(int(quotes['Volume']))

qq = gl.SFrame({
    'datetime' : l_date, 
    'open' : l_open, 
    'high' : l_high, 
    'low' : l_low, 
    'close' : l_close, 
    'volume' : l_volume
})
# datetime is a string, so convert into datetime object
qq['datetime'] = qq['datetime'].apply(lambda x:datetime.strptime(x, '%Y-%m-%d'))

# just to check if data is sorted in ascending mode
qq.head(3)

qq.save(“SP500_daily.bin”)
# once data is saved, we can use the following instruction to retrieve it 
qq = gl.SFrame(“SP500_daily.bin/”)

# add the outcome variable, 1 if the trading session was positive (close>open), 0 otherwise
qq['outcome'] = qq.apply(lambda x: 1 if x['close'] > x['open'] else -1)
# we also need to add three new columns ‘ho’ ‘lo’ and ‘gain’
# they will be useful to backtest the model, later
qq['ho'] = qq['high'] - qq['open'] # distance between Highest and Opening price
qq['lo'] = qq['low'] - qq['open'] # distance between Lowest and Opening price
qq['gain'] = qq['close'] - qq['open']

ts = gl.TimeSeries(qq, index='datetime')
# add the outcome variable, 1 if the bar was positive (close>open), 0 otherwise
ts['outcome'] = ts.apply(lambda x: 1 if x['close'] > x['open'] else -1)

# GENERATE SOME LAGGED TIMESERIES
ts_1 = ts.shift(1) # by 1 day
ts_2 = ts.shift(2) # by 2 days
# ...etc....
# it's an arbitrary decision how many days of lag are needed to create a good forecaster, so
# everyone can experiment by his own decision

# add_features is a helper function, which is out of the scope of this article,
# and it returns a tuple with:
# ts: a timeseries object with, in addition to the already included columns, also lagged columns
# as well as some features added to train the model, as shown above with feat1 and feat2 examples
# l_features: a list with all features used to train Classifier models
# l_lr_features: a list all features used to train Linear Regression models

ts, l_features, l_lr_features = add_features(ts)

# add the gain column, for trading operations with LONG only positions. 
# The gain is the difference between Closing price - Opening price
ts['gain'] = ts['close'] - ts['open']

ratio = 0.8 # 80% of training set and 20% of testing set
training = ts.to_sframe()[0:round(len(ts)*ratio)]
testing = ts.to_sframe()[round(len(ts)*ratio):]




'''
Multiple Machine Learning Classifiers
'''
# create DECISION TREE CLASSIFIER with graphlab
max_tree_depth = 6
decision_tree = gl.decision_tree_classifier.create(training, validation_set=None, 
                                                   target='outcome', features=l_features, 
                                                   max_depth=max_tree_depth, verbose=False)

# display accuracy of the fitted model both with training set and testing set
decision_tree.evaluate(training)['accuracy'], decision_tree.evaluate(testing)['accuracy']
# example response
# (0.6077348066298343, 0.577373211963589)

predictions = decision_tree.predict(testing)
# and we add the predictions  column in testing set
testing['predictions'] = predictions

# let's see the first 10 predictions, compared to real values (outcome column)
testing[['datetime', 'outcome', 'predictions']].head(10)

# backtesting model
pnl = testing[testing['predictions'] == 1]['gain'] # the gain column contains (Close - Open) values
# I have written a simple helper function to plot the result of all the trades applied to the
# testing set and represent the total return expressed by the index basis points
# (not expressed in dollars $)
plot_equity_chart(pnl,'Decision tree model')


# create LOGISTIC classifier with graphlab
model = gl.logistic_classifier.create(training, target='outcome', features=l_features, 
                                      validation_set=None, verbose=False)
predictions_prob = model.predict(testing, 'probability')
THRESHOLD = 0.6
bt_2_2 = backtest_ml_model(testing, predictions_prob, target='outcome', 
                           threshold=THRESHOLD, STOP=-3, plot_title=model.name())
backtest_summary(bt_2_2)


# create LINEAR REGRESSION classifier
model = gl.linear_regression.create(training, target='gain', features = l_lr_features,
                                   validation_set=None, verbose=False, max_iterations=100)
predictions = model.predict(testing)
# a linear regression model, predict continuous values, so we need to make an estimation of their
# probabilities of success and normalize all values in order to have a vector of probabilities
predictions_max, predictions_min = max(predictions), min(predictions)
predictions_prob = (predictions - predictions_min)/(predictions_max - predictions_min)


# create BOOSTED TREE classifier
model = gl.boosted_trees_classifier.create(training, target='outcome', features=l_features, 
                                           validation_set=None, max_iterations=12, verbose=False)
predictions_prob = model.predict(testing, 'probability')

THRESHOLD = 0.7
bt_4_2 = backtest_ml_model(testing, predictions_prob, target='outcome', 
                           threshold=THRESHOLD, STOP=-3, plot_title=model.name())
backtest_summary(bt_4_2)


# create RANDOM FOREST classifier
model = gl.random_forest_classifier.create(training, target='outcome', features=l_features, 
                                      validation_set=None, verbose=False, num_trees = 10)
predictions_prob = model.predict(testing, 'probability')
THRESHOLD = 0.6
bt_5_2 = backtest_ml_model(testing, predictions_prob, target='outcome', 
                           threshold=THRESHOLD, STOP=-3, plot_title=model.name())
backtest_summary(bt_5_2)
