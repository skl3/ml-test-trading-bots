trading-bot-amaterasu
=====================
Trading algorithm leveraging various classic machine learning classifiers to train multiple trading models. 

Installation
------------

* Create a virtual environment
```
# Create a virtual environment named e.g. dato-env
$ virtualenv venv

# Activate the virtual environment
$ source venv/bin/activate
```

* Ensure pip version >= 7
```
# Make sure pip is up to date
$ pip install --upgrade pip
```

* Ensure install of IPython and IPython Notebook
```
# Install IPython Notebook (optional)
$ pip install "ipython[notebook]"
```

* Install Yahoo Finanace Python packge

```
$ pip install yahoo-finance
```

* Install GraphLab Create
```
# Install your licensed copy of GraphLab Create
$ pip install --upgrade --no-cache-dir https://get.graphlab.com/GraphLab-Create/1.10.1/**YOUR_EMAIL_HERE**/**YOUR_KEY_HERE**/GraphLab-Create-License.tar.gz
```

Run
---
Start IPython notebook or IPython command line with

    $ ipython notebook

or

    $ ipython

Read about running IPython notebook here:
<http://opentechschool.github.io/python-data-intro/core/notebook.html>

**Note**

The functions and scripts in the file are not all supposed to be executed at once. The classifiers should be ran one at a time then the appropriate backtesting function and general analysis should be applied to each classfier/model one at a time.

Analysis
--------

Each of the machine learning models/classifiers was used to calculate 4 values:
* Accuracy
* Precision
* Round Turns
  * profit and loss for a complete daily trade
* Sharpe Ratio
  *Vvalue describing how much excess return you are receiving for the extra volatility that you endure for holding a risker asset

Read more about Sharpe Ratios at <http://www.investopedia.com/articles/07/sharpe_ratio.asp>

As a trading model, we want to increase the precision of our classifiers/models to reduce the number of false positives in our predictions.

#### Decision Tree
```
Mean of PnL is 118.244689 
Sharpe is 6.523478
Round turns 234
Name: DecisionTree
Accuracy: 0.560468140442
Precision: 0.662393162393
Recall: 0.374396135266
Max Drawdown: -1769.00025
```

#### Logistic Classifier
```
Mean of PnL is 112.704215 
Sharpe is 6.447859
Round turns 426
Name: LogisticClassifier
Accuracy: 0.638491547464
Precision: 0.659624413146
Recall: 0.678743961353
Max Drawdown: -1769.00025
```

#### Linear Regression
```
Mean of PnL is 138.868280 
Sharpe is 7.650187
Round turns 319
Name: LinearRegression
Accuracy: 0.631989596879
Precision: 0.705329153605
Recall: 0.54347826087
Max Drawdown: -1769.00025
```

#### Boosted Tree
```
Mean of PnL is 112.002338
Sharpe is 6.341981
Round turns 214
Name: BoostedTreesClassifier
Accuracy: 0.563068920676
Precision: 0.682242990654
Recall: 0.352657004831
Max Drawdown: -1769.00025
```

#### Random Forest
```
Mean of PnL is 114.786962 
sharpe is 6.384243
Round turns 311
Name: RandomForestClassifier
Accuracy: 0.598179453836
Precision: 0.668810289389
Recall: 0.502415458937
Max Drawdown: -1769.00025
```

Final Results from multiple machine learning models sorted by their precision.

| Model                    | Accuracy | Precision  | Round Turns | Sharpe  |
| ------------------------ |:--------:| :---------:| :----------:|--------:|
| Linear Regression        | 0.63     | 0.71       | 319         | 7.65    |
| Boosted Trees Classifier | 0.56     | 0.68       | 214         | 6.34    |
| Random Forest Classifier | 0.6      | 0.67       | 311         | 6.38    |
| Decision Tree            | 0.56     | 0.66       | 234         | 6.52    |
| Logistic Classifier      | 0.64     | 0.66       | 426         | 6.45    |


Conclusion
----------
The linear regression model seemed to do the best of all the models in terms of Sharpe Ratio by far.
Granted in the future when testing many parameters when creating the classifiers can still be tweaked to improve it like increasing the number of max_iterations.

