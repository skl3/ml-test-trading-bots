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

Analysis
--------

Each of the machine learning models/classifiers was used to calculate 4 values:
* accuracy
* precision
* round turns
  * profit and loss for a complete daily trade
* sharpe ratio
  * value describing how much excess return you are receiving for the extra volatility that you endure for holding a risker asset

Read more about Sharpe Ratios at <http://www.investopedia.com/articles/07/sharpe_ratio.asp>

As a trading model, we want to increase the precision of our classifiers/models to reduce the number of false positives.

Final Results from multiple machine learning models sorted by their precision
| Model                    | Accuracy | Precision  | Round Turns | Sharpe  |
| ------------------------ |:--------:| :---------:| :----------:|--------:|
| Linear Regression        | 0.63     | 0.71       | 319         | 7.65    |
| Boosted Trees Classifier | 0.56     | 0.68       | 214         | 6.34    |
| Random Forest Classifier | 0.6      | 0.67       | 311         | 6.38    |
| Decision Tree            | 0.56     | 0.66       | 234         | 6.52    |
| Logistic Classifier      | 0.64     | 0.66       | 426         | 6.45    |


Conclusion
----------
The linear regression model seemed to do 

