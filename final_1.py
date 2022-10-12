# วุฒินันท์ สุขพูล 1611051541105
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('cars.csv')

x = df['hp'].values.reshape(-1 ,1)
y = df['drat'].values.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)

# y = ax + b
model = LinearRegression()
model.fit(x_train, y_train)
a = model.intercept_
b = model.coef_

y_pred = model.predict(x_test)

def get_prediction_interval(prediction, y_test, test_predictions, pi=.95):
    '''
    Get a prediction interval for a linear regression.
    INPUTS:
    - Single prediction,
    - y_test
    - All test set predictions,
    - Prediction interval threshold (default = .95)
    OUTPUT:
    - Prediction interval for single prediction
    '''
    #get standard deviation of y_test
    sum_errs = np.sum((y_test - test_predictions)**2)
    stdev = np.sqrt(1 / (len(y_test) - 2) * sum_errs)
    
    #get interval from standard deviation
    one_minus_pi = 1 - pi
    ppf_lookup = 1 - (one_minus_pi / 2)
    z_score = stats.norm.ppf(ppf_lookup)
    interval = z_score * stdev
    
    #generate prediction interval lower and upper bound
    lower, upper = prediction - interval, prediction + interval

    return lower, prediction, upper

## Plot and save confidence interval of linear regression  - 95% 
lower_vet = []
upper_vet = []
for i in y_pred:
    lower, prediction, upper =  get_prediction_interval(i, y_test, y_pred)
    lower_vet.append(lower)
    upper_vet.append(upper)

plt.scatter(x, y,  color='black')
plt.plot(x_test, y_pred,  label="y = "+ str(b) +"+ "+ str(a) +"x")
plt.plot(x_test, lower_vet, 'r',label="95% Interval")
plt.plot(x_test, upper_vet, 'r')
plt.xlabel('Gross Horse Power')
plt.ylabel('Rear Axes Ratio')
plt.legend(loc="upper left")
plt.title("Graph Linear Regression Confidence 95%")

plt.grid()
plt.show()