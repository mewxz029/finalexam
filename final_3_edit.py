# วุฒินันท์ สุขพูล 1611051541105
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv('cars.csv')
df = df.drop('Unnamed: 0', axis=1)

x = df.drop('am', axis=1).values
y = df['am'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)

model = GaussianNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("Naive Bayes")
print(classification_report(y_test, y_pred, target_names=['Actually', 'Predicted']))