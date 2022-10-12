# วุฒินันท์ สุขพูล 1611051541105
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

def cleanData(dataset):
    for column in dataset.columns:
        if dataset[column].dtype == type(object):
            le = LabelEncoder()
            dataset[column] = le.fit_transform(dataset[column])
    return dataset

df = pd.read_csv('PlayTennis.csv')
df  = cleanData(df)

x = df.drop('Play Tennis', axis=1).values
y = df['Play Tennis'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.5)

model = MLPClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print(classification_report(y_test, y_pred, target_names=['Actually', 'Predicted']))