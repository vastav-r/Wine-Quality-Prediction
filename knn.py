from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

import pandas as pd

df = pd.read_csv('dataset/winequality-red.csv', header=0, sep=';')

X = df[list(df.columns)[:-1]]
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = KNeighborsClassifier()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)

print ('\nAccuracy:', model.score(X_test, y_test)*100)
print ('\nRMSE:', mean_squared_error(y_predict, y_test) ** 0.5)
print ('\nConfusion Matrix :')
print (confusion_matrix(y_test,y_predict))
