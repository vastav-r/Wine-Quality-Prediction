from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix

import pandas as pd

df = pd.read_csv('dataset/winequality-red.csv', header=0, sep=';')

X = df[list(df.columns)[:-1]]
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y)

forest = RandomForestClassifier(n_estimators = 150)
forest.fit(X_train, y_train)
y_predict = forest.predict(X_test)
forest.score(X_test, y_test)
acc=forest.score(X_test, y_test)*100
print ('\nAccuracy ', acc )
print ('\nRMSE:', mean_squared_error(y_predict, y_test) ** 0.5)
print ('\nConfusion Matrix :\n',confusion_matrix(y_test,y_predict))
