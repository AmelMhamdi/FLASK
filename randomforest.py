from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
x = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.5)

clf = RandomForestClassifier(n_estimators=12)

clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
print(accuracy_score(predicted, y_test))

import pickle

with open('C:/Users/Amel/Documents/IAworkspace/FLASK/FLASK/rf.pkl', 'wb') as model_pkl:
    pickle.dump(clf, model_pkl)
