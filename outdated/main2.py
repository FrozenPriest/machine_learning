import numpy as np
import pandas
from sklearn.tree import DecisionTreeClassifier

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

sliced_data = data[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']]
sliced_data = sliced_data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
survived = sliced_data['Survived']
sliced_data = sliced_data[['Pclass', 'Fare', 'Age', 'Sex']]
dictionary = {'male': 1, 'female': 2}
sliced_data['Sex'].replace(dictionary, inplace=True)

print(sliced_data)

X = sliced_data
Y = survived

clf = DecisionTreeClassifier(random_state=241)

clf.fit(X, Y)

print(clf.feature_importances_)




##################################################
import numpy as np
import pandas
import sklearn
from numpy import size
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold


data = pandas.read_csv('wine.data', names=['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', '0D280', 'Proline'])

Y = data['Class']
X = data.drop(['Class'], axis=1)

kfold = KFold(shuffle=True, n_splits=5, random_state=42)

X = sklearn.preprocessing.scale(X=X)

fres = 0
kres = 0
for k in range(1, 50):
    result = sklearn.model_selection.cross_val_score(cv=kfold, estimator=KNeighborsClassifier(n_neighbors=k), scoring='accuracy', X=X, y=Y)
    midres = sum(result)/size(result)
    if fres < midres:
        fres = midres
        kres = k
    print("k = " + str(k))
    print("Midres = " + str(midres))

print(fres)
print(kres)

#############################################################################

import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor

boston = load_boston()
X = scale(X=boston.data)
Y = boston.target

kfold = KFold(shuffle=True, n_splits=5, random_state=42)

fres = 0
kres = 0
for p in np.linspace(start=1, stop=10, num=200, endpoint=True, retstep=False):
    regressor = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p=p)
    regressor.fit(X=X, y=Y)
    result = cross_val_score(cv=kfold, estimator=regressor, scoring='neg_mean_squared_error', X=X, y=Y)

    print(result)
    midres = sum(result) / np.size(result)
    if fres < midres:
        fres = midres
        kres = p
#    print("P = " + str(p))
#    print("Midres = " + str(midres))

print(fres)
print(kres)



















import sklearn
from sklearn.linear_model import Perceptron
import pandas
from sklearn.preprocessing import StandardScaler

perceptron_train = pandas.read_csv('perceptron-train.csv', header=None)
perceptron_test = pandas.read_csv('perceptron-test.csv', header=None)

X_train = perceptron_train.iloc[:, 1:3]
y_train = perceptron_train[0]

X_test = perceptron_test.iloc[:, 1:3]
y_test = perceptron_test[0]

clf = Perceptron(random_state=241)
clf.fit(X_train, y_train)

accuracy = sklearn.metrics.accuracy_score(y_test, clf.predict(X_test))

print(accuracy)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = Perceptron()
clf.fit(X_train_scaled, y_train)
predictions_scaled = clf.predict(X_test_scaled)

scaled_accuracy = sklearn.metrics.accuracy_score(y_test, clf.predict(X_test_scaled))
print(scaled_accuracy)
print(scaled_accuracy-accuracy)







import sklearn
import numpy as np
from sklearn.linear_model import Perceptron
import pandas
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn import datasets

newsgroups = datasets.fetch_20newsgroups(
                    subset='all',
                    categories=['alt.atheism', 'sci.space']
             )

X_train = newsgroups.data
y_train = newsgroups.target

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer()
X_transformed = vectorizer.fit_transform(X_train)


cv = KFold(n_splits=5, shuffle=True, random_state=241)
grid = {"C": np.power(10.0, np.arange(-5, 6))}

clf = sklearn.svm.SVC(random_state=241, kernel='linear')
gs = GridSearchCV(clf, grid, scoring="accuracy", cv=cv, verbose=1, n_jobs=-1)

gs.fit(X_transformed, y_train)
C = gs.best_params_.get('C')
print(C)

clf = sklearn.svm.SVC(random_state=241, kernel='linear', C=C)
clf.fit(X_transformed, y_train)

words = np.array(vectorizer.get_feature_names())
word_weights = pandas.Series(clf.coef_.data, index=words[clf.coef_.indices], name="weight")
word_weights.index.name = "word"

top_words = word_weights.abs().sort_values(ascending=False).head(10)
print(top_words)




from typing import Tuple

import sklearn
import numpy as np
from sklearn.linear_model import Perceptron
import pandas
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler


def calc_w1(X: pandas.DataFrame, y: pandas.Series, w1: float, w2: float, k: float, C: float) -> float:
    l = len(y)
    S = 0
    for i in range(0, l):
        S += y[i] * X[1][i] * (1.0 - 1.0 / (1.0 + np.exp(-y[i] * (w1*X[1][i] + w2*X[2][i]))))

    return w1 + (k * (1.0 / l) * S) - k * C * w1


def calc_w2(X: pandas.DataFrame, y: pandas.Series, w1: float, w2: float, k: float, C: float) -> float:
    l = len(y)
    S = 0
    for i in range(0, l):
        S += y[i] * X[2][i] * (1.0 - 1.0 / (1.0 + np.exp(-y[i] * (w1*X[1][i] + w2*X[2][i]))))

    return w2 + (k * (1.0 / l) * S) - k * C * w2


def gradient_descent(X: pandas.DataFrame, y: pandas.Series, w1: float=0.0, w2: float=0.0,
         k: float=0.1, C: float=0.0, precision: float=1e-5, max_iter: int=10000) -> Tuple[float, float]:
    for i in range(max_iter):
        w1_prev, w2_prev = w1, w2
        w1, w2 = calc_w1(X, y, w1, w2, k, C), calc_w2(X, y, w1, w2, k, C)
        if np.sqrt((w1_prev - w1) ** 2 + (w2_prev - w2) ** 2) <= precision:
            break

    return w1, w2


def a(X: pandas.DataFrame, w1: float, w2: float) -> pandas.Series:
    return 1.0 / (1.0 + np.exp(-w1 * X[1] - w2 * X[2]))


df = pandas.read_csv("data-logistic.csv", header=None)
X = df.loc[:, 1:]
y = df[0]

w1, w2 = gradient_descent(X, y)
w1_reg, w2_reg = gradient_descent(X, y, C=10.0)


y_proba = a(X, w1, w2)
y_proba_reg = a(X, w1_reg, w2_reg)

auc = roc_auc_score(y, y_proba)
auc_reg = roc_auc_score(y, y_proba_reg)

print(auc)
print(auc_reg)




import numpy as np
import pandas
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, \
    precision_recall_curve

df = pandas.read_csv("classification.csv")
print(df)

TP = df[(df["pred"] == 1) & (df["true"] == 1)]
FP = df[(df["pred"] == 1) & (df["true"] == 0)]
FN = df[(df["pred"] == 0) & (df["true"] == 1)]
TN = df[(df["pred"] == 0) & (df["true"] == 0)]

print(len(TP))
print(len(FP))
print(len(FN))
print(len(TN))

accuracy = accuracy_score(df["true"], df["pred"])
precision = precision_score(df["true"], df["pred"])
recall = recall_score(df["true"], df["pred"])
f1 = f1_score(df["true"], df["pred"])

print(accuracy)
print(precision)
print(recall)
print(f1)

df = pandas.read_csv("scores.csv")
print(df)

clf_names = df.columns[1:]
scores = pandas.Series([roc_auc_score(df["true"], df[clf]) for clf in clf_names], index=clf_names)

print(scores.sort_values(ascending=False).index[0])

pr_scores = []
for clf in clf_names:
    pr_curve = precision_recall_curve(df["true"], df[clf])
    pr_scores.append(pr_curve[0][pr_curve[1] >= 0.7].max())

print(clf_names[np.argmax(pr_scores)])
