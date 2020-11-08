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


