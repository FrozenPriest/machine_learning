import datetime

import matplotlib.pyplot as plt
import pandas
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler

result_path = 'C:\\Users\\boral\\PycharmProjects\\machine_learning\\dataset.csv'
train = pandas.read_csv(result_path, sep=";")
print(train.head())
train.drop("Name", axis=1, inplace=True)

print("--------------------")

Y_train = train["Type"]
X_train = train.drop("Type", axis=1)

scaler = StandardScaler()
X_train = pandas.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)

kfold = KFold(n_splits=8, shuffle=True)

# -----------------GRADIENT BOOSTING------------------------------------------------------
scores = {}
for n_estimators in [10, 20, 30, 40, 50, 80, 120, 200, 250, 300, 400]:
    print("n_estimators = " + str(n_estimators))
    model = GradientBoostingClassifier(n_estimators=n_estimators)

    start_time = datetime.datetime.now()
    cross_score = cross_val_score(model, X=X_train, y=Y_train, cv=kfold, n_jobs=-1).mean()
    print("Score = " + str(cross_score))
    print("Time = " + str(datetime.datetime.now() - start_time))

    scores[n_estimators] = cross_score
    print("------------------------")

plt.figure()
plt.plot(pandas.Series(scores))
plt.show()

max_iteration = pandas.Series(scores).sort_values(ascending=False).head(1)
max_score = max_iteration.values[0]
max_C = max_iteration.index[0]
print("Max_N = " + str(max_C))
print("Max_score = " + str(max_score))
# -----------------LOGISTIC REGRESSION------------------------------------------------------
scores = {}

for i in range(-6, 8):
    C = 10.0 ** i

    print("C = " + str(C))
    model = LogisticRegression(C=C, max_iter=10000)

    start_time = datetime.datetime.now()
    cross_score = cross_val_score(model, X_train, Y_train, cv=kfold, n_jobs=-1).mean()
    print("Score = " + str(cross_score))
    print("Time = " + str(datetime.datetime.now() - start_time))

    scores[i] = cross_score
    print("------------------------")

plt.figure()
ans = pandas.Series(scores)
plt.plot(ans)
plt.show()

max_iteration = ans.sort_values(ascending=False).head(1)
max_score = max_iteration.values[0]
max_C = 10.0 ** max_iteration.index[0]

print("Max_C = " + str(max_C))
print("Max_score = " + str(max_score))

#
# model = LogisticRegression(C=0.1)
# model.fit(X_train, Y_train)


# X_test = pandas.DataFrame(scaler.transform(test), index=test.index, columns=test.columns)

# predictions = pandas.Series(model.predict_proba(X_test)[:, 1])
# print(predictions.describe())
