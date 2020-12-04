import datetime
import numpy
import pandas
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler


def get_pick_from_train(train: pandas.DataFrame) -> pandas.DataFrame:
    pick = numpy.zeros((train.shape[0], max))
    for i, match_id in enumerate(train.index):
        for p in range(1, 6):
            pick[i, train.loc[match_id, f"r{p}_hero"] - 1] = 1
            pick[i, train.loc[match_id, f"d{p}_hero"] - 1] = -1
    return pandas.DataFrame(pick, index=train.index, columns=["hero_" + str(i) for i in range(max)])


def compute_l2r_score(X_train: pandas.DataFrame, Y_train: pandas.Series) -> pandas.Series:
    scores = {}

    for i in range(-5, 5):
        C = 10.0 ** i

        print("C = " + str(C))
        model = LogisticRegression(C=C)

        start_time = datetime.datetime.now()
        cross_score = cross_val_score(model, X_train, Y_train, cv=kfold, scoring="roc_auc", n_jobs=-1).mean()
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

    return ans


train = pandas.read_csv("data/features.csv", index_col="match_id")
train.drop([
    "duration",
    "tower_status_radiant",
    "tower_status_dire",
    "barracks_status_radiant",
    "barracks_status_dire",
], axis=1, inplace=True)
test = pandas.read_csv("data/features_test.csv", index_col="match_id")
test.fillna(0, inplace=True)

na_size = len(train) - train.count()
na_size_sorted = na_size[na_size > 0].sort_values(ascending=False) / len(train)
print(na_size_sorted)

train.fillna(0, inplace=True)
Y_train = train["radiant_win"]
X_train = train.drop("radiant_win", axis=1)

kfold = KFold(n_splits=5, shuffle=True)
scores = {}

for n_estimators in [10, 20, 30, 40, 50, 80, 100, 120, 200, 400]:
    print("n_estimators = " + str(n_estimators))
    model = GradientBoostingClassifier(n_estimators=n_estimators)

    start_time = datetime.datetime.now()
    cross_score = cross_val_score(model, X=X_train, y=Y_train, cv=kfold, scoring="roc_auc", n_jobs=-1).mean()
    print("Score = " + str(cross_score))
    print("Time = " + str(datetime.datetime.now() - start_time))

    scores[n_estimators] = cross_score
    print("------------------------")
plt.figure()
plt.plot(pandas.Series(scores))
plt.show()

scaler = StandardScaler()
X_train = pandas.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
compute_l2r_score(X_train, Y_train)


redundant_column_names = ["d"+str(i)+"_hero" for i in range(1, 6)] + \
                         ["r"+str(i)+"_hero" for i in range(1, 6)] + \
                         ["lobby_type"]
X_train.drop(redundant_column_names, axis=1, inplace=True)
compute_l2r_score(X_train, Y_train)


hero_columns = ["d"+str(i)+"_hero" for i in range(1, 6)] + \
               ["r"+str(i)+"_hero" for i in range(1, 6)]
len = len(numpy.unique(train[hero_columns].values.ravel()))
max = max(numpy.unique(train[hero_columns].values.ravel()))
print("Len = " + str(len))
print("Max = " + str(max))

X_pick = get_pick_from_train(train)

X_train = pandas.concat([X_train, X_pick], axis=1)
compute_l2r_score(X_train, Y_train)

model = LogisticRegression(C=0.1)
model.fit(X_train, Y_train)


X_test = pandas.DataFrame(scaler.transform(test), index=test.index, columns=test.columns)
X_test.drop(redundant_column_names, axis=1, inplace=True)
X_test = pandas.concat([X_test, get_pick_from_train(test)], axis=1)

predictions = pandas.Series(model.predict_proba(X_test)[:, 1])
print(predictions.describe())
