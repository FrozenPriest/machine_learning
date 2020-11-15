import numpy as np
import scipy
import pandas
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, \
    precision_recall_curve


def text_transform(text: pandas.Series) -> pandas.Series:
    return text.str.lower().replace("[^a-zA-Z0-9]", " ", regex=True)


df = pandas.read_csv("salary-train.csv")

df['FullDescription'] = df['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)

vectorizer = TfidfVectorizer(min_df=5)
X_train_text = vectorizer.fit_transform(df["FullDescription"])

df['LocationNormalized'].fillna('nan', inplace=True)
df['ContractTime'].fillna('nan', inplace=True)

enc = DictVectorizer()
X_train_cat = enc.fit_transform(df[["LocationNormalized", "ContractTime"]].to_dict("records"))

X_train = scipy.sparse.hstack([X_train_text, X_train_cat])

y_train = df["SalaryNormalized"]
model = Ridge(alpha=1, random_state=241)
model.fit(X_train, y_train)

test = pandas.read_csv("salary-test-mini.csv")
test['FullDescription'] = test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)

X_test_text = vectorizer.transform(test["FullDescription"])
X_test_cat = enc.transform(test[["LocationNormalized", "ContractTime"]].to_dict("records"))
X_test = scipy.sparse.hstack([X_test_text, X_test_cat])

y_test = model.predict(X_test)
print(y_test[0])
print(y_test[1])
