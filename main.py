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
