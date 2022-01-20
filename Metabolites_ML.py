import pandas, numpy, seaborn
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

filePath = "/home/amousas/Downloads/metabol.xlsx"

data = pandas.read_excel(filePath)
dataNaN = data.dropna()

X = dataNaN.iloc[:,range(139)]
y = dataNaN.iloc[:,140]

rf = RandomForestClassifier(random_state=1983)
rf.fit(X, y)
rf.score(X,y)
rf.feature_importances_

plt.bar(range(139),rf.feature_importances_)