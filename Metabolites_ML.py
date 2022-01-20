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

plt.bar(range(len(rf.feature_importances_)),rf.feature_importances_)


corr_matrix = X.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(numpy.triu(numpy.ones(corr_matrix.shape), k=1).astype(numpy.bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]

# Drop features 
Xthin = X.drop(to_drop, axis=1)

rf2 = RandomForestClassifier(random_state=1983)
rf2.fit(Xthin, y)
rf2.score(Xthin,y)
rf2.feature_importances_

plt.bar(range(len(rf2.feature_importances_)),rf2.feature_importances_)
