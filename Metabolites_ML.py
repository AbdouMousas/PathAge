import pandas, numpy, seaborn
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

filePath = "/home/amousas/Downloads/metabol.xlsx"

data = pandas.read_excel(filePath)
dataNaN = data.dropna()

X = dataNaN.iloc[:,range(139)]
y = dataNaN.iloc[:,140]

from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)
X_scaled = pandas.DataFrame(X_scaled)


##########################################################################
# Lasso
##########################################################################
import sklearn
lasso = sklearn.linear_model.LassoCV(cv=10,max_iter=10000,normalize=False,random_state=1983).fit(X_scaled,y)

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

########################################################################################
# Βlock modeling
########################################################################################
names = X.columns

idx = numpy.zeros(len(names))
corr= X.corr(method="spearman").abs().values
threshold = 0.8
ii = 0
exit = 0

while exit != 1:
    index = numpy.where(corr[0,:] > threshold)[0]
    corr = numpy.delete(corr,0,axis=0)
    corr = numpy.delete(corr,0,axis=1)
    if len(index) > 1:
       idx[index] = ii + 1
       ii = ii + 1
    if corr.shape[0] == 0:
        exit = 1
    

corr= X.corr(method="spearman").abs().values
threshold = 0.8    
clusters = []
seeds = numpy.arange(len(corr))

while len(seeds) > 0:
    index = numpy.where(corr[seeds[0],:] > threshold)[0]
    clusters.append(index)
    seeds = numpy.setdiff1d(seeds,index)


suma = 0
for i in range(len(clusters)):
    suma = suma + len(clusters[i])

numpy.argsort(idx)

ind = []
for i in range(len(corr)):
   index = numpy.where((corr[i,:] > threshold) & (corr[i,:] < 1))[0]
   ind.extend(index)

ind = sorted(list(set(ind)))


ind = []
for i in range(len(corr)):
   index = numpy.where(corr[i,:] > threshold)[0]
   ind.append(index)

index1 = []
for i in range(len(ind)):
   if 133 in ind[i]:
       index1.append(i)

seaborn.heatmap(corr)

import scipy
import scipy.cluster.hierarchy as sch

def cluster_corr(corr_array, inplace=False):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly 
    correlated variables are next to eachother 
    
    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix 
        
    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)
    
    if not inplace:
        corr_array = corr_array.copy()
    
    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx]


corrReArrange = cluster_corr(corr)
corrReArrangePD = cluster_corr(X.transpose())

seaborn.heatmap(cluster_corr(corr))



pairwise_distances = sch.distance.pdist(corr)
linkage = sch.linkage(pairwise_distances, method='complete')
cluster_distance_threshold = pairwise_distances.max()/2
idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
                                        criterion='distance')
idx = numpy.argsort(idx_to_cluster_array)

import plotly.express as px
fig = px.bar(y=rf.feature_importances_,x=X.columns.values)
#fig.write_image("/home/amousas/Downloads/test.png")
fig.write_html("/home/amousas/Downloads/test.html")

fig2 = px.imshow(X.corr(method="spearman").abs())
fig2.write_html("/home/amousas/Downloads/test2.html")

fig3 = px.imshow(corrReArrange)
fig3.write_html("/home/amousas/Downloads/test3.html")


