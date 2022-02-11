# Import libraries
import pandas, numpy, seaborn
import sklearn
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


# Load data
filePath = "/home/amousas/Downloads/metabol.xlsx"
data = pandas.read_excel(filePath)

# Identify samples with missing values
missingValues = []
for i in range(data.shape[0]):
   if len(set(numpy.isnan(data.iloc[i,:]))) > 1:
       missingValues.append(i)

# Impute missing values
for i in range(len(missingValues)):
    index = numpy.where(numpy.isnan(data.iloc[missingValues[i],:]) == 1)[0]
    for j in range(len(index)):
        data.iloc[:,index[j]] = data.iloc[:,index[j]].fillna(numpy.mean(data.iloc[:,index[j]])) 
#dataNaN = data.dropna()

# Train/Test set split
X = data.iloc[:,range(139)]
featureNames = X.columns.values
y = data.iloc[:,140]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1983)

# Normalization of the data matrix
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_train_scaled = pandas.DataFrame(X_train_scaled)

scaler = preprocessing.StandardScaler().fit(X_test)
X_test_scaled = scaler.transform(X_test)
X_test_scaled = pandas.DataFrame(X_test_scaled)


##########################################################################
# Lasso
##########################################################################
a = numpy.random.random_sample(size=1000)
aa = numpy.random.random_sample(size=100)
b = aa+1
c = aa+2
aGrid = numpy.concatenate((a,b,c))
lassoTrain = sklearn.linear_model.LassoCV(alphas=aGrid,positive=True,cv=10,max_iter=10000,normalize=False,random_state=1983).fit(X_train_scaled,y_train)

finalFit = sklearn.linear_model.Lasso(alpha=0.0002093020658164857,positive=True,max_iter=10000,random_state=1983).fit(X_train_scaled,y_train)

y_predict = finalFit.predict(X_test_scaled)
sklearn.metrics.accuracy_score(y_test,numpy.round(abs(y_predict)))
##########################################################################

hmdbData = data.iloc[:,[37,38,140]]

sampleNames = []
for i in range(hmdbData.shape[0]):
   sampleNames.append("S" + repr(i+1))
hmdbData["Samples"] = sampleNames

hmdbData = hmdbData.rename(columns={"Glutamine":"HMDB0000641","Glycine":"HMDB0000123","Health_metrics_all_ideal_81651$ideal_categ":"Group"})

index0 = numpy.where(hmdbData["Group"] == 0)[0]
hmdbData["Group"].iloc[index0] = "Control"

index1 = numpy.where(hmdbData["Group"] == 1)[0]
hmdbData["Group"].iloc[index1] = "Case"

hmdbData = hmdbData[["Samples","Group","HMDB0000641","HMDB0000123"]]
hmdbData.to_csv("/home/amousas/Downloads/hmdbData.csv",sep=",",index=False)


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


