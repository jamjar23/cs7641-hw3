# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 12:49:17 2018

@author: James
based on code from https://github.com/JonathanTay/CS-7641-assignment-3/
"""

#%% Imports
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib import cm
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
#import seaborn as sns; sns.set()

# helpers.py should be in the same direcory as this script
this_dir='D:\\GoogleDrive\\_Study\\GT_ML\\hw3'
os.chdir(this_dir)
import helpers as hlp
from helpers import nn_arch,nn_reg

#out = './{}/'.format(sys.argv[1])
out = '.\\out\\'
cmap = cm.get_cmap('Spectral') 
np.random.seed(54)

# Load Titanic
dfT, _ = hlp.load_titanic(this_dir + '\\titanic\\train.csv')
fnT = list(dfT.columns)[1:]     # feature names
# convert to arrays
XT_orig = dfT.iloc[:, 1:].values
yT = dfT.iloc[:, 0].values
# scale Titanic data
XT = StandardScaler().fit_transform(XT_orig)
# split into training and test sets
#XT_train, XT_test, yT_train, yT_test = train_test_split(XT_std, yT, test_size=0.3, random_state=1, stratify=yT)

# Load WILT 
XW_train, yW_train, XW_test, yW_test, fnW = hlp.load_wilt(this_dir + '\\wilt')
fnW = ['GLCM_pan','Mean_Green','Mean_Red','Mean_NIR','SD_pan']
# convert from df to arrays
XW_train = XW_train.values; yW_train = yW_train.values; XW_test = XW_test.values; yW_test = yW_test.values
# Scale WILT data 
stdsc = StandardScaler()
XW_train = stdsc.fit_transform(XW_train)
XW_test  = stdsc.transform(XW_test)
yW_train = yW_train.ravel()
yW_test = yW_test.ravel()

#%% Part 2 PCA

# PCA for Titanic
pcaT = PCA(random_state=54)
pcaT.fit(XT)
pcaT.explained_variance_,
tmpT = pd.Series(data = pcaT.explained_variance_)
tmpT.index += 1
tmpT.name= 'Titanic'
tmpT.to_csv(out+'titanic scree.csv')
cumT = np.cumsum(pcaT.explained_variance_ratio_)

# PCA for Wilt
pcaW = PCA(random_state=54)
pcaW.fit(XW_train)
pcaW.explained_variance_
tmpW = pd.Series(data = pcaW.explained_variance_)
tmpW.index += 1
tmpW.name= 'Wilt'
tmpW.to_csv(out+'digits scree.csv')
cumW = np.cumsum(pcaW.explained_variance_ratio_)

# Plot Variance Explained
t = 'PCA Variance Explained'; h=2.5; w=4
fig = plt.figure(t, figsize=(w, h))
plt.clf()
plt.suptitle(t)
fig.subplots_adjust(bottom=0.17)
# Titanic
ax1 = fig.add_subplot(121)
plt.xlabel(tmpT.name, size='small')
ax1.plot(tmpT,'-o', markersize=3)
plt.tick_params(labelsize=8)
plt.xticks(range(1,11,2))
# Wilt
ax2 = fig.add_subplot(122)
plt.xlabel(tmpW.name, size='small')
ax2.plot(tmpW,'-o', markersize=3)
plt.tick_params(labelsize=8)
plt.xticks(range(1,6))

# Cumulative
t = 'PCA Cumulative Variance Explained'; h=3; w=5
fig = plt.figure(t, figsize=(w, h))
plt.plot(cumT)
plt.xlabel('components')
plt.plot(cumW)
fig.subplots_adjust(bottom=0.17)
#plt.plot(lossT)
#plt.plot(lossW)
plt.legend(['Titanic CEV','Wilt CEV', 'Titanic Recon Cos Sim', 'Wilt Recon Cos Sim'])
plt.title('Cumulative Variance Explained')

#%% Cosine Similarity
# projections of each reconstructed vector onto original?
# this only captures first element of each reconstruction, based on suspect advice
# I believe proper analysis would require analyzing each vector projection
# results here are not used, except to set pca and Xpca values to last element in list
from sklearn.metrics.pairwise import cosine_similarity
# Titanic
lossT=[]
X=XT
X.shape
dims=[1,2,3,4,5]
dims=[1,2,3,4,5,6,7,8,9,10,5]
for i in dims:   # last element repeats desired dimensions so we have variables set
    pcaT = PCA(random_state=54, n_components=i)
    pcaT.fit(X)
    XTpca = pcaT.transform(X)
    XTproj = pcaT.inverse_transform(XTpca)
#    lossT.append(cosine_similarity(XTproj, XT)[0][0])
    lossT.append(((X - XTproj) ** 2).mean())
lossT.pop()
lossT
np.cumsum(pcaT.explained_variance_ratio_)

# Wilt
lossW=[]; i=1
for i in [1,2,3,4,5,4]:
    pcaW = PCA(random_state=54, n_components=i)
    pcaW.fit(XW_train)
    XWpca = pcaW.transform(XW_train)
    XWproj = pcaW.inverse_transform(XWpca)
    lossW.append(cosine_similarity(XWproj, XW_train)[0][0])
#    lossW.append(((XW_train - XWproj) ** 2).mean())
lossW.pop()
XTpca.shape
XWpca.shape

#%% Clustering after DR
XT_train, XT_test, yT_train, yT_test = train_test_split(XTpca, yT, test_size=0.3, random_state=1, stratify=yT)
datasetsPCA={}
datasetsPCA['Titanic']={'X_train':XT_train.copy(), 'y_train':yT_train.copy(), 'X_test':XT_test.copy(), 'y_test':yT_test.copy(), 'X_full':XT, 'y_full':yT}
datasetsPCA['Wilt']={'X_train':XWpca.copy(), 'y_train':yW_train.copy(), 'X_test':pcaW.transform(XW_test), 'y_test':yW_test.copy()}
clusters =  [2,3,4,5,6,8,10,12,15,20,25,30,35,40,50]
scoresPCA = hlp.explore_clustering(datasetsPCA, clusters)
scoresPCA['Titanic']['kmeans'].keys()
hlp.plot_CE([scoresPCA,scores], clusters, 'DR using PCA')

# Plot Silhouettes
from silhouette import plot_silhouettes
from sklearn.neighbors.nearest_centroid import NearestCentroid
km = KMeans(random_state=6, n_init=10)
gmm = GaussianMixture(random_state=6, n_init=1)

def plot_sil(X, mod, t):
    mod.fit(X)
    mod.score(X)
    pred = mod.predict(X)
    clf = NearestCentroid()
    clf.fit(X, pred)
    plot_silhouettes(X, pred, clf.centroids_, title=t)                  

km.set_params(n_clusters=5)
t = "Silhouette Analysis, Titanic k-Means with n_clusters = %d" % km.n_clusters
X = datasetsPCA['Titanic']['X_train']
plot_sil(X, km, t)

gmm.set_params(n_components=15)
t = "Post-PCA Silhouette Analysis, Titanic GMM with %d components" % gmm.n_components
plot_sil(X, gmm, t)
#plt.gca().set_xlim([-0.6,1.5])

km.set_params(n_clusters=6)
t = "Silhouette Analysis, Wilt k-Means with n_clusters = %d" % km.n_clusters
X = datasetsPCA['Wilt']['X_train']
plot_sil(X, km, t)
gmm.set_params(n_components=6)
t = "Post-PCA Silhouette Analysis, Wilt GMM with %d components" % gmm.n_components
plot_sil(X, gmm, t)

# average silhouette score per cluster
from sklearn.metrics import silhouette_samples

def cluster_scores(X, y, pred):
    acc, H0, H1, cluster_summary = hlp.cluster_acc(y, pred)
    df = pd.DataFrame(cluster_summary)[['Size','Pos','Neg','H', 'Acc']]  #.sort_values(['Acc'], ascending=False)
    df2 = pd.DataFrame(pred,silhouette_samples(X, pred)).reset_index()[[0,'index']]
    df2.columns=['label','silh']
    df['Silh']=df2.groupby('label').mean()
    return df.sort_values(['Acc'], ascending=False)

# Wilt k-means 6 clusters
X = datasetsPCA['Wilt']['X_train']
km.set_params(n_clusters=6)
km.fit(X)
pred = km.predict(X)
y = datasetsPCA['Wilt']['y_train']
df=cluster_scores(X, y, pred)
df

# Wilt GMM 6 clusters
X = datasetsPCA['Wilt']['X_train']
gmm.set_params(n_components=50)
gmm.fit(X)
pred = gmm.predict(X)
y = datasetsPCA['Wilt']['y_train']
df=cluster_scores(X, y, pred)
df

import seaborn as sns
g = sns.pairplot(df[['Acc','Silh']])
g.fig.suptitle('Accuracy vs Silhouette Scores', size='medium')
g.fig.subplots_adjust(top=0.93)

# Titanic k-means 5 clusters
X = datasetsPCA['Titanic']['X_train']
km.set_params(n_clusters=5)
km.fit(X)
pred = km.predict(X)
y = datasetsPCA['Titanic']['y_train']
df=cluster_scores(X, y, pred)
df

# Titanic GMM 5 clusters
X = datasetsPCA['Titanic']['X_train']
gmm.set_params(n_components=5)
gmm.fit(X)
pred = gmm.predict(X)
y = datasetsPCA['Titanic']['y_train']
df=cluster_scores(X, y, pred)
df

#%% Run ANN from A1 on Titanic
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

sz=np.linspace(0.1, 1.0, 20)   #training set sizes
mlp = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=45, random_state=0, activation='logistic', solver='lbfgs')

X_train = datasetsPCA['Titanic']['X_train']
y_train = datasetsPCA['Titanic']['y_train']
X_test = datasetsPCA['Titanic']['X_test']
y_test = datasetsPCA['Titanic']['y_test']

tsz, tr, va = learning_curve(mlp, X_train, y_train, train_sizes=sz, cv=10, n_jobs=1)
fig = hlp.plot_LC(mlp, tsz, tr, va, "Titanic LC, MLP after PCA") #, h=3, w=4)
fig.show()

np.mean(cross_val_score(mlp, X_train, y_train, cv=10))
np.std(cross_val_score(mlp, X_train, y_train, cv=10))
mlp.fit(X_train, y_train)
mlp.score(X_train, y_train)
mlp.score(X_test, y_test)

f1_score(y_train, y_pred=mlp.predict(X_train))
f1_score(y_test, y_pred=mlp.predict(X_test))
y_pred = cross_val_predict(mlp, X_train, y_train, cv=10)
print(classification_report(y_train, y_pred))
hlp.summarise_CM(confusion_matrix(y_train, y_pred), 'CV')
y_pred = mlp.predict(X_test) 
print(classification_report(y_test, y_pred))
hlp.summarise_CM(confusion_matrix(y_test, y_pred), 'Test')
X_train.shape

hlp.summarise_CM(confusion_matrix(y_true=y_train, y_pred=mlp.predict(X_train)), 'training')
hlp.summarise_CM(confusion_matrix(y_true=y_test, y_pred=mlp.predict(X_test)), 'Test')
perf = hlp.pred_perf(X_test, y_test, mlp, digits=5)
hlp.clf_test_score(mlp, X_train, y_train, X_test, y_test)

#%% Part 5 Use Clusters as features on ANN

# Build training set, scale, and transform )

# Scaling and PCA and clustering allowed on all data (no labels required)
pcaT = PCA(random_state=54, n_components=5)
StdSc = StandardScaler()
X_all = datasetsPCA['Titanic']['X_full']   # was scaled previously (before PCA/DR)
X_all.shape
yT = datasetsPCA['Titanic']['y_full']
#X_all = StdSc.fit(X_all)
#pcaT.fit(X_all)    # just in case we lost previously fitted pcaT
# data from DR after pcaT previously stored:
#X_train = datasetsPCA['Titanic']['X_train']
#X_train.shape
#X_test = datasetsPCA['Titanic']['X_test']
#y_train = datasetsPCA['Titanic']['y_train']
#XTpca = pcaT.transform(X_train)
#XTproj = pcaT.inverse_transform(XTpca)
#np.cumsum(pcaT.explained_variance_ratio_)
#((X_train - XTproj) ** 2).mean()      # verify reconstruction error 

fn=['f1','f2','f3','f4','f5', 'c1','c2','c3','c4','c5']
# Redo cluster assignment based on all data
X_all = pcaT.transform(X_all)
km.set_params(n_clusters=5)
km.fit(X_all)
pred = km.predict(X_all)              # new clusters
# add clusters as new features to data and rescale
X_all = StdSc.fit_transform(np.concatenate([X_all,pd.get_dummies(pred)], axis=1))
X_all.shape
X_train, X_test, y_train, y_test = train_test_split(X_all, yT, test_size=0.3, random_state=1, stratify=yT)
X_train.shape

# Analyze new dataset with clusters added
import matplotlib as mpl
tsz, tr, va = learning_curve(mlp, X_train, y_train, train_sizes=sz, cv=10, n_jobs=1)
fig = hlp.plot_LC(mlp, tsz, tr, va, "Titanic LC, MLP after PCA+clusters") #, h=3, w=4)
ax = fig.gca()
ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
plt.ylim(top=0.9)
fig.show()
X_train.shape
mlp.fit(X_train, y_train)
mlp.score(X_train, y_train)
mlp.score(X_test, y_test)
np.mean(cross_val_score(mlp, X_train, y_train, cv=10))
np.std(cross_val_score(mlp, X_train, y_train, cv=10))

f1_score(y_train, y_pred=mlp.predict(X_train))
f1_score(y_test, y_pred=mlp.predict(X_test))
y_pred = cross_val_predict(mlp, X_train, y_train, cv=10)
print(classification_report(y_train, y_pred))
hlp.summarise_CM(confusion_matrix(y_train, y_pred), 'CV')
y_pred = mlp.predict(X_test) 
print(classification_report(y_test, y_pred))
hlp.summarise_CM(confusion_matrix(y_test, y_pred), 'Test')

