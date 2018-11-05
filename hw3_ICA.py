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
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import FastICA
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
#import seaborn as sns; sns.set()

# helpers.py should be in the same direcory as this script
this_dir='D:\\GoogleDrive\\_Study\\GT_ML\\hw3'
os.chdir(this_dir)
import helpers as hlp

#out = './{}/'.format(sys.argv[1])
out = '.\\out\\'
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
XT_train, XT_test, yT_train, yT_test = train_test_split(XT, yT, test_size=0.3, random_state=1, stratify=yT)

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

#%% Baseline scores
datasets={}
datasets['Titanic']={'X_train':XT_train.copy(), 'y_train':yT_train.copy(), 'X_test':XT_test.copy(), 'y_test':yT_test.copy()}
datasets['Wilt']={'X_train':XW_train.copy(), 'y_train':yW_train.copy(), 'X_test':XW_test.copy(), 'y_test':yW_test.copy()}

clusters =  [2,3,4,5,6,8,10,12,15,20,25,30,35,40,50]
scores = hlp.explore_clustering(datasets, clusters)

#%% Part 2 ICA

# ICA for Titanic
icaT = FastICA(random_state=54)
dims = [2,3,4,5,6,7,8,9,10]
kurtT = {}
for dim in dims:
    icaT.set_params(n_components=dim)
    trans = icaT.fit_transform(XT)
    proj = icaT.inverse_transform(trans)
    tmp = pd.DataFrame(trans)
    tmp = tmp.kurt(axis=0)
    rec_err = ((XT - proj)**2).mean()
    kurtT[dim] = (round(tmp.abs().mean(),3), round(tmp.abs().min(),3), round(rec_err,3), tmp)
kurtT = pd.Series(kurtT) 
kurtT  # examine average and minimum kurtosis
kurtT[7]
kurtT[4]

# check what kurt returns on a normal distribution:
pd.DataFrame(np.random.normal(155, 72, 100000)).kurt(axis=0)

icaW = FastICA(random_state=54)
dims = [2,3,4,5]
kurtW = {}
for dim in dims:
    icaW.set_params(n_components=dim)
    trans = icaW.fit_transform(XW_train)
    proj = icaW.inverse_transform(trans)
    tmp = pd.DataFrame(trans)
    tmp = tmp.kurt(axis=0)
    rec_err = ((XW_train - proj)**2).mean()
    kurtW[dim] = (round(tmp.abs().mean(),3), round(tmp.abs().min(),3), round(rec_err,3), tmp)
kurtW = pd.Series(kurtW) 
kurtW  # examine average and minimum kurtosis
kurtW[4]

icaT4 = FastICA(random_state=54)
icaT4.set_params(n_components=4)
XT4 = icaT4.fit_transform(XT)

icaT7 = FastICA(random_state=54)
icaT7.set_params(n_components=7)
XT7 = icaT7.fit_transform(XT)

icaW2 = FastICA(random_state=54)
icaW2.set_params(n_components=2)
XW2 = icaW2.fit_transform(XW_train)

icaW4 = FastICA(random_state=54)
icaW4.set_params(n_components=4)
XW4 = icaW4.fit_transform(XW_train)


#%% Create dataset with DR
datasetsICA={}
XT_train, XT_test, yT_train, yT_test = train_test_split(XT4, yT, test_size=0.3, random_state=1, stratify=yT)
datasetsICA['Titan4']={'X_train':XT_train.copy(), 'y_train':yT_train.copy(), 'X_test':XT_test.copy(), 'y_test':yT_test.copy(), 'X_full':XT, 'y_full':yT}
XT_train, XT_test, yT_train, yT_test = train_test_split(XT7, yT, test_size=0.3, random_state=1, stratify=yT)
datasetsICA['Titan7']={'X_train':XT_train.copy(), 'y_train':yT_train.copy(), 'X_test':XT_test.copy(), 'y_test':yT_test.copy(), 'X_full':XT, 'y_full':yT}
datasetsICA['Wilt2']={'X_train':XW2.copy(), 'y_train':yW_train.copy(), 'X_test':icaW2.transform(XW_test), 'y_test':yW_test.copy()}
datasetsICA['Wilt4']={'X_train':XW4.copy(), 'y_train':yW_train.copy(), 'X_test':icaW4.transform(XW_test), 'y_test':yW_test.copy()}

datasetsICA={}
XT_train, XT_test, yT_train, yT_test = train_test_split(XT4, yT, test_size=0.3, random_state=1, stratify=yT)
datasetsICA['Titanic']={'X_train':XT_train.copy(), 'y_train':yT_train.copy(), 'X_test':XT_test.copy(), 'y_test':yT_test.copy(), 'X_full':XT, 'y_full':yT}
datasetsICA['Wilt']={'X_train':XW2.copy(), 'y_train':yW_train.copy(), 'X_test':icaW2.transform(XW_test), 'y_test':yW_test.copy()}

#%% Clustering after DR
clusters =  [2,3,4,5,6,8,10,12,15,20,25,30,35,40,50]
scoresICA = hlp.explore_clustering(datasetsICA, clusters)

datasetsICA2={}
XT_train, XT_test, yT_train, yT_test = train_test_split(XT7, yT, test_size=0.3, random_state=1, stratify=yT)
datasetsICA2['Titanic']={'X_train':XT_train.copy(), 'y_train':yT_train.copy(), 'X_test':XT_test.copy(), 'y_test':yT_test.copy(), 'X_full':XT, 'y_full':yT}
datasetsICA2['Wilt']={'X_train':XW4.copy(), 'y_train':yW_train.copy(), 'X_test':icaW4.transform(XW_test), 'y_test':yW_test.copy()}
scoresICA2 = hlp.explore_clustering(datasetsICA2, clusters)

hlp.plot_CE([scoresICA], clusters, 'DR using ICA')
hlp.plot_CE([scoresICA, scoresICA2], clusters, 'DR using ICA')
plt.suptitle('ICA analysis - (solid) 4 components [Titanic] or 2 [Wilt]  (dotted) 7 [Titanic] or 4 [Wilt]')
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

km.set_params(n_clusters=7)
t = "Silhouette Analysis, Titanic k-Means with n_clusters = %d" % km.n_clusters
X = datasetsICA['Titanic']['X_train']
plot_sil(X, km, t)

gmm.set_params(n_components=15)
t = "Post-ICA Silhouette Analysis, Titanic GMM with %d components" % gmm.n_components
plot_sil(X, gmm, t)
#plt.gca().set_xlim([-0.6,1.5])

km.set_params(n_clusters=6)
t = "Silhouette Analysis, Wilt k-Means with n_clusters = %d" % km.n_clusters
X = datasetsICA['Wilt']['X_train']
plot_sil(X, km, t)
gmm.set_params(n_components=6)
t = "Post-ICA Silhouette Analysis, Wilt GMM with %d components" % gmm.n_components
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
X = datasetsICA['Wilt']['X_train']
km.set_params(n_clusters=6)
km.fit(X)
pred = km.predict(X)
y = datasetsICA['Wilt']['y_train']
df=cluster_scores(X, y, pred)
df

# Wilt GMM 6 clusters
X = datasetsICA['Wilt']['X_train']
gmm.set_params(n_components=50)
gmm.fit(X)
pred = gmm.predict(X)
y = datasetsICA['Wilt']['y_train']
df=cluster_scores(X, y, pred)
df

import seaborn as sns
g = sns.pairplot(df[['Acc','Silh']])
g.fig.suptitle('Accuracy vs Silhouette Scores', size='medium')
g.fig.subplots_adjust(top=0.93)

# Titanic k-means 5 clusters
X = datasetsICA['Titanic']['X_train']
km.set_params(n_clusters=5)
km.fit(X)
pred = km.predict(X)
y = datasetsICA['Titanic']['y_train']
df=cluster_scores(X, y, pred)
df

# Titanic GMM 5 clusters
X = datasetsICA['Titanic']['X_train']
gmm.set_params(n_components=5)
gmm.fit(X)
pred = gmm.predict(X)
y = datasetsICA['Titanic']['y_train']
df=cluster_scores(X, y, pred)
df

#%% Run ANN from A1 on Titanic
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

sz=np.linspace(0.1, 1.0, 20)   #training set sizes
mlp = MLPClassifier(hidden_layer_sizes=(10,10), max_iter=45, random_state=0, activation='logistic', solver='lbfgs')
mlp
X_train = datasetsICA2['Titanic']['X_train']
y_train = datasetsICA2['Titanic']['y_train']
X_test = datasetsICA2['Titanic']['X_test']
y_test = datasetsICA2['Titanic']['y_test']
X_train = datasetsICA['Titanic']['X_train']
y_train = datasetsICA['Titanic']['y_train']
X_test = datasetsICA['Titanic']['X_test']
y_test = datasetsICA['Titanic']['y_test']

cv_sets = hlp.get_cv_sets(X_train,y_train, k=10, rs=0)
def mlp_title(clf):
    a = clf.activation 
    b = clf.hidden_layer_sizes 
    c = clf.max_iter
    d = clf.learning_rate 
    return "'act.=" + a + "', layers=%s, iter=%s, lr=%s)" % (b,c,d)
# TITANIC GRID SEARCH- start with wide coarse grid then proressively narrow down
# set parameter space 
rsv = [0]
#cv = [c/100 for c in range(1,600,20)]; gv = [g/1000 for g in range(1,600,20)]  #shows cliff wes of C=0.6 !
act = ['relu','identity','logistic','tanh']
sol = ['lbfgs','sgd','adam']
lr = ['constant', 'invscaling', 'adaptive']
act = ['logistic']
sol = ['lbfgs']
mi = [25,50,75,100,150,300]
lr = ['constant']
hls = [(7),(8),(9),(7,7),(8,8),(9,9)]
GS_params = {'hidden_layer_sizes':hls, 'activation': act, 'solver': sol, 'max_iter': mi, 'random_state': rsv, 'shuffle':[True], 'learning_rate': lr}
clf = hlp.run_GS(MLPClassifier(), GS_params, X_train, y_train, cv_sets)
hlp.GS_summary(clf, X_train, y_train)
#perf = hlp.pred_perf(X_train, y_train, clf, digits=5)
mlp = clf.best_estimator_
tsz, tr, va = learning_curve(mlp, X_train, y_train, train_sizes=sz, cv=cv_sets, n_jobs=1)
fig = hlp.plot_LC(mlp, tsz, tr, va, "Learning Curve (mlp " + mlp_title(mlp))
plt.ylim(top=0.9)
mlp4 = clf.best_estimator_
np.mean(cross_val_score(mlp4, X_train, y_train, cv=10))

np.mean(cross_val_score(mlp, X_train, y_train, cv=10))
np.std(cross_val_score(mlp, X_train, y_train, cv=10))
mlp.fit(X_train, y_train)
mlp.score(X_train, y_train)
mlp.score(X_test, y_test)
#f1_score(y_train, y_pred=mlp.predict(X_train))
#f1_score(y_test, y_pred=mlp.predict(X_test))
y_pred = cross_val_predict(mlp, X_train, y_train, cv=10)
print(classification_report(y_train, y_pred))
hlp.summarise_CM(confusion_matrix(y_train, y_pred), 'CV')
y_pred = mlp.predict(X_test) 
print(classification_report(y_test, y_pred))
hlp.summarise_CM(confusion_matrix(y_test, y_pred), 'Test')
X_train.shape
datasetsICA['Titanic']['X_train'][:,:5]
datasetsICA2['Titanic']['X_train'][:,:5]

hlp.summarise_CM(confusion_matrix(y_true=y_train, y_pred=mlp.predict(X_train)), 'training')
hlp.summarise_CM(confusion_matrix(y_true=y_test, y_pred=mlp.predict(X_test)), 'Test')
perf = hlp.pred_perf(X_test, y_test, mlp, digits=5)
hlp.clf_test_score(mlp, X_train, y_train, X_test, y_test)

#%% Part 5 Use Clusters as features on ANN

# Build training set, scale, and transform )

# Scaling and ICA and clustering allowed on all data (no labels required)
icaT = FastICA(random_state=54, n_components=5)
StdSc = StandardScaler()
X_all = datasetsICA['Titanic']['X_full']   # was scaled previously (before ICA/DR)
X_all.shape
yT = datasetsICA['Titanic']['y_full']
#X_all = StdSc.fit(X_all)
#icaT.fit(X_all)    # just in case we lost previously fitted icaT
# data from DR after icaT previously stored:
#X_train = datasetsICA['Titanic']['X_train']
#X_train.shape
#X_test = datasetsICA['Titanic']['X_test']
#y_train = datasetsICA['Titanic']['y_train']
#XTica = icaT.transform(X_train)
#XTproj = icaT.inverse_transform(XTica)
#np.cumsum(icaT.explained_variance_ratio_)
#((X_train - XTproj) ** 2).mean()      # verify reconstruction error 

fn=['f1','f2','f3','f4','f5', 'c1','c2','c3','c4','c5']
# Redo cluster assignment based on all data
X_all = icaT.transform(X_all)
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
fig = hlp.plot_LC(mlp, tsz, tr, va, "Titanic LC, MLP after ICA+clusters") #, h=3, w=4)
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

