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
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from sklearn.pipeline import Pipeline

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

#%% Part 2 RCA

from itertools import product
from sklearn.metrics.pairwise import pairwise_distances
from collections import defaultdict
import scipy.sparse as sps
from scipy.linalg import pinv
from numpy import mean

#%% jontay functions
def pairwiseDistCorr(X1,X2):
    assert X1.shape[0] == X2.shape[0]
    
    d1 = pairwise_distances(X1)
    d2 = pairwise_distances(X2)
    return np.corrcoef(d1.ravel(),d2.ravel())[0,1]

def reconstructionError(projections,X):
    W = projections.components_
    if sps.issparse(W):
        W = W.todense()
    p = pinv(W)
    reconstructed = ((p@W)@(X.T)).T # Unproject projected data
    errors = np.square(X-reconstructed)
    return np.nanmean(errors)

#%% evaluate RP pairwise correlation and recon_error

def eval_RP(X_train, dims):
    tmp = defaultdict(dict)
    for d in dims:
        pdc = 0; rec_err=0
        tmp[d]['pdc'] = []
        tmp[d]['rec'] = []
        for i in range(30):
            rp = GaussianRandomProjection(random_state=i, n_components=d)
            trans = rp.fit_transform(X_train)
            pdc = pairwiseDistCorr(trans, X_train)
            rec_err = reconstructionError(rp, X_train)
            tmp[d]['pdc'].append(round(pdc,4))
            tmp[d]['rec'].append(round(rec_err,4))
    pd.DataFrame(tmp).T
    
    tmp_sum = defaultdict(dict)
    for d in dims:
        pdc = 0; rec_err=0
        print (d, round(mean(tmp[d]['pdc']),3), round(np.std(tmp[d]['pdc']),3), round(mean(tmp[d]['rec']),3))
        tmp_sum[d]=(d, round(mean(tmp[d]['pdc']),3), round(np.std(tmp[d]['pdc']),3), round(mean(tmp[d]['rec']),3))
    return tmp, tmp_sum

a3 = [2,3,4,5,6,7,8,9,10]
a1, a2 = eval_RP(XT_train, a3)
y = [a2[d][1] for d in a3]
sd = [a2[d][2] for d in a3]
ra = [1 - a2[d][3] for d in a3]
fig = plt.figure('Titanic RP Stats Gaussian')
plt.errorbar(a3, y, yerr=sd)
plt.plot(a3, ra, '-o', markersize=3)
plt.title('Titanic RP Stats (30 samples, Gaussian)')
plt.legend(['reconstruction accuracy','pairwise distance correlation (with sd)'], frameon=False, fontsize='medium')
plt.grid(True, linestyle='-', linewidth='0.15')

b3 = [2,3,4,5]
b1,b2 = eval_RP(XW_train, b3)
y = [b2[d][1] for d in b3]
sd = [b2[d][2] for d in b3]
ra = [1 - b2[d][3] for d in b3]
fig = plt.figure('Wilt RP Stats Gaussian')
plt.errorbar(b3, y, yerr=sd)
plt.plot(b3, ra, '-o', markersize=3)
plt.title('Wilt RP Stats (30 samples, Gaussian)')
plt.legend(['reconstruction accuracy','pairwise distance correlation (with sd)'], frameon=False, fontsize='medium')
plt.grid(True, linestyle='-', linewidth='0.15')

datasetsRCA={}
datasetsRCA['Titanic']={'X_train':XT_train.copy(), 'y_train':yT_train.copy(), 'X_test':XT_test.copy(), 'y_test':yT_test.copy(), 'X_full':XT, 'y_full':yT}
datasetsRCA['Wilt']={'X_train':XW_train.copy(), 'y_train':yW_train.copy(), 'X_test':XW_test, 'y_test':yW_test.copy()}

#%% Run ANN from A1 on Titanic
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

sz=np.linspace(0.1, 1.0, 20)   #training set sizes
#mlp = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=45, random_state=0, activation='logistic', solver='lbfgs')

def mlp_title(clf):
    a = clf.activation 
    b = clf.hidden_layer_sizes 
    c = clf.max_iter
    d = clf.learning_rate 
    return "'act.=" + a + "', layers=%s, iter=%s, lr=%s)" % (b,c,d)

# TITANIC GRID SEARCH- start with wide coarse grid then proressively narrow down
X_train = datasetsRCA['Titanic']['X_train']
y_train = datasetsRCA['Titanic']['y_train']
X_test = datasetsRCA['Titanic']['X_test']
y_test = datasetsRCA['Titanic']['y_test']
cv_sets = hlp.get_cv_sets(X_train,y_train, k=10, rs=0)
# WILT
X_train = datasetsRCA['Wilt']['X_train']
y_train = datasetsRCA['Wilt']['y_train']
X_test = datasetsRCA['Wilt']['X_test']
y_test = datasetsRCA['Wilt']['y_test']
cv_sets = hlp.get_cv_sets(X_train,y_train, k=5, rs=0)

# set parameter space 
rsv = list(range(0,30))
#act = ['relu','identity','logistic','tanh']
#sol = ['lbfgs','sgd','adam']
#lr = ['constant', 'invscaling', 'adaptive']
#act = ['logistic']
#sol = ['lbfgs']
mi=[25]
mi = [25,35,50,75,100,150]
#lr = ['constant']
hls = [(10),(11),(9),(11,8),(9,9),(10,10,10)]
hls = [(4),(3),(4,4),(3,4),(10,10,10)]
hls = [(4),(5)]
rp = GaussianRandomProjection(n_components=3, random_state=14)
mlp = MLPClassifier(activation='logistic',solver='lbfgs',early_stopping=True,random_state=5, shuffle=True, learning_rate='constant')
pipe = Pipeline([('rp',rp),('NN',mlp)])
GS_params = {'NN__hidden_layer_sizes':hls, 'NN__max_iter': mi}
clf = hlp.run_GS(pipe, GS_params, X_train, y_train, cv_sets, scoring='accuracy')
hlp.GS_summary(clf, X_train, y_train)
# cycle through above settings and store results below
T7 = clf
T8 = clf
T9 = clf
T9.best_estimator_
W3 = clf
W4 = clf
mlp = T8.best_estimator_
tsz, tr, va = learning_curve(mlp, X_train, y_train, train_sizes=sz, cv=cv_sets, n_jobs=1)
fig = hlp.plot_LC(mlp.steps[1][1], tsz, tr, va, "Learning Curve (mlp " + mlp_title(mlp.steps[1][1]))
plt.ylim(top=0.9)
X_train.shape

np.mean(cross_val_score(mlp, X_train, y_train, cv=5))
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

hlp.summarise_CM(confusion_matrix(y_true=y_train, y_pred=mlp.predict(X_train)), 'training')
hlp.summarise_CM(confusion_matrix(y_true=y_test, y_pred=mlp.predict(X_test)), 'Test')
perf = hlp.pred_perf(X_test, y_test, mlp, digits=5)
hlp.clf_test_score(mlp, X_train, y_train, X_test, y_test)

#%% Create dataset with DR
datasetsRCA={}
rp=T8.best_estimator_.steps[0][1]
X = rp.fit_transform(XT,yT)
X = StandardScaler().fit_transform(X)
XT_train, XT_test, yT_train, yT_test = train_test_split(X, yT, test_size=0.3, random_state=1, stratify=yT)
datasetsRCA['Titanic']={'X_train':XT_train.copy(), 'y_train':yT_train.copy(), 'X_test':XT_test.copy(), 'y_test':yT_test.copy(), 'X_full':XT, 'y_full':yT}
rp=W4.best_estimator_.steps[0][1]
datasetsRCA['Wilt']={'X_train':rp.fit_transform(XW_train), 'y_train':yW_train.copy(), 'X_test':rp.transform(XW_test), 'y_test':yW_test.copy()}

#%% Clustering after DR
clusters =  [2,3,4,5,6,8,10,12,15,20,25,30,35,40,50]
scoresRCA = hlp.explore_clustering(datasetsRCA, clusters)
pd.DataFrame(scores)

hlp.plot_CE([scores], clusters, 'DR using RCA (RP)')
hlp.plot_CE([scoresRCA, scores], clusters, 'DR using RCA (RP) vs baseline')
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
X = datasetsRCA['Titanic']['X_train']
plot_sil(X, km, t)

gmm.set_params(n_components=15)
t = "Post-ICA Silhouette Analysis, Titanic GMM with %d components" % gmm.n_components
plot_sil(X, gmm, t)
#plt.gca().set_xlim([-0.6,1.5])

km.set_params(n_clusters=6)
t = "Silhouette Analysis, Wilt k-Means with n_clusters = %d" % km.n_clusters
X = datasetsRCA['Wilt']['X_train']
plot_sil(X, km, t)
gmm.set_params(n_components=6)
t = "Post-ICA Silhouette Analysis, Wilt GMM with %d components" % gmm.n_components
plot_sil(X, gmm, t)

# average silhouette score per cluster

# Wilt GMM 4 clusters
gmm = GaussianMixture(random_state=6, n_init=1)
X = datasetsRCA['Wilt']['X_train']
gmm.set_params(n_components=4)
gmm.fit(X)
pred = gmm.predict(X)
y = datasetsRCA['Wilt']['y_train']
df=hlp.cluster_scores(X, y, pred)
df



