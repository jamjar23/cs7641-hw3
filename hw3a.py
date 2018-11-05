# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 14:19:25 2018

@author: James
"""

import os
import pandas as pd
import numpy as np
#from numpy import mean
import matplotlib.pyplot as plt 
#%matplotlib qt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from collections import defaultdict
from sklearn.metrics import adjusted_mutual_info_score
#from sklearn.model_selection import learning_curve
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import confusion_matrix
import time
#from sklearn.neural_network import MLPClassifier
#from sklearn.manifold import TSNE
#from sklearn.pipeline import Pipeline
from collections import defaultdict
#from helpers import cluster_acc, myGMM,nn_arch,nn_reg
from sklearn.metrics import adjusted_mutual_info_score

# helpers.py should be in the same direcory as this script
this_dir='D:\\GoogleDrive\\_Study\\GT_ML\\hw3'
os.chdir(this_dir)
import helpers as hlp

np.set_printoptions(precision=3, suppress=True)
#pd.options.display.float_format = lambda x: '{:,.0f}'.format(x) if x > 1e3 else '{:,.2f}'.format(x)
pd.options.display.float_format = lambda x: '{:,.4f}'.format(x)

# change display settings
pd.set_option('display.width', 999)
pd.set_option('display.max_columns', 99)
pd.set_option('display.max_rows', 40)

#out = './{}/'.format(sys.argv[1])
out = 'D:\\GoogleDrive\\_Study\\GT_ML\\hw3\\out'

np.random.seed(54)

# Load Titanic
dfT, _ = hlp.load_titanic(this_dir + '\\titanic\\train.csv')
fnT = list(dfT.columns)[1:]     # feature names
# convert to arrays
XT = dfT.iloc[:, 1:].values
yT = dfT.iloc[:, 0].values
# scale Titanic data
XT_std = StandardScaler().fit_transform(XT)
# split into training and test sets
XT_train, XT_test, yT_train, yT_test = train_test_split(XT_std, yT, test_size=0.3, random_state=1, stratify=yT)

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

datasets={}
datasets['Titanic']={'X_train':XT_train.copy(), 'y_train':yT_train.copy(), 'X_test':XT_test.copy(), 'y_test':yT_test.copy()}
datasets['Wilt']={'X_train':XW_train.copy(), 'y_train':yW_train.copy(), 'X_test':XW_test.copy(), 'y_test':yW_test.copy()}
#datasets.keys()


#%% Data for 1-3
clusters =  [2,3,4,5,6,8,10,12,15,20,25,30,35,40,50]
scores = hlp.explore_clustering(datasets, clusters)

# Print Charts 
hlp.plot_CE([scores], clusters, 'DR using RCA (RP)')

# play around
km = KMeans(random_state=6, n_init=10)
gmm = GaussianMixture(random_state=6, n_init=1)
km.set_params(n_clusters=5)
km.fit(datasets['Titanic']['X_train'])
km.score(datasets['Titanic']['X_train'])
X_train = datasets['Titanic']['X_train']
y_train = datasets['Titanic']['y_train']
acc, H0, H1 = hlp.cluster_acc(y_train,km.predict(X_train))
H0-H1
adjusted_mutual_info_score(y_train,km.predict(X_train))
gmm.set_params(n_components=5)
gmm.fit(datasets['Titanic']['X_train'])
gmm.score(datasets['Titanic']['X_train'])
gmm.bic(X_train)

# Silhouette plots
from sklearn.neighbors.nearest_centroid import NearestCentroid
# set m = either km or gmm
gmm.set_params(n_components=35)
km.set_params(n_clusters=60)
m=gmm
# fit & plot
#X_train = datasets['Titanic']['X_train']
X_train = datasets['Wilt']['X_train']
#X_test = datasets['Titanic']['X_test']
X=X_train
m.fit(X)
m.score(X)
pred = m.predict(X)
clf = NearestCentroid()
clf.fit(X, pred)
#hlp.plot_sil(X, pred, clf.centroids_, fnT, ['Fare','Age'])  # Titanic
from silhouette import plot_silhouettes
t = "Silhouette Analysis, Wilt GMM with n_clusters = %d" % len(clusters)
plot_silhouettes(X, pred, clf.centroids_, fnW, ['Mean_Red','Mean_NIR'], t)                  # Wilt
plt.gca().set_xlim([-0.6,1.5])

y_train = datasets['Titanic']['y_train']
y_train = datasets['Wilt']['y_train']
y_test = datasets['Titanic']['y_test']
y=y_train
acc, H0, H1, cluster_summary = hlp.cluster_acc(y, pred)
df = pd.DataFrame(cluster_summary)[['Size','Pos','Neg','H', 'Acc']]  #.sort_values(['Acc'], ascending=False)
acc

# average silhouette score per cluster
from sklearn.metrics import silhouette_samples
silhouette_samples(X, pred)
df2 = pd.DataFrame(pred,silhouette_samples(X, pred)).reset_index()[[0,'index']]
df2.columns=['label','silh']
df['Silh']=df2.groupby('label').mean()
df.sort_values(['Acc'], ascending=False)

import seaborn as sns
g = sns.pairplot(df[['Acc','Silh']])
g.fig.suptitle('Accuracy vs Silhouette Scores')


#%% Use clusters as DR features!  (Part 5)
# https://piazza.com/class/jkr51z4y8ez58h?cid=1129

X_train = datasets['Titanic']['X_train']
y_train = datasets['Titanic']['y_train']
y_test = datasets['Titanic']['y_test']
X_test = datasets['Titanic']['X_test']

sz=np.linspace(0.1, 1.0, 20)   #training set sizes
gmm.set_params(n_components=8)
km.set_params(n_clusters=5)
mlp = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=45, random_state=0, activation='logistic', solver='lbfgs')

Xk = km.fit_transform(X_train, y_train)   # distances to cluster centers
ssk = StandardScaler()
Xk = ssk.fit_transform(Xk)
predk = km.predict(X_train)

Xgf = gmm.fit(X_train, y_train)   # distances to cluster centers
Xg = Xgf._estimate_log_prob(X_train)
ssg = StandardScaler()
Xg = ssg.fit_transform(Xg)
predg = gmm.predict(X_train)

mi = [25,35,50,75,100,150]
hls = [(10),(8),(9),(8,8),(9,9),(10,10,10)]
hls = [(5),(6),(7),(8),(9),(5,5),(6,6),(10,10,10)]
mlp = MLPClassifier(activation='logistic',solver='lbfgs',early_stopping=True,random_state=5, shuffle=True, learning_rate='constant')
GS_params = {'hidden_layer_sizes':hls, 'max_iter': mi}
clf = hlp.run_GS(mlp, GS_params, Xk, y_train, 10, scoring='accuracy')
hlp.GS_summary(clf, X_train, y_train)
mlp_k5 = clf.best_estimator_
mlp_g8 = clf.best_estimator_
be = clf.best_estimator_
X=Xk
X=Xg

tsz, tr, va = learning_curve(be, X, y_train, train_sizes=sz, cv=10, n_jobs=1)
fig = hlp.plot_LC(be, tsz, tr, va, "Learning Curve (mlp " + mlp_title(be))
plt.ylim(top=0.9)
plt.grid(True, linewidth='0.15')
plt.title("GMM 8 cluster LC (mlp " + mlp_title(be), fontsize='small')
X_train.shape

np.mean(cross_val_score(be, X_train, y_train, cv=5))
np.std(cross_val_score(be, X_train, y_train, cv=10))
be.fit(X_train, y_train)
be.score(X_train, y_train)
be.score(X_test, y_test)
#f1_score(y_train, y_pred=mlp.predict(X_train))
#f1_score(y_test, y_pred=mlp.predict(X_test))
y_pred = cross_val_predict(be, X_train, y_train, cv=10)
print(classification_report(y_train, y_pred))
hlp.summarise_CM(confusion_matrix(y_train, y_pred), 'CV')
y_pred = be.predict(X_test) 
print(classification_report(y_test, y_pred))
hlp.summarise_CM(confusion_matrix(y_test, y_pred), 'Test')
X_train.shape

