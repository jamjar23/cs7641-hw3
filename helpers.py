# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 17:05:14 2018

@author: James
"""

import os
import pandas as pd
import numpy as np
from numpy import mean
import matplotlib.pyplot as plt 
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from collections import defaultdict
from sklearn.metrics import adjusted_mutual_info_score
#%matplotlib qt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.image as mpimg
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
import time
from mpl_toolkits import mplot3d
from sklearn.metrics import silhouette_samples

def load_titanic(file_path):
    # load and prepare the titanic data, given full file_path
    fullset = pd.read_csv(file_path)
    df = fullset
    
    # replace handful of 0.00 fares with average based on class and port of embarkation
    df.Fare.replace({0.0:np.nan}, inplace=True)
    df.loc[(df.Fare!=df.Fare), 'Fare'] = df.groupby(['Pclass','Embarked'])['Fare'].transform('mean')
    
    # set up Age Unknown flag
    df['AgeUnk']=0
    df.loc[(df.Age!=df.Age), 'AgeUnk'] = 1
    
    # fix Age NaN (else can't scale later)
    # input based on similar Sex, SibSp, Parch
    # find Mean and StD for similar group
    # careful! ths operation messes up original sorting!
    lu = df.groupby(['Sex','SibSp','Parch'])[['Age']].agg(['mean','std'])
    df = pd.merge(df, lu, how='outer', on=['Sex', 'SibSp', 'Parch'])

    # re-sort by PassengerId, then drop 
    df.sort_values(['PassengerId'], inplace=True)
    df = df.drop(columns=['PassengerId','Name','Ticket','Cabin'])

    # for missing Mean Age (8 SibSp) assume mean=7
    # for missing StD Age, assume StD = 0.3 * Mean Age
    df.loc[(df[('Age', 'mean')]!=df[('Age', 'mean')]), ('Age', 'mean')] = 7
    df.loc[(df[('Age', 'std')]!=df[('Age', 'std')]), ('Age', 'std')] = 0.3 * df[('Age', 'mean')]
    # select random age based on Mean, StD
    np.random.seed(42)
    a = df[(df.Age!=df.Age)][('Age', 'mean')]
    b = df[(df.Age!=df.Age)][('Age', 'std')]
    df.loc[(df.Age!=df.Age), 'Age'] = np.random.normal(a, b)
    df = df.drop(columns=[('Age', 'mean'), ('Age', 'std')])
    
    # make dummy categorical columns\
    df['Pclass'] = df.Pclass.apply(str)
    df = pd.get_dummies(df, columns=['Sex','Pclass','Embarked'], drop_first=True)
    
    return df, fullset      #returns both modified data and original file 

#file_dir='D:\\GoogleDrive\\Study\\GT_ML\\hw1\\wilt'
def load_wilt(file_dir):
    # load and prepare the Wilt data, given file_dir
    # returns 4 arrays X_train, y_train, X_test, y_test + field names list 
    tr_set = pd.read_csv(file_dir + '\\training.csv')
    te_set = pd.read_csv(file_dir + '\\testing.csv')
    return tr_set.iloc[:,1:], tr_set.iloc[:,0:1], te_set.iloc[:,1:], te_set.iloc[:,0:1], list(tr_set.columns)[1:] 

def calc_H(dataset):
    labels = set(dataset)
    class_count = len(labels)
    H = 0
    if class_count==1:
        return H
    for c in labels:
        C = [b for b in dataset if b==c]
        sz = len(C)
        pC = sz / len(dataset)
        hC = - (pC * log(pC, class_count))
        H += hC
    return H

def explore_clustering(datasets, clusters):
    scores = defaultdict(lambda: defaultdict(dict))
    km = KMeans(random_state=6, n_init=10)
    gmm = GaussianMixture(random_state=6, n_init=1)
    
    for nm,ds in datasets.items():
        X_train = ds['X_train']
        y_train = ds['y_train']
        for m in [km, gmm]:
            st = time.clock()
            SSE=[]; acc=[]; sil=[]; ami=[]; IG=[]; BIC=[]
            for k in clusters:
                try:
                    m.set_params(n_clusters=k)
                    mod = 'kmeans'
                except:
                    m.set_params(n_components=k)
                    mod = 'GMM'
                m.fit(X_train)
                SSE.append(-m.score(X_train)) 
                pred = m.predict(X_train)
                if len(set(pred))>1:
                    sil.append(silhouette_score(X_train, pred))
                else:
                    sil.append(1)
                if mod=='GMM':  
                    BIC.append(m.bic(X_train))
                else:
                    BIC.append(0)
                accuracy, H0, H1, _ = cluster_acc(y_train, pred)
                acc.append(accuracy)
                IG.append(H0-H1)
                ami.append(adjusted_mutual_info_score(y_train, pred))
                print(nm, mod, k, '%.1f secs' % (time.clock()-st))
            scores[nm][mod]={'SSE/LL':SSE,'Silhouette':sil, 'BIC': BIC, 'AMI':ami, 'accuracy':acc, 'IG':IG}
    return scores

#df=SSE.reset_index(); t=''; w=5; h=4; pos=1
def plot_df(df, t, h=2, w=3, xlab='x', ylab='y', pos='best'):
    #df = df with titles, first col=x index, other cols = y values
    # begin plotting
    fig = plt.figure(t, figsize=(w, h))
    plt.clf()
    ax = fig.add_subplot(111)
    x = df.iloc[:,0].values
    s = list(df.columns[1:])
#    print (np.array(df[s]))
    ax.plot(x, np.array(df[s]),'-o', markersize=3)
    plt.title(t, size='small')
    plt.legend(s, loc=pos, fontsize='x-small', frameon=False, title='')
    plt.xlabel(xlab)
    plt.tick_params(labelsize=8)
    return fig

def axplot_df(ax, df, t, xlab='x', ylab='y', pos='best', linestyle='-'):
    #df = df with titles, first col=x index, other cols = y values
    plt.sca(ax)
    plt.gca().set_prop_cycle(None)
    x = df.iloc[:,0].values
    s = list(df.columns[1:])
    ax.plot(x, np.array(df[s]),'-o', markersize=3, linestyle=linestyle)
    plt.title(t, size='small')
    plt.legend(s, loc=pos, fontsize='x-small', frameon=False, title='')
#    plt.xlabel(xlab)
    plt.tick_params(labelsize=8)
    plt.grid(True)

def plot_CE(scores, clusters, t='Cluster Exploration'):
    # plot 8 cluster evaluation charts
    fig = plt.figure(t, figsize=(10, 6))
    ax1 = fig.add_subplot(241)
    ax2 = fig.add_subplot(242)
    ax3 = fig.add_subplot(243)
    ax4 = fig.add_subplot(244)
    ax5 = fig.add_subplot(245)
    ax6 = fig.add_subplot(246)
    ax7 = fig.add_subplot(247)
    ax8 = fig.add_subplot(248)

    for cyc, sc in enumerate(scores):
        linestyle=['-',':','--','-.'][cyc]
        plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.23, hspace=0.25)
        t = 'k-means, Titanic, SSE/10000'
        df=pd.DataFrame(sc['Titanic']['kmeans'], index=clusters)[['SSE/LL','Silhouette']]
        df[['SSE/LL']] = df[['SSE/LL']]/1e5
        axplot_df(ax1, df.reset_index(), t, xlab='k', linestyle=linestyle)
        plt.ylim(top=1, bottom=0)

        t = 'k-means, Titanic, vs labels'
        df=pd.DataFrame(sc['Titanic']['kmeans'], index=clusters)[['AMI','accuracy','IG']]
        axplot_df(ax5, df.reset_index(), t, xlab='k', linestyle=linestyle)
    
        t = 'GMM, Titanic, LL/100, BIC/1e5'
        df=pd.DataFrame(sc['Titanic']['GMM'], index=clusters)[['SSE/LL','Silhouette','BIC']]
        df[['SSE/LL']] = df[['SSE/LL']]/100
        df[['BIC']] = df[['BIC']]/1e5
        axplot_df(ax2, df.reset_index(), t, xlab='k', linestyle=linestyle)
    
        t = 'GMM, Titanic, vs labels'
        df=pd.DataFrame(sc['Titanic']['GMM'], index=clusters)[['AMI','accuracy','IG']]
        axplot_df(ax6, df.reset_index(), t, xlab='k', linestyle=linestyle)
        
        # Print Charts Wilt
        t = 'k-means, Wilt, SSE/10'
        df=pd.DataFrame(sc['Wilt']['kmeans'], index=clusters)[['SSE/LL','Silhouette']]
        df[['SSE/LL']] = df[['SSE/LL']]/10
        axplot_df(ax3, df.reset_index(), t, xlab='k', linestyle=linestyle)
#        ax3.set_ylim([0, 0.3])
    
        t = 'k-means, Wilt, vs labels'
        df=pd.DataFrame(sc['Wilt']['kmeans'], index=clusters)[['AMI','accuracy','IG']]
        axplot_df(ax7, df.reset_index(), t, xlab='k', linestyle=linestyle)
    
        t = 'GMM, Wilt, (LL/10, BIC/1e5)'
        df=pd.DataFrame(sc['Wilt']['GMM'], index=clusters)[['SSE/LL','Silhouette','BIC']]
        df[['SSE/LL']] = df[['SSE/LL']]/10
        df[['BIC']] = df[['BIC']]/1e5
        axplot_df(ax4, df.reset_index(), t, xlab='k', linestyle=linestyle)
        
        t = 'GMM, Wilt, vs labels'
        df=pd.DataFrame(sc['Wilt']['GMM'], index=clusters)[['AMI','accuracy','IG']]
        axplot_df(ax8, df.reset_index(), t, xlab='k', linestyle=linestyle)

def cluster_scores(X, y, pred):
    acc, H0, H1, cluster_summary = cluster_acc(y, pred)
    df = pd.DataFrame(cluster_summary)[['Size','Pos','Neg','H', 'Acc']]  #.sort_values(['Acc'], ascending=False)
    df2 = pd.DataFrame(pred,silhouette_samples(X, pred)).reset_index()[[0,'index']]
    df2.columns=['label','silh']
    df['Silh']=df2.groupby('label').mean()
    return df.sort_values(['Acc'], ascending=False)

#%% Jon Tay functions
from collections import Counter
from sklearn.metrics import accuracy_score
from math import log  
 
def cluster_acc(Y,clusterLabels):
    assert (Y.shape == clusterLabels.shape)
    pred_y = np.empty_like(Y)
    sz=len(Y); T=sum(Y); pT=T/sz; pF=(sz-T)/sz
    H0 = -(pT * log(pT,2))-(pF * log(pF,2))
    H1=0
    clusters=[]
    #cycle over cluster labels and find simple majority (from Y) for each  
    for label in set(clusterLabels):
        mask = clusterLabels == label
        sub = Y[mask]
        simple_majority = Counter(sub).most_common(1)[0][0]
        pred_y[mask] = simple_majority
        H = calc_H(sub)
        sz = sum(mask); pos = sum(sub)
#        print ('cluster size: %s  True: %s  H: %.3f' % (sum(mask), sum(sub), H))
        H1 += H * sum(mask) / len(Y)
        acc = accuracy_score(sub, pred_y[mask])
        clusters.append({'label':label, 'Size':sz, 'Pos':pos, 'Neg':sz-pos, 'H':H, 'Acc':acc})
    return accuracy_score(Y,pred_y), H0, H1, clusters

#%% DECISION TREE FUNCTIONS

# plot decision tree function
def plotDT(tr, fn):
    dot_data = export_graphviz(tr, filled=True, rounded=True, class_names=['Dec.','Surv.'], feature_names=fn, out_file=None)
    graph = graph_from_dot_data(dot_data)
    graph.write_png('tree.png')
    img=mpimg.imread('tree.png')
    plt.figure(57) #, figsize=(19, 16))
    plt.imshow(img)
    return True

# Parse the tree structure:  from http://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
def show_tree(tree):
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    
    print("The binary tree structure has %s nodes and has "
          "the following tree structure:"
          % n_nodes)
    for i in range(n_nodes):
        if is_leaves[i]:
            print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
        else:
            print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
                  "node %s."
                  % (node_depth[i] * "\t",
                     i,
                     children_left[i],
                     feature[i],
                     threshold[i],
                     children_right[i],
                     ))

#%% make CV sets
def get_cv_sets(X_train, y_train, k, rs):
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=rs).split(X_train,y_train)
    cv_sets=[]
    for (train,val) in kfold:
        cv_sets.append([train,val])
    return cv_sets

#%% Learning Curve functions
            
def clf_learning(X, y, clf, k=10, rs=0):
    # generates learning curve for sizes from 2% to 80% of the data 
    tr_scores={}
    va_scores={}
    te_scores=[]
    size=[]
    for i in np.linspace(0.1, 1.0, 20):
        # new Training set from 5% to 80%
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i, random_state=rs, stratify=y)
        sz = len(y_train)
        size.append(sz)
        # new CV sets
        kfold = StratifiedKFold(n_splits=k, random_state=rs).split(X_train,y_train)
        cv_sets=[]
        for (train,val) in kfold:
            cv_sets.append([train,val])
    
        # train classifier using CV and record accuracy scores
        tr_scores[sz]=[]
        va_scores[sz]=[]
        for (train,test) in cv_sets:
            clf.fit(X_train[train], y_train[train])
            tr_scores[sz].append(clf.score(X_train[train], y_train[train]))
            va_scores[sz].append(clf.score(X_train[test], y_train[test]))
    
        # fit this classifier on full training set for this sample size and score vs test set
        clf.fit(X_train, y_train)
        te_scores.append(round(clf.score(X_test, y_test),4))
    return size, tr_scores, va_scores, te_scores

def explore_CV(clf, Xf, yf, kv=[10], rs=0):
    # generates learning curve for sizes from 2% to 80% of the data 
    tr_scores={}
    va_scores={}
    for k in kv:
        # create CV sets
        kfold = StratifiedKFold(n_splits=k, random_state=rs).split(Xf, yf)
        cv_sets=[]
        for (train,val) in kfold:
            cv_sets.append([train,val])
    
        # train classifier using CV and record accuracy scores
        tr_scores[k] = []
        va_scores[k] = []
            
        for (train,test) in cv_sets:
            clf.fit(Xf[train], yf[train])
            tr_scores[k].append(clf.score(Xf[train], yf[train]))
            va_scores[k].append(clf.score(Xf[test],  yf[test]))
    print (clf)
    for k in kv:
        m = mean(va_scores[k])
        s = np.std(va_scores[k])
        print ('%s-fold:  va_sc: %.4f  sd: %.3f' % (k,m,s))
    return tr_scores, va_scores

# Plots learning curve (decision tree)
def plot_LC(clf, size, tr_scores, va_scores, t, h=4, w=5):
    # inputs - arrays of CV scores
    # first, prep data:
    tr_mean = np.mean(tr_scores, axis=1)
    tr_std = np.std(tr_scores, axis=1)
    va_mean = np.mean(va_scores, axis=1)
    va_std = np.std(va_scores, axis=1)
    # begin plotting
    fig = plt.figure(t, figsize=(w, h))
    plt.clf()
    ax = fig.add_subplot(111)
    ax.plot(size, tr_mean,'-o', markersize=3)
    plt.fill_between(size, tr_mean - tr_std, tr_mean + tr_std, color='blue', alpha=0.2)
    ax.plot(size, va_mean,'-o', markersize=3)
    plt.fill_between(size, va_mean - va_std, va_mean + va_std, color='orange', alpha=0.2)
    ax.set_ylim([0.7, 1])
    ax.set_xlim(left=0)
    plt.title(t, size='small')
    plt.xlabel("Training set size")
    plt.legend(['Training','Validation'], loc='best', frameon=False, title='')
    return fig

def plot_k(xy_pairs, t, h=4, w=5, xlab='x', ylab='y'):
    #xy_pairs = iterable of iterables ([x value, y value, series label] ,...)
    # begin plotting
    fig = plt.figure(t, figsize=(w, h))
    plt.clf()
    ax = fig.add_subplot(111)
    for x,y in xy_pairs:
        ax.plot(x, y,'-o', markersize=3)
#        plt.fill_between(size, tr_mean - tr_std, tr_mean + tr_std, color='blue', alpha=0.2)
#    ax.set_ylim([0.7, 1])
#    ax.set_xlim(left=0)
    plt.title(t, size='medium')
    plt.xlabel(xlab)
    leg = [a for _,_,a in xy_pairs]
    plt.legend(leg, loc=4, frameon=False, title='')
    return fig

#%% 3D PLOT RESULTS
# general 3D plotting function
def plot_GS_3D(clf, type3D, param1, param2, h=8, w=6):
    # suitable for plotting grid search results across 2 parameters 
    # type3D from scatter contour surface wireframe
    # output figure, grid search results as table, and a2m (mean accuracy for each param2 value)   
    a1 = [a for a in clf.cv_results_[param1]]
    a2 = [a for a in clf.cv_results_[param2]]
    a3 = [a for a in clf.cv_results_['mean_test_score']]
    a1v = sorted(list(set(a1)))
    a2v = sorted(list(set(a2)))
    # make dictionary of results
    a3d = {(a1,a2):a3 for a1, a2, a3 in zip(a1,a2,a3)}
    A3=[]
    a2m=[]  #mean score for each value of a2 across all values of a1, use of a1 is random_state
    for b in a2v:
        z=[]
        for a in a1v:
            z.append(a3d[(a,b)])
        A3.append(z)
        a2m.append(mean(z))
    A3 = np.array(A3)
    A1, A2 = np.meshgrid(a1v, a2v)
    fig = plt.figure('Grid Search CV results (' + type3D + ')', figsize=(w, h))
    ax = plt.axes(projection='3d')
    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    ax.set_zlabel('cv test');
    plt.title('Grid Search CV results')
    if type3D=='scatter':
        ax.scatter3D(A1, A2, A3)
    if type3D=='contour':
        ax.contour3D(A1, A2, A3, 50, cmap='binary')
    if type3D=='surface':
        ax.plot_surface(A1, A2, A3, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    if type3D=='wireframe':
        ax.plot_wireframe(A1, A2, A3, color='black')
    return fig, A3, a2m

#%% EXPLORE HYPER PARAMETERS

def explore_max_leafs(leafs, cv_sets, spl='random', rs=36):
    # Explore complexity scores by varying max_leaf_nodes
    # Plot results
    # Return (training score, validation score, best value for max_leaf_nodes, best DT classifier) 
    tr_scores=[]
    va_scores=[]
    for n in leafs:
        dt = DecisionTreeClassifier(splitter=spl, random_state=rs, max_leaf_nodes=n)
        cv_scores=[]
        for (train,test) in cv_sets:
            dt.fit(X_train[train], y_train[train])
            sc_val = dt.score(X_train[test], y_train[test])
            sc_trn = dt.score(X_train[train], y_train[train])
            cv_scores.append((sc_trn, sc_val))
        tr_scores.append(round(mean([s[0] for s in cv_scores]),4)) 
        va_scores.append(round(mean([s[1] for s in cv_scores]),4))
        if max(va_scores)==va_scores[-1]:
            # best classifier so far
            best = (n,dt)
    return tr_scores, va_scores, best

def plot_EML(leafs, tr_scores, va_scores, rs, t):
    # Plot training vs validation accuracy
    fig = plt.figure(t, figsize=(5, 4))
    ax = fig.add_subplot(111)
    if tr_scores!=0:
        ax.plot(leafs, tr_scores,'-o', markersize=2, label='training rs=%s' %rs)
    ax.plot(leafs, va_scores,'-o', markersize=2, label='validation rs=%s' %rs)
    ax.set_ylim([0.7, 1])
    plt.title(t)
    plt.legend(loc=2, frameon=False, title='')
    plt.xlabel("max_leaf_nodes")

# test prediction function (replaced by sklearn.score)
def pred_perf(X, y, clf, digits=4, pr_txt=''):
    y_pred = clf.predict(X)
    perf = pd.DataFrame({'Act':y.flatten(), 'Pred':y_pred})
    perf['Corr']= np.abs(perf.Act + perf.Pred - 1)
    print (pr_txt + ' accuracy', sum(perf.Corr), '/', len(y_pred), '=', round(sum(perf.Corr) / len(y_pred), digits))
    return perf

def clf_test_score(clf, Xf_tr, yf_tr, Xf_te, yf_te):
    clf.fit(Xf_tr, yf_tr)
    pred_perf(Xf_tr, yf_tr, clf, digits=5, pr_txt='training')
    perf = pred_perf(Xf_te, yf_te, clf, digits=5, pr_txt='test')
    sum(perf['Corr'])
    print('(sklearn scores) %.5f / %.5f' %(clf.score(Xf_tr, yf_tr), clf.score(Xf_te, yf_te)))


# GRID SEARCH

#Xf=X_train; yf=y_train
# general GS functions
def GS_summary(clf, Xf, yf):
    # only works after fitting
    print ('best validation score: %.5f' % clf.best_score_)
    print ('best parameters: \n', clf.best_params_)
#    print ('full set (length %s) score: %.5f' % (len(Xf), clf.score(Xf, yf)))

def run_GS(est, GS_params, X_train, y_train, cv_sets, scoring='accuracy'):
    # uses global X_train, y_train, X_test, y_test
    start_time = time.time()
    print ('grid search started ', time.strftime("%H:%M:%S", time.localtime(time.time())))
    clf = GridSearchCV(estimator=est, param_grid=GS_params, cv=cv_sets, scoring=scoring)
    clf.fit(X_train, y_train)
    end_time = time.time()
    print("--- GridSearchCV %.1f seconds ---" % (end_time - start_time))
    GS_summary(clf, X_train, y_train)
    return clf


# Function for summarising Confusion matrix 
from sklearn.metrics import confusion_matrix
def summarise_CM(cm, tr_or_te):
    # cm = a confusion matrix
    # tr_or_te = a string indicating either 'training' or 'test'
    (tn,fp),(fn,tp) = cm
    acc = (tn + tp)/(tn + tp + fn + fp)
    print ('\nconfusion matrix (' + tr_or_te + '): \n', cm)
    print ('sensitivity (TPR): %.3f, specificity (TNR): %.3f, average: %.3f, accuracy: %.3f' % (tp/(tp+fn), tn/(tn+fp), (tp/(tp+fn)/2)+(tn/(tn+fp)/2), acc))
