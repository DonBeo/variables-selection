# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 21:00:32 2013

@author: Luca
"""


def stability_selection(X,y,alphas=False,sim=10,min_weight=0.2,plot_me=False):
    # this is a very beta version
    import numpy as np
    import sklearn as sk
    import sklearn.linear_model    
    (n,p)=X.shape #n is the number of samples and p the number of variables
    X=sk.preprocessing.scale(X) 
    y=sk.preprocessing.scale(y)
    #set the lasso path   
    if alphas.any():    
        alphas, coefs_lasso, _ = sk.linear_model.lasso_path(X, y,return_models=False,fit_intercept=False)
    
    prob=np.zeros(shape=(p,alphas.shape[0]))
    # run random lasso 
    for i in range(sim):
        w=np.random.uniform(low=min_weight,high=1,size=p)
        samples=np.random.randint(low=0,high=n-1,size=int(n/2))    
        
        #weight the penalty    
        X_weighted=np.apply_along_axis(lambda x: x*w,axis=1,arr=X[samples,])
        coef=sk.linear_model.lasso_path(X_weighted, y[samples],alphas=alphas,return_models=False,fit_intercept=False)[1]
        prob[coef!=0]=prob[coef!=0]+1
    prob=prob/sim    
    
    if plot_me:    
        import matplotlib.pyplot as pl

        plot=pl.plot(np.log(alphas),prob.T)
        pl.show(plot)
    
    max_prob=prob.max(axis=1)
    return max_prob,prob,alphas


import numpy as np
# create a sample dataset
p=200
n=200
sigma=np.diag([1.]*p)
sigma[0,2]=0.6
sigma[2,0]=0.6
sigma[1,2]=0.6
sigma[2,1]=0.6
X=np.random.multivariate_normal(mean=np.zeros(shape=(p,)),cov=sigma,size=n)
beta=np.zeros(shape=(p,1))
beta[0]=beta[1]=1
y=np.dot(X,beta).squeeze()
# choose the lasso path    
alphas=np.arange(0.0001,1,step=0.0051)
#run stability selection on our samples dataset
l=stability_selection(X,y,alphas=alphas,sim=100,min_weight=0.2,plot_me=True)
