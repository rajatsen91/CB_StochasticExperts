#Author: Rajat Sen
#Dependency: xgboost, logistic rgression, numpy, scikit-learn, matplotlib, pandas


import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score,accuracy_score
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
import itertools
import matplotlib
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from multiprocessing import Pool


def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return izip(a, a)



def pairmodel_divergence( Xtrain,M1,M2, option = 1):
    '''Function to calculate divergence between two experts 
    1. When Xtrain is a data-set of features/contexts and M1,M2 are models (eg. xgboost) then we get predict_proba predictions on Xtrain with M1,
    M2 and then calculate divergence using the probability values.
    2. When Xtrain is none and the probability values themselves are supplied as M1, M2 then the divergence is calculated directly
    '''
    if Xtrain is None:
        P1 = M1
        P2 = M2
    else:
        P1 = M1.predict_proba(Xtrain)
        P2 = M2.predict_proba(Xtrain)
    
    if option == 1:
        X = P1/P2
        fx = X*np.exp(X-1) - 1
        Efx = P2*fx
        Efx_sum = np.sum(Efx,axis=1)
        l = np.where(Efx_sum > 1e100)
        inan = np.where(np.isnan(Efx_sum) == True)
        Efx_sum[inan] = 1e100
        Efx_sum[l] = 1e100
        return MedianofMeans(Efx_sum)
    if option == 2:
        X = P1/P2
        fx = X**2 - 1
        Efx = P2*fx
        Efx_sum = np.sum(Efx,axis=1)
        inan = np.where(np.isnan(Efx_sum) == True)
        Efx_sum[inan] = 1e10 
        l = np.where(Efx_sum > 1e10)
        Efx_sum[l] = 1e10
        return MedianofMeans(Efx_sum)
    
def divergence_helper(sims):
    '''helper function to calculate divergences in parallel between pairs of models '''
    div = pairmodel_divergence(sims[2],sims[3],sims[4],sims[5])
    return (sims[0],sims[1],div,sims[5])

def MedianofMeans( A,ngroups = 55):
    '''Calculates median of means given an Array A, ngroups is the number of groups '''
    if ngroups > len(A):
        ngroups = len(A)
    Aperm = np.random.permutation(A)
    try:
        lists = np.split(Aperm,ngroups)
    except:
        x = len(A)/ngroups
        ng = len(A)/(x+1)
        Atrunc = Aperm[0:ng*x]
        lists = np.split(Atrunc,ng) + [Aperm[ng*x::]]
    plist = pd.Series(lists)
    mlist = plist.apply(np.mean)
    return np.median(mlist)


class stochExp:
    '''Base class for running contextual bandits with stochastic experts 
    Xdata: Features/contexts ar numpy array
    Rdata: labels/rewards as numpy array; size : number of samples * number of classes. Each row has the rewards for each class
    K: Number of arms/labels
    T0: Run random arm pull till time t = T0
    C1: constant used in confidence bound for MoM algorithm
    C2: constant used in confidence bound for Clipped algorithm
    isMoM: True then run MoM otherwise run clipped version
    calibrate: Ture implies use calibrated classifiers otherwise base scikit learn classifiers
    bsize_mult: Multiplier of sqrt(t) in the batch-size
    initial_model: eg. [3,3,1] initally spawn 3 xgboost experts, 3 logistic regression experts and 1 dummy random arm choosing experts
    model_ratio: eg. [3,1] in all succeeding batches choose 3 xgboost experts and 1 logistic regression experts
    max_depths,n_estimators,colsample_bytrees = [0.6,0.8]: parameters searched over for xgboost
    nthread: number of threads to run xgboost in parallel
    penalty,C : parameters searched for logistic model 
    log_file: .npy to save rewards in as the algorithm progresses
    '''
    def __init__(self,Xdata,Rdata,K,T0 = 50,C1 = 2.0 ,C2 = 1.0,isMoM = True,calibrate = False,bsize_mult = 7,initial_model = [3,3,1], model_ratio = [3,1],max_depths = [3,6,10], n_estimators = [50,100,200], colsample_bytrees = [0.6,0.8],  nthread = 16, penalty = ['l1','l2'], C = [1e-3,1e-2,1e-1,1.0,1e1,1e2],log_file = 'rewards_tn.npy',cl_limit = 2.0):
        self.T0 = max([3*K,T0])
        self.Xdata = Xdata
        self.Rdata = Rdata
        self.K = K
        self.C1 = C1
        self.C2 = C2
        self.isMoM = isMoM
        self.calibrate = False
        self.bsize_mult = bsize_mult
        self.initial_model = initial_model
        self.model_ratio = model_ratio
        self.max_depths = max_depths
        self.n_estimators = n_estimators
        self.penalty = penalty
        self.nthread = nthread
        self.C = C
        self.log_file = log_file
        self.colsample_bytrees = colsample_bytrees
        self.cl_limit = cl_limit
        self.Mlist = None
        


    def modelfit(self,Xtrain,Ytrain,weights = None, cvbreak = None, max_depths = [3,6,10], n_estimators = [50,100,200], colsample_bytrees = [0.6,0.8], nthread = 4,calibrate = True, cv = False):
        n,m = Xtrain.shape
        if weights is None:
            weights = np.ones(len(Ytrain))
        if cvbreak is None:
            cvbreak = min([0.20,1000.0/float(n)])
        sss = StratifiedShuffleSplit(n_splits=2, test_size=cvbreak)
        try:
            for train,test in sss.split(Xtrain,Ytrain):
                Xt1,Xt2 = Xtrain[train,:],Xtrain[test,:]
                Yt1,Yt2 = Ytrain[train],Ytrain[test]
                wt1,wt2 = weights[train],weights[test]
        except:
            nt = int(n*(1.0 - cvbreak))
            indices = np.random.choice(n,nt,False)
            nindices = list(set(range(n)).difference(set(indices)))
            Xt1,Xt2 = Xtrain[indices,:],Xtrain[nindices,:]
            Yt1,Yt2 = Ytrain[indices],Ytrain[nindices]
            wt1,wt2 = weights[indices],weights[nindices]
        grid = {"max_depth":max_depths[0], "colsample_bytree":colsample_bytrees[0], "n_estimators":n_estimators[0], "objective":"multi:softprob"}
        gbm = xgb.XGBClassifier()
        gbm.set_params(**grid)
        if calibrate == False:
            gbf = gbm.fit(Xtrain,Ytrain,weights)
        else:
            gbf = gbm.fit(Xt1,Yt1,wt1)
        if calibrate == True:
            ccv = CalibratedClassifierCV(gbf)
            if weights is None:
                try:
                    ccvt = ccv.fit(Xt2,Yt2)
                except:
                    ccvt = gbf
            else:
                try:
                    ccvt = ccv.fit(Xt2,Yt2,wt2)
                except:
                    ccvt = gbf
        else:
            ccvt = gbf
        
        return ccvt

    def modelfit_logistic(self, Xtrain,Ytrain,weights = None, cvbreak = None, penalty = 'l2', C = 1.0, nthread = 4,calibrate = True):
        n,m = Xtrain.shape
        if weights is None:
            weights = np.ones(len(Ytrain))
        if cvbreak is None:
            cvbreak = min([0.20,1000.0/float(n)])
        sss = StratifiedShuffleSplit(n_splits=2, test_size=cvbreak)
        try:
            for train,test in sss.split(Xtrain,Ytrain):
                Xt1,Xt2 = Xtrain[train,:],Xtrain[test,:]
                Yt1,Yt2 = Ytrain[train],Ytrain[test]
                wt1,wt2 = weights[train],weights[test]
        except:
            nt = int(n*(1.0 - cvbreak))
            indices = np.random.choice(n,nt,False)
            nindices = list(set(range(n)).difference(set(indices)))
            Xt1,Xt2 = Xtrain[indices,:],Xtrain[nindices,:]
            Yt1,Yt2 = Ytrain[indices],Ytrain[nindices]
            wt1,wt2 = weights[indices],weights[nindices]
        lr = LogisticRegression(penalty= penalty, C= C, n_jobs=nthread)
        if calibrate == False:
            lrf= lr.fit(Xtrain,Ytrain,weights)
        else:
            lrf = lr.fit(Xt1,Yt1,wt1)
        if calibrate == True:
            ccv = CalibratedClassifierCV(lrf)
            if weights is None:
                try:
                    ccvt = ccv.fit(Xt2,Yt2)
                except:
                    ccvt = lrf
            else:
                try:
                    ccvt = ccv.fit(Xt2,Yt2,wt2)
                except:
                    ccvt = lrf
        else:
            ccvt = lrf
        return ccvt
    
    def bootstrap(self,Xtrain,Ytrain,weights = None, cvbreak = None, max_depths = [3,6,10], n_estimators = [50,100,200], colsample_bytrees = [0.6,0.8], nthread = 4,calibrate = True, cv = False, Nboot = 100):
        L = [0]*Nboot
        for i in range(Nboot):
            Ind = np.random.choice(len(Ytrain), len(Ytrain), True )
            Xt = Xtrain[Ind,:]
            Yt = Ytrain[Ind]
            if weights is None:
                wt = None
            else:
                wt = weights[Ind]
            md = np.random.choice(max_depths)
            cb = np.random.choice(colsample_bytrees)
            ne = np.random.choice(n_estimators)
            if calibrate:
                cbl = np.random.choice([False,True])
            else:
                cbl = False
            
            L[i] = self.modelfit(Xt,Yt,wt, cvbreak, [md], [ne] , [cb] , nthread,cbl, cv)
            
        return L

    def bootstrap_logistic(self,Xtrain,Ytrain,weights = None, cvbreak = None, penalty = ['l1','l2'], C = [1e-3,1e-2,1e-1,1.0,1e1,1e2], nthread = 4,calibrate = True, cv = False, Nboot = 100):
        L = [0]*Nboot
        for i in range(Nboot):
            Ind = np.random.choice(len(Ytrain), len(Ytrain), True )
            Xt = Xtrain[Ind,:]
            Yt = Ytrain[Ind]
            if weights is None:
                wt = None
            else:
                wt = weights[Ind]
            pl = np.random.choice(penalty)
            cs = np.random.choice(C)
            if calibrate:
                cbl = np.random.choice([False,True])
            else:
                cbl = False
            
            cbl = np.random.choice([False,True])
            
            L[i] = self.modelfit_logistic(Xt,Yt,wt, cvbreak, pl, cs , nthread,cbl)
            
        return L


    
    
    def div_matrix_parallel(self, Xtrain, Mlist, option = 1,num_jobs = 16):
        N = len(Mlist)
        iters = range(N)
        Plist = [0]*N
        for i in range(N):
            Plist[i] = Mlist[i].predict_proba(Xtrain)
        pool = Pool(processes=num_jobs)
        
        M = np.ones([N,N])
        arguments = []
        for i,j in itertools.product(iters, repeat=2):
            if i == j:
                continue
            else:
                arguments = arguments + [(i,j,None,Plist[i],Plist[j], option)]
        result = pool.map(divergence_helper,arguments)
        
        for R in result:
            if R[3] == 1:
                M[R[0],R[1]] = 1 + np.log(1 + R[2])
            else:
                M[R[0],R[1]] = 1 + R[2]
        pool.close()
        pool.join()
        
        return M

    def MedianofMeans(self, A,ngroups = 55):
        if ngroups > len(A):
            ngroups = len(A)
        Aperm = np.random.permutation(A)
        try:
            lists = np.split(Aperm,ngroups)
        except:
            x = len(A)/ngroups
            ng = len(A)/(x+1)
            Atrunc = Aperm[0:ng*x]
            lists = np.split(Atrunc,ng) + [Aperm[ng*x::]]
        plist = pd.Series(lists)
        mlist = plist.apply(np.mean)
        return np.median(mlist)

    def weightedmean(self,A):
        x = np.sum(A[:,0]*A[:,1])
        w = np.sum(A[:,1])
        return x/w

    def weightedmean_all(self,A,M):
        A2 = A[:,0:M]*A[:,M::]
        x = np.sum(A2, axis = 0)
        w = np.sum(A[:,M::],axis = 0)
        return x/w


    def clippedmean_all(self,A,M,t):
        A1 = A[:,0:M]
        A2 = 1.0/A[:,M::]
        A3 = A[:,M::]*(2*np.log(t))
        I = np.where(A1 > A3)
        A1[I] = 0
        Am = A1*A2
        x = np.sum(Am, axis = 0)
        w = np.sum(A2,axis = 0)
        return x/w

    def calculta_UCB_clipped_together(self,prun,Rrun,t,mrun, Plist, DivM, C):
        N,M = Plist.shape
        Dlist = Plist/prun
        Rlist = Rrun*Dlist
        
        sigma2 = np.array(pd.Series(mrun.reshape(len(mrun))).map(lambda x: DivM[0,x])).reshape(len(mrun),1)
        for i in range(1,M):
            s = np.array(pd.Series(mrun.reshape(len(mrun))).map(lambda x: DivM[i,x])).reshape(len(mrun),1)
            sigma2 = np.hstack([sigma2,s])

        
        weightedA = np.hstack([Rlist , sigma2])
        
        u  = self.clippedmean_all(weightedA,M,t)
        inv = 1.0/sigma2
        wtot = np.sum(inv,axis = 0)
        ucb = u +  ((1.0/(wtot)**2)*(C*np.log(t)*float(t)))**(1.0/2.000001)
        if t%500 == 0:
            print 'Current Expert means: ', 
            print u
            print 'Current Expert ucbs: ',
            print ucb
        
        
        return u,ucb

    def calculta_UCB_MOM_together(self,prun,Rrun,t,mrun, Plist, DivM, C):
        N,M = Plist.shape
        Dlist = Plist/prun
        Rlist = Rrun*Dlist
        
        sigma2 = np.array(pd.Series(mrun.reshape(len(mrun))).map(lambda x: DivM[0,x])).reshape(len(mrun),1)
        for i in range(1,M):
            s = np.array(pd.Series(mrun.reshape(len(mrun))).map(lambda x: DivM[i,x])).reshape(len(mrun),1)
            sigma2 = np.hstack([sigma2,s])
        invsigma = np.sqrt(1.0/sigma2)
        
        weightedA = np.hstack([Rlist,invsigma])
        
        ngroups = 4*int(np.log(t))
        weightedAperm = np.random.permutation(weightedA)
        
        try:
            lists = np.split(weightedAperm,ngroups)
        except:
            x = N/ngroups
            ng = N/(x+1)
            weightedAtrunc = weightedAperm[0:ng*x,:]
            lists = np.split(weightedAtrunc,ng) + [weightedAperm[ng*x::,:]]
        
        plist = pd.Series(lists)
        mlist = plist.apply(lambda A: self.weightedmean_all(A,M))
        

        
        
        
        u = np.median(np.array(list(mlist)),axis = 0)
        
        wtot = np.sum(invsigma,axis = 0)
        wfactor = wtot/float(N)
        
        
        ucb = u +  (1.0/wfactor)*np.sqrt(C*np.log(t)/float(t))
        
        if t%500 == 0:
            print 'Current Expert means: ', 
            print u
            print 'Current Expert ucbs: ',
            print ucb
        
        
        return u,ucb

    def run_algorithm(self):
        Xdata = self.Xdata
        Rdata = self.Rdata
        N,M = Xdata.shape
        calibrate = True
        if self.isMoM:
            option = 2
        else:
            option = 1
        T0 = self.T0
        K = self.K
        Tcurr = 0
        t = 1
        Xrun = Xdata[t-1,:]
        Yrun = np.array([np.random.choice(K)+1])
        Rrun = np.array(Rdata[t-1,Yrun[t-1]-1])
        prun = np.array([1.0/K])

        for t in range(2,T0+1):
            Xrun = np.vstack([Xrun, Xdata[t-1,:]])
            if t <= 3*K:
                Yrun = np.vstack([Yrun, np.array([t%K + 1])])
            else:
                Yrun = np.vstack([Yrun, np.array([np.random.choice(K)+1])])
            Rrun = np.vstack([ Rrun, Rdata[t-1,Yrun[t-1] - 1]])
            prun = np.vstack([prun, np.array([1.0/K])])


        mrun = np.array([0]*len(prun)).reshape(len(prun),1)

        Tcurr = T0 + 1    
        Lxgb = self.bootstrap(Xrun,Yrun.reshape(len(Yrun)),weights = (Rrun/prun).reshape(len(Yrun)), cvbreak = None, max_depths = self.max_depths, n_estimators = self.n_estimators, colsample_bytrees = self.colsample_bytrees, nthread = self.nthread,calibrate = calibrate, cv = False, Nboot = self.initial_model[0])
        Llogistic = self.bootstrap_logistic(Xrun,Yrun.reshape(len(Yrun)),weights = (Rrun/prun).reshape(len(Yrun)), cvbreak = None, penalty= self.penalty , C=self.C , nthread = self.nthread,calibrate = calibrate, cv = False, Nboot = self.initial_model[1])
        L1 = [DummyClassifier('uniform').fit(Xrun,Yrun.reshape(len(Yrun)))]

        Mlist = L1 + Lxgb + Llogistic

        Mlist2 = []

        for i in range(len(Mlist)):
            PP = Mlist[i].predict_proba(Xrun[0:10,:])
            n,m = PP.shape
            
            if m == K:
                Mlist2 = Mlist2 + [Mlist[i]]
            
        Mlist = Mlist2

        Plist = Mlist[0].predict_proba(Xrun)[:,Yrun - 1][:,0,:]


        for i in range(1,len(Mlist)):
            try:
                X =  Mlist[i].predict_proba(Xrun)[(np.arange(t),(Yrun - 1).reshape(1,t))].reshape(t,1)
            except:
                X =  Mlist[i].predict_proba(Xrun)[0][(np.arange(t),(Yrun - 1).reshape(1,t))].reshape(t,1)
            Plist = np.hstack([Plist,X])

        DivM = self.div_matrix_parallel(Xrun, Mlist, option = option,num_jobs = self.nthread)

        t = Tcurr
        oldlen = 0
        batch_size = self.bsize_mult*int(np.sqrt(t))
        while t < N-1:
            for r in range(batch_size):
                if self.isMoM:
                    u,UCBmeans = self.calculta_UCB_MOM_together(prun,Rrun,t,mrun, Plist, DivM, self.C1)
                else:
                    u,UCBmeans = self.calculta_UCB_clipped_together(prun,Rrun,t,mrun, Plist, DivM, self.C2)
                if t%10 == 0:
                    print 'Time: ' + str(t)
                if t%100 == 0:
                    np.save(self.log_file,Rrun)
            
                Xrun = np.vstack([Xrun, Xdata[t-1,:]])
                m = np.argmax(UCBmeans) 
                if u[m] > self.cl_limit:
                    calibrate = False
                pvalues = Mlist[m].predict_proba(Xrun[t-1,:].reshape(1,M))[0]
                r = np.random.choice(a = K,p = pvalues) + 1
                Yrun = np.vstack([Yrun, r])
                Rrun = np.vstack([ Rrun, Rdata[t-1,Yrun[t-1] - 1]])
                mrun = np.vstack([mrun, m])
                prun = np.vstack([prun, pvalues[r - 1]])
                X = np.ones([1,len(Mlist)])
                for i in range(len(Mlist)):
                    X[0,i] = Mlist[i].predict_proba(Xrun[t-1,:].reshape(1,M))[0][r-1]
                Plist = np.vstack([Plist,X])
                t = t + 1
                if t == N:
                    np.save(self.log_file,Rrun)
                    break
            batch_size = self.bsize_mult*int(np.sqrt(t))
            Lxgb = self.bootstrap(Xrun,Yrun.reshape(len(Yrun)),weights = (Rrun/prun).reshape(len(Yrun)), cvbreak = None, max_depths = self.max_depths, n_estimators = self.n_estimators, colsample_bytrees = self.colsample_bytrees, nthread = self.nthread,calibrate = calibrate, cv = False, Nboot = self.model_ratio[0])
            Llogistic = self.bootstrap_logistic(Xrun,Yrun.reshape(len(Yrun)),weights = (Rrun/prun).reshape(len(Yrun)), cvbreak = None, penalty= self.penalty , C=self.C,  nthread = self.nthread,calibrate = calibrate, cv = False, Nboot = self.model_ratio[1])
            oldlen = len(Mlist)
            Mlist = Mlist + Lxgb + Llogistic
            Mlist2 = []
            
            for i in range(len(Mlist)):
                PP = Mlist[i].predict_proba(Xrun[0:10,:])
                n,m = PP.shape
            
                if m == K:
                    Mlist2 = Mlist2 + [Mlist[i]]
            
            Mlist = Mlist2
            print "# models: ", len(Mlist)
            
            for i in range(oldlen,len(Mlist)):
                try:
                    X =  Mlist[i].predict_proba(Xrun)[(np.arange(t-1),(Yrun - 1).reshape(1,t-1))].reshape(t-1,1)
                except:
                    X =  Mlist[i].predict_proba(Xrun)[0][(np.arange(t-1),(Yrun - 1).reshape(1,t-1))].reshape(t-1,1)
                Plist = np.hstack([Plist,X])

            DivM = self.div_matrix_parallel(Xrun, Mlist, option = option,num_jobs = self.nthread)
            self.Mlist = Mlist
        print 'Achieved Mean Progressive Validation Accuracy: ' + str(np.mean(Rrun))

    
def error_plot(filename,savefile):
    R = np.load(filename)
    average_error = []
    for i in range(2,len(R)):
        average_error = average_error + [1.0 - np.mean(R[0:i])]
    
    plt.plot(average_error)
    plt.xlabel('Time')
    plt.ylabel('Progressive Validation Error')
    plt.savefig(savefile)
    plt.close()
    
    
    
    





