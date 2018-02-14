# CB_StochasticExperts
__Code for Contextual Bandits with Stochastic Experts__

Dependencies: xgboost, scikit-learn, pandas, numpy, scipy, matplotlib, multiprocessing, itertools

Base Class: stochExp

Initialization Parameters:


```python
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


```

