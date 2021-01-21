# On 20151231 by MLdP <lopezdeprado@lbl.gov> 
import scipy.cluster.hierarchy as sch,random
import pandas as pd
import numpy as np

from HRP import correlDist, getIVP, getQuasiDiag, getRecBipart 
from DPCCA import compute_dpcca_others

# First, we generate 10 series of random Gaussian returns (520 observations, equivalent to two years of daily history), with 0 mean and an arbitrary standard deviation of 10%. Real prices exhibit frequent jumps
    # (Merton [1976]) and returns are not cross-sectionally independent, so we must add random shocks and a random correlation structure to our generated data. Second, we compute HRP, CLA, and IVP portfolios by
    # looking back at 260 observations (a year of daily history). These portfolios are re-estimated and rebalanced every 22 observations (equivalent to a monthly frequency). Third, we compute the out-of-sample returns associated
    # with those three portfolios. This procedure is repeated 10,000 times.
#------------------------------------------------------------------------------ 

def generateData(nObs, backTwindow, size0, size1, mu0, sigma0, sigma1F): 
    # Time series of correlated variables
    #1) generate random uncorrelated data 
    x = np.random.normal(mu0, sigma0, size = (nObs, size0)) # each row is a variable 
    #2) create correlation between the variables 
    cols = [random.randint(0, size0 - 1) for i in range(size1)] 
    y = x[:, cols] + np.random.normal(0, sigma0 * sigma1F, size = (nObs, len(cols))) 
    x = np.append(x, y, axis = 1) 
    #3) add common random shock 
    point = np.random.randint(backTwindow, nObs - 1, size = 2) 
    x[np.ix_(point, [cols[0], size0])] = np.array([[-.5, -.5], [2, 2]]) 
    #4) add specific random shock 
    point = np.random.randint(backTwindow, nObs - 1, size = 2) 
    x[point, cols[-1]] = np.array([-.5, 2]) 
    return x, cols 
#------------------------------------------------------------------------------ 

def getHRP(cov, corr): 
    # Construct a hierarchical portfolio 
    corr = pd.DataFrame(corr)
    cov = pd.DataFrame(cov) 
    dist = correlDist(corr) 
    link = sch.linkage(dist, 'ward') 
    sortIx = getQuasiDiag(link) 
    sortIx = corr.index[sortIx].tolist() # recover labels 
    hrp = getRecBipart(cov,sortIx) 
    return hrp.sort_index() 
#------------------------------------------------------------------------------ 

# def getCLA(cov,**kargs): # Compute CLA's minimum variance portfolio 
    # mean = np.arange(cov.shape[0]).reshape(-1, 1) # Not used by C portf 
    # lB = np.zeros(mean.shape) 
    # uB = np.ones(mean.shape) 
    # cla = CLA.CLA(mean, cov, lB, uB) 
    # cla.solve() 
    # return cla.w[-1].flatten() 
# #------------------------------------------------------------------------------ 

def hrpMC(numIters = 1000, nObs = 520, size0 = 5, size1 = 5, mu0 = 0, sigma0 = 1e-2, sigma1F = .25, backTwindow = 260, rebalancing = 5, rolling_window = 250):    #sigma 1e-2 = 0.01
    # Monte Carlo experiment on HRP 
    methods = [getIVP, getHRP] #missing getCLA
    stats = {i.__name__:pd.Series() for i in methods}   
    #print("\n Le funzioni dentro methods:", stats)
    numIter = 0
    pointers = range(backTwindow, nObs, rebalancing) 
    while numIter < numIters: 
        print(numIter)
        #1) Prepare data for one experiment 
        x, cols = generateData(nObs, backTwindow, size0, size1, mu0, sigma0, sigma1F) 
        r = {i.__name__:pd.Series() for i in methods} 
        #2) Compute portfolios in-sample 
        for pointer in pointers:        
            x_ = x[pointer - backTwindow : pointer] 
            cov_= np.cov(x_, rowvar = 0) 

            #CORR estimation METHODS--------------------------------------  your choice
            #corr_ = np.corrcoef(x_, rowvar = 0)     #coeff di pearson
            
            corr_ = compute_dpcca_others(x_, rolling_window)    #DCCA method
            #CORR estimation METHODS--------------------------------------  your choice

            #3) Compute performance out-of-sample 
            x_ = x[pointer:pointer + rebalancing] 
            for func in methods:           
                w_ = func(cov = cov_, corr = corr_) # callback 
                r_ = pd.Series(np.dot(x_, w_)) 
                r[func.__name__] = r[func.__name__].append(r_) 
        #4) Evaluate and store results 
        for func in methods: 
            r_ = r[func.__name__].reset_index(drop = True) 
            p_ = (1 + r_).cumprod() 
            stats[func.__name__].loc[numIter] = p_.iloc[-1] - 1 # terminal return      
        numIter += 1 
    #5) Report results 
    stats = pd.DataFrame.from_dict(stats, orient = 'columns') 
    stats.to_csv('MC-DCCA-bal5-sw250.csv') 
    df0 = stats.std()
    df1 = stats.var() 
    print(pd.concat([df0, df1, df1 / df1['getHRP'] - 1], axis = 1))
    return 
#------------------------------------------------------------------------------ 

if __name__=='__main__':hrpMC()