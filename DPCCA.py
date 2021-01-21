
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import numpy as np
from numpy.matlib import repmat

from datetime import datetime      #dal GIT
from datetime import timedelta      #dal GIT

import investpy as inv      #dal GIT
import pandas as pd      #dal GIT
from scipy import signal
##########################

def sliding_window(xx, k):
    # Function to generate boxes given dataset(xx) and box size (k)

    # generate indexes! O(1) way of doing it :)
    idx = np.arange(k)[None, :] + np.arange(len(xx) - k + 1)[:, None]
    return xx[idx], idx

def compute_dpcca_others(cdata,k):              
    # Input: cdata(nsamples,nvars), k: time scale for dpcca
    # Output: dcca, dpcca, corr, partialCorr
    #
    # Date(last modification): 02/15/2018
    # Author: Jaime Ide (jaime.ide@yale.edu)
    
    # Code distributed "as is", in the hope that it will be useful, but WITHOUT ANY WARRANTY;
    # without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
    # See the GNU General Public License for more details.
    
    # Define
    nsamples, nvars = cdata.shape

    # Cummulative sum after removing mean
    #cdata = signal.detrend(cdata,axis=0) # different from only removing the mean...
    cdata = cdata - cdata.mean(axis = 0)
    xx = np.cumsum(cdata, axis = 0)
    
    F2_dfa_x = np.zeros(nvars)
    allxdif = []
    # Get alldif and F2_dfa
    for ivar in range(nvars): # do for all vars
        xx_swin , idx = sliding_window(xx[:, ivar], k)
        nwin = xx_swin.shape[0]
        b1, b0 = np.polyfit(np.arange(k), xx_swin.T, deg = 1) # linear fit (UPDATE if needed)
        
        #x_hat = [[b1[i]*j+b0[i] for j in range(k)] for i in range(nwin)] # Slower version
        x_hatx = repmat(b1, k, 1).T * repmat(range(k), nwin, 1) + repmat(b0, k,  1).T
    
        # Store differences to the linear fit
        xdif = xx_swin - x_hatx
        allxdif.append(xdif)
        # Eq.4
        F2_dfa_x[ivar] = (xdif ** 2).mean()
    # Get the DCCA matrix
    dcca = np.zeros([nvars, nvars])
    for i in range(nvars): # do for all vars
        for j in range(nvars): # do for all vars
            # Eq.5 and 6
            F2_dcca = (allxdif[i] * allxdif[j]).mean()
            # Eq.1: DCCA
            dcca[i, j] = F2_dcca / np.sqrt(F2_dfa_x[i] * F2_dfa_x[j])   
    
    # Get DPCCA
    C = np.linalg.inv(dcca)
    
    # (Clear but slow version)
    #dpcca = np.zeros([nvars,nvars])
    #for i in range(nvars):
    #    for j in range(nvars):
    #        dpcca[i,j] = -C[i,j]/np.sqrt(C[i,i]*C[j,j])
    
    # DPCCA (oneliner version)
    mydiag = np.sqrt(np.abs(np.diag(C)))
    # dpcca = (-C / repmat(mydiag, nvars, 1).T) / repmat(mydiag, nvars, 1) + 2 * np.eye(nvars)          #DPCCA
    
    # Include correlation and partial corr just for comparison ;)
    # Compute Corr
    # corr = np.corrcoef(cdata.T)       #CORRCOEF
    # Get parCorr
    cov = np.cov(cdata.T)
    C0 = np.linalg.inv(cov)
    mydiag = np.sqrt(np.abs(np.diag(C0)))
    # parCorr = (-C0 / repmat(mydiag, nvars, 1).T) / repmat(mydiag, nvars, 1) + 2 * np.eye(nvars)       #PARCORR
    
    return dcca #corr, parCorr, dpcca

