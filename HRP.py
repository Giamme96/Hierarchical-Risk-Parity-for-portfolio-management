#On 20151227 by MLdP <lopezdeprado@lbl.gov> 
# Hierarchical Risk Parity 
import matplotlib.pyplot as mpl
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch,random
import numpy as np
import pandas as pd
from DPCCA import compute_dpcca_others
from HRP_investing import getStdPortafoglio, getSharpeRatio, getRendimentoPortafoglio
#------------------------------------------------------------------------------ 

def getIVP(cov, **kargs): 
    # Compute the inverse-variance portfolio 
    ivp = 1. / np.diag(cov) 
    ivp /= ivp.sum() 
    return ivp 
#------------------------------------------------------------------------------ 

def getClusterVar(cov, cItems): 
    # Compute variance per cluster 
    cov_  = cov.loc[cItems, cItems]  # matrix slice
    #print("\nLa cov_ con labels assets già bisez.:\n", cov_) 
    w_ = getIVP(cov_).reshape(-1, 1)
    #print("\nIl peso W_:", w_)                    #tutti i valori su una singola colonna Matrice [N,1]
    cVar = np.dot(np.dot(w_.T, cov_), w_)[0, 0]        #.T, trasposta #dopo l moltiplicazione matriciale 1x5 x 5x5 = 1x5, ancora 1x5 x 5x1 del peso, dando come risultato Cvar = 1x1
    #print("\nVarainza del cluster:", cVar)
    return cVar 
#------------------------------------------------------------------------------ 

def getQuasiDiag(link):             #i valori più grossi sono nella diagonale, gli investimenti simili sono raggruppati assieme, quelli diversi sono lontani
    # Sort clustered items by distance 
    link = link.astype(int) 
    sortIx = pd.Series([link[-1, 0], link[-1, 1]])     #Valori in basso a sinistra della matrice
    # print("link:", link)
    numItems = link[-1, 3] # number of original items    #Ultima riga ultima colonna c'è il numero di elementi (size0+size1=10)  
    while sortIx.max() >= numItems: 
        sortIx.index = range(0, sortIx.shape[0] * 2, 2) # make space #index restituisce il valore associato all'indice posizionale 0,1,2,3
        df0 = sortIx[sortIx >= numItems] # find clusters
        i = df0.index 
        j = df0.values - numItems 
        sortIx[i] = link[j, 0] # item 1 
        df0 = pd.Series(link[j, 1], index = i + 1) 
        sortIx = sortIx.append(df0) # item 2 
        sortIx = sortIx.sort_index() # re-sort 
        # print("Sortix prima dell'indexing", sortIx)
        sortIx.index = range(sortIx.shape[0]) # re-index
    # print("Sortix finale", sortIx, "\nDf0 finale:", df0) 
    return sortIx.tolist() 
#------------------------------------------------------------------------------ 

def getRecBipart(cov, sortIx): #divide il peso tot = 1 per i diversi cluster(2 poi 4 ecc..), w e 1-w, fino a che non ci sono più cluster in cItems e quindi tutti i pesi distribuiti per gli asset
    # Compute HRP alloc 
    w = pd.Series(1, index = sortIx) 
    cItems = [sortIx] # initialize all items in one cluster 
    while len(cItems) > 0:          #quando non ci sono più cluster la condizione diventa falsa
        cItems = [i[j:k] for i in cItems for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1] # bi-section      
        # print("\nEcco cItems dopo la bisezione: ", cItems)
        for i in range(0, len(cItems), 2):  # parse in pairs, incrementa di 2
            cItems0 = cItems[i] # cluster 1 
            cItems1 = cItems[i + 1] # cluster 2 
            cVar0 = getClusterVar(cov, cItems0) #varinza del primo cluster dopo la bisez, i primi 5 numeri, poi 3, 2 ecc...
            cVar1 = getClusterVar(cov, cItems1) 
            alpha = 1 - cVar0 / (cVar0 + cVar1) 
            #print("L'alpha dalle Cvar (sono Ncluster/2) è:", alpha)
            w[cItems0] *= alpha # weight 1 
            w[cItems1] *= 1 - alpha # weight 2
            # print("\nPesi printati:", w[cItems0], w[cItems1]) 
    return w 
#------------------------------------------------------------------------------ 

def correlDist(corr): 
    # A distance matrix based on correlation, where 0<=d[i,j]<=1 
    # This is a proper distance metric 
    dist = ((1 - corr) / 2.) ** .5 # distance matrix
    #print("distanze:", dist)
    return dist 
#------------------------------------------------------------------------------ 

def plotCorrMatrix(path , corr, labels = None): 
    #Heatmap of the correlation matrix 
    if labels is None: labels = []
    mpl.pcolor(corr)
    mpl.colorbar() 
    mpl.yticks(np.arange(.5, corr.shape[0] + .5), labels) 
    mpl.xticks(np.arange(.5, corr.shape[0] + .5), labels) 
    mpl.savefig(path) 
    mpl.clf(); mpl.close() # reset pylab 
    return 
# ------------------------------------------------------------------------------ 

def generateData(nObs, size0, size1, sigma1): 
    # Time series of correlated variables 
    #1) generating some uncorrelated data 
    np.random.seed(seed = 12345); random.seed(12345) #de prado
    x = np.random.normal(0, 1, size = (nObs, size0)) # each row is a variable, le colonne sono gli asset e le righe i rendimenti????????
    #2) creating correlation between the variables 
    cols = [np.random.randint(0, size0 - 1) for i in range(size1)]         #xrange restituisce un obj range, allocazione meno spazio
    y = x[:, cols] + np.random.normal(0, sigma1, size = (nObs, len(cols)))   #[:,y] tutti gli elementi della colonna y, prende tutti gli elementi della colonna in base al valore rand di cols[i]
    x = np.append(x, y, axis = 1) 
    x = pd.DataFrame(x, columns = range(1, x.shape[1] + 1))             #restituisce una matrice [Obs x Obs] le prime size0=5 colonne sono Random, le altre size1=5 colonne sono valori modificati in pasto a y=cols + x
    return x, cols 
#------------------------------------------------------------------------------ 

def main(): 
    
    #1) Generate correlated data 
    nObs, size0, size1, sigma1 = 10000, 5, 5, .25        #DRIFT size0 = incorrelato, size1 = correlato
    rolling_window = 125                                                #Rolling window per la DCCA, cambiano anche i pesi assegnati in HRP, deve essere inferiore al nObs
    x, cols = generateData(nObs, size0, size1, sigma1) 

    #2) compute and plot correl matrix 
    #USING DCCA##########################
    # corr_normale = x.corr()
    corr = pd.DataFrame(compute_dpcca_others(np.array(x), rolling_window), columns = range(1, x.shape[1] + 1))      #Brutto modo per rinominare index e columns
    corr = corr.T
    corr.columns = range(1, x.shape[1] + 1)
    #USING DCCA##########################
    cov = x.cov()
    # corr = x.corr()
    # print("\nMatrice obs:\n", x, "\nCorrelazioni:\n", corr, "\nCorrelazioni normali:\n", corr_normale)
    plotCorrMatrix('Heat_DCCA_HRP.png', corr, labels = corr.columns)  
    #3) cluster 
    dist = correlDist(corr) 
    link = sch.linkage(dist, 'ward') 
    #print("Ecco il link:", link)
    sortIx = getQuasiDiag(link)
    # print("Sortix before tolist:", sortIx, "\n")
    sortIx = corr.index[sortIx].tolist() # recover labels 
    # print("Sortix post tolist:", sortIx, "\n")  
       
    df0 = corr.loc[sortIx, sortIx] # reorder        #Matrice di correlazioni tra clusters
    # print("df0 prima del plotting", df0)
    plotCorrMatrix('Heat_DCCA_HRP_clust.png', df0, labels = df0.columns)
    mpl.hist(x, bins = 5)
    
    # sch.dendrogram(link)
    # mpl.show()

    #4) Capital allocation 
    #print("\nDistance\n", dist, "\nLinkage crea i cluster:\n", link, "\nSortix dopoo aver indexato:\n", sortIx, "\ndf0 riordinato (correlazioni tra cluster):\n", df0)
    hrp = getRecBipart(cov, sortIx)         #Peso in portafoglio assegnato ad ogni cluster somma = 1 come vincolo
    print("\n I pesi assegnati (HRP), sum~1:\n", hrp)      
    return 
#------------------------------------------------------------------------------ 
if __name__=='__main__':main()


