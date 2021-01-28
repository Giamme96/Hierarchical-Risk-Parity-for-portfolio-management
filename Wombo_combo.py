from datetime import datetime      
from datetime import timedelta     
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from itertools import combinations

import scipy.cluster.hierarchy as sch,random
import matplotlib.pyplot as mpl 

import investpy as inv  
import pandas as pd     
import numpy as np

# from HRP_investing import getStdPortafoglio, getRendimentoPortafoglio, getSharpeRatio
from DPCCA import compute_dpcca_others

#------------------------------------------------------------------------------ 
country_isin = {            #dict country
    "US" : "united states",
    "IT" : "italy",
    "GB" : "great britain",
    "NL" : "netherlands",
    "FR" : "france",
    "IE" : "ireland"
    }
#------------------------------------------------------------------------------
def getStocksIsin(n_asset_mkt):

    df = pd.DataFrame(inv.get_stocks(country = country_isin.get("US"))) #get infos about US market
    df = np.array(df["isin"])
    array_pulito = []

    for i in df:
        if i[:2] == "US":
            array_pulito.append(i)

    df = pd.DataFrame(array_pulito)
    df.columns = ["Isin"]
    head = df[:n_asset_mkt].values.tolist()     #get first n_asset_mkt
    head_lista = []
    for i in head:
        head_lista.append(i[0])

    return head_lista #First 15obs from mkt
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
    cVar = np.dot(np.dot(w_.T, cov_), w_)[0, 0]        #.T, trasposta #after matrix dot 1x5 x 5x5 = 1x5, and 1x5 x 5x1 from weight, dtot result of Cvar = 1x1
    #print("\nVarainza del cluster:", cVar)

    return cVar 
#------------------------------------------------------------------------------ 

def getQuasiDiag(link):             #kinda same assets are near the diagonal, vice versa
    # Sort clustered items by distance 
    link = link.astype(int) 
    sortIx = pd.Series([link[-1, 0], link[-1, 1]])     
    #print("link:", link)
    numItems = link[-1, 3] # number of original items    #last row last column tot assets (size0+size1=10)  
    while sortIx.max() >= numItems: 
        sortIx.index = range(0, sortIx.shape[0] * 2, 2) # make space #index 
        df0 = sortIx[sortIx >= numItems] # find clusters
        i = df0.index 
        j = df0.values - numItems 
        sortIx[i] = link[j, 0] # item 1 
        df0 = pd.Series(link[j, 1], index = i + 1) 
        sortIx = sortIx.append(df0) # item 2 
        sortIx = sortIx.sort_index() # re-sort 
        #print("Sortix prima dell'indexing", sortIx)
        sortIx.index = range(sortIx.shape[0]) # re-index
    #print("Sortix finale", sortIx, "\nDf0 finale:", df0) 
    return sortIx.tolist() 
#------------------------------------------------------------------------------ 

def getRecBipart(cov, sortIx): #return weight of every group of asset using iterative process

    # Compute HRP allocation
    w = pd.Series(1, index = sortIx) 
    cItems = [sortIx] # initialize all items in one cluster 
    while len(cItems) > 0:          
        cItems = [i[j:k] for i in cItems for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1] # bi-section      
        #print("\nEcco cItems dopo la bisezione: ", cItems)
        for i in range(0, len(cItems), 2):  # parse in pairs, ++ 2
            cItems0 = cItems[i] # cluster 1 
            cItems1 = cItems[i + 1] # cluster 2 
            cVar0 = getClusterVar(cov, cItems0) #varinza del primo cluster dopo la bisez, i primi 5 numeri, poi 3, 2 ecc...
            cVar1 = getClusterVar(cov, cItems1) 
            alpha = 1 - cVar0 / (cVar0 + cVar1) 
            #print("L'alpha dalle Cvar (sono Ncluster/2) è:", alpha)
            w[cItems0] *= alpha # weight 1 
            w[cItems1] *= 1 - alpha # weight 2
            #print("\nPesi printati:", w[cItems0], w[cItems1]) 
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

def getDailyFromStock(isin):        

    years_obs = timedelta(days=365.24) * 1      #~250 obs
    endus = datetime.now()                      
    endeu = endus.strftime("%d/%m/%Y")          #EU to US date conversion
    startus = endus - years_obs
    starteu = startus.strftime("%d/%m/%Y")

    country_iniziali = isin[:2]         #ISIN conversion
    country = country_isin.get(country_iniziali)

    periodicita = "Daily"           #time basis from API, daily-weekly-monthly...

    stock = inv.stocks.get_stocks(country = country)
    info_gen = stock.loc[stock["isin"] == isin]
    #info_tech = inv.stocks.get_stock_information(info_gen["symbol"].values[0], country, as_json=False)
    df = inv.get_stock_historical_data(stock = info_gen["symbol"].values[0], country = country, 
        from_date = starteu, to_date = endeu, as_json=False, order='ascending', interval = periodicita)

    df = df.dropna()
    if df.Volume[-1] == 0:
        df = df.iloc[:-1]

    return df, info_gen, #info_tech
# ------------------------------------------------------------------------------ 

def getListaRend(lista_isin):   #get data from asset isin

    lista_rendimenti = []
    for i in lista_isin:

        df_stock = getDailyFromStock(i)
        rendimento = getRendimento(df_stock)
        lista_rendimenti.append(rendimento)

    return lista_rendimenti
# ------------------------------------------------------------------------------ 

def getRendimento(dataframe):    #get returns from data, drop Nan values

    rend = pd.Series(dataframe[0].Close.pct_change(), name = dataframe[1].iloc[-1].symbol).dropna()

    return rend
# ------------------------------------------------------------------------------ 

def getStdPortafoglio(vettore_hrp, matrice_cov):        #get standard deviation from data

    var_portafoglio = np.dot(vettore_hrp, np.dot(matrice_cov, vettore_hrp.T))
    dev_std_portafoglio = np.sqrt(var_portafoglio)

    return dev_std_portafoglio
# ------------------------------------------------------------------------------ 

def getRendimentoPortafoglio(vettore_hrp, vettore_rendimenti):  #get portfolio return's

    rendimento_portafoglio = np.dot(vettore_hrp, vettore_rendimenti.mean().T)      

    return rendimento_portafoglio
# ------------------------------------------------------------------------------ 

def getSharpeRatio(rendimento_port, std_port, risk_free):   #get sharpe ration from portfolio

    sharperatio = (rendimento_port - risk_free) / std_port
    
    return sharperatio
# ------------------------------------------------------------------------------ 

def getCovCorr(dataframe_returns, rolling_window, corr_est_method):             #-----corr_est_method = "normale", corr_est_method = "dcca"--------- #maybe could be improved with **kwargs
    
    #2) compute and plot correl matrix 
    if corr_est_method == "normale":

        corr = dataframe_returns.corr()

    elif corr_est_method == "dcca":                               

        dcca_data = np.array(dataframe_returns)
        dcca = pd.DataFrame(compute_dpcca_others(dcca_data, rolling_window))

        dcca.columns = dataframe_returns.columns             #to improve, really bad form
        dcca = dcca.T
        dcca.columns = dataframe_returns.columns
        corr = dcca
             
    cov = dataframe_returns.cov()
    print("\n", list(dataframe_returns.columns))
    print("IVP method: ", getIVP(cov), "\n")
    # plotCorrMatrix('HRP3_corr0Stock.png', corr, labels = corr.columns)  
    return cov, corr
# ------------------------------------------------------------------------------

def getHRP(cov, corr, linkage_method):
    
    #3) cluster 
    dist = correlDist(corr) 
    link = sch.linkage(dist, linkage_method) 
    sortIx = getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist() # recover labels    
    df0 = corr.loc[sortIx, sortIx] # reorder       
    # plotCorrMatrix('HRP3_corr1Stock.png', df0, labels = df0.columns)

    #4) Capital allocation 
    hrp = getRecBipart(cov, sortIx)         #portfolio weights over iterative steps

    return hrp
# ------------------------------------------------------------------------------ 

def getPortfolioInfos(hrp, cov, dataframe_returns, riskfree):
    
    #5) Portfolio tools calc.
    std_p = getStdPortafoglio(hrp, cov)
    rend_p = getRendimentoPortafoglio(hrp, dataframe_returns)
    sr_p = getSharpeRatio(rend_p, std_p, riskfree)     

    return std_p, rend_p, sr_p
# ------------------------------------------------------------------------------ 

def getWomboCombo(dataframe_returns, n_asset_port, riskfree, rolling_window, corr_est_method, linkage_method):  

    array_wombocombos = []   #array with best comb and infos(hrp, std, rend, SR)
    array_posizionale_sr = []
    comb = list(combinations(dataframe_returns.columns, n_asset_port))      
    comb = np.asarray(comb)         #get combinations from assets retrieved
    for i in comb:

        df_combo = dataframe_returns[i]
        cov, corr = getCovCorr(df_combo, rolling_window, corr_est_method)
        hrp = getHRP(cov, corr, linkage_method)
        std_p, rend_p, sr_p = getPortfolioInfos(hrp, cov, df_combo, riskfree)   #get infos about simulated portfolio
        array_wombocombos.append(np.array([hrp, std_p, rend_p, sr_p]))
        array_posizionale_sr.append(np.array(sr_p))

    return array_wombocombos, array_posizionale_sr
# ------------------------------------------------------------------------------ 

def getInputMethod(input_method, lista_investing, n_asset_mkt_combo, n_asset_port, corr_est_method, linkage_method, riskfree, rolling_window):   #"investing" as asset list in input, "combo" as random combination

    if input_method == "investing":

        lista_rend = pd.DataFrame(getListaRend(lista_investing))
        lista_rend_t = pd.DataFrame(lista_rend.T)     #lista trasposta per avere i titoli sulle colonne e i rendimenti sulle righe
        cov, corr = getCovCorr(lista_rend_t, rolling_window, corr_est_method)
        # print("Covariance", cov)
        # print("Correlation", corr)
        hrp = getHRP(cov, corr, linkage_method)
        std_p, rend_p, sr_p = getPortfolioInfos(hrp, cov, lista_rend_t, riskfree)   #get infos about simulated portfolio
        results = [hrp, std_p, rend_p, sr_p]

        return results

    elif input_method == "combo":

        lista_assets_isin = getStocksIsin(n_asset_mkt_combo)
        lista_rend = pd.DataFrame(getListaRend(lista_assets_isin))
        lista_rend_t = pd.DataFrame(lista_rend.T)     #lista trasposta per avere i titoli sulle colonne e i rendimenti sulle righe

        array_wombocombos, array_posizionale_sr = getWomboCombo(lista_rend_t, n_asset_port, riskfree, rolling_window, corr_est_method, linkage_method)
        sr_max = np.argmax(array_posizionale_sr)
        wombocombo = array_wombocombos[sr_max]

        return wombocombo
# ------------------------------------------------------------------------------

def main(): 
    #1) INPUT
    lista_investing = ["US5949181045", "US0378331005", "US64110L1061", "US88160R1014", "US67066G1040", "US4581401001", "US0970231058", "US38141G1040", "US6541061031", "US5801351017"]   #asset's pool in the market
    rolling_window = 50 #rolling window for DCCA
    riskfree = -0.00586  #risk free yield for sharpe ratio
    n_asset_port = 2 #asset number inside portfolio
    n_asset_mkt = 5   #assets picked from mkt 

    ########################EXE METHOD
    corr_est_method = "normale"    #compute method for estimating correlation matrix
    input_method = "investing"  #"investing" as asset list in input, "combo" as random combination
    linkage_method = "ward"     #different kind of linkage approach, https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
    ########################EXE METHOD

    #2) MAIN
    composition = getInputMethod(input_method, lista_investing, n_asset_mkt, n_asset_port, corr_est_method, linkage_method, riskfree, rolling_window)
    print("\n The best composition is (HRP%-STD-YIELD-SR): \n", composition)
                                                                                                                           
    return 


if __name__=='__main__':main()

