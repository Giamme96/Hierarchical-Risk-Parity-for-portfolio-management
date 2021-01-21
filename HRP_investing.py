from datetime import datetime      
from datetime import timedelta     
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

import scipy.cluster.hierarchy as sch,random
import matplotlib.pyplot as mpl 

import investpy as inv  
import pandas as pd     
import numpy as np

import seaborn as sns
from scipy.stats import kurtosis

from scipy.stats import skew
from scipy import signal
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

    country = "united states"
    df = pd.DataFrame(inv.get_stocks(country = country))
    df = np.array(df["isin"])
    array_pulito = []

    for i in df:
        if i[:2] == "US":
            array_pulito.append(i)

    df = pd.DataFrame(array_pulito)
    df.columns = ["Isin"]
    head = df[:n_asset_mkt].values.tolist()

    return head
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
        #print("\nEcco cItems dopo la bisezione: ", cItems)
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

def plotCorrMatrix(path, corr, labels = None): 
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

    return df, info_gen, #info_tech
# ------------------------------------------------------------------------------ 

def getListaRend(lista_isin):   #crea lista di tutti i titoli con i propri rendimenti in log sulle righe

    lista_rendimenti = []
    for i in lista_isin:

        df_stock = getDailyFromStock(i)
        rendimento = getRendimento(df_stock)
        lista_rendimenti.append(rendimento)

    return lista_rendimenti
# ------------------------------------------------------------------------------ 

def getRendimento(dataframe):    #restituisce il rendimento elimina i Nan

    # rend = pd.Series(np.diff(dataframe[0].Close) , name = dataframe[1].iloc[-1].symbol).dropna() 
    # print("rendiemnto:", rend)
    # rendlog = pd.Series(np.log(dataframe[0].Close.pct_change()), name = dataframe[1].iloc[-1].symbol).dropna()
    rend = pd.Series(dataframe[0].Close.pct_change(), name = dataframe[1].iloc[-1].symbol).dropna()
    return rend
# ------------------------------------------------------------------------------ 

def getStdPortafoglio(vettore_hrp, matrice_cov):        #restituisce l'STD di portafoglio

    var_portafoglio = np.dot(vettore_hrp, np.dot(matrice_cov, vettore_hrp.T))

    dev_std_portafoglio = np.sqrt(var_portafoglio)

    return dev_std_portafoglio
# ------------------------------------------------------------------------------ 

def getRendimentoPortafoglio(vettore_hrp, vettore_rendimenti):  

    rendimento_portafoglio = np.dot(vettore_hrp, vettore_rendimenti.mean().T)      #Il vettore rendimenti è la media dei rendimenti per ogni asset, viene trasposto

    return rendimento_portafoglio
# ------------------------------------------------------------------------------ 

def getSharpeRatio(rendimento_port, std_port, risk_free):

    sharperatio = (rendimento_port - risk_free) / std_port
    
    return sharperatio
# ------------------------------------------------------------------------------ 

def main(): 

    lista_assets_isin = ["US5949181045", "US0378331005", "US64110L1061", "US88160R1014", "US67066G1040", "US4581401001", "US0970231058", "US38141G1040", "US6541061031", "US5801351017"]   #titoli del portafoglio

    # lista_assets_isin = ["US64110L1061", "US88160R1014"]
    lista_rend= pd.DataFrame(getListaRend(lista_assets_isin))
    lista_rend_t = pd.DataFrame(lista_rend.T)     #lista trasposta per avere i titoli sulle colonne e i rendimenti sulle righe

    rolling_window = 20                                                                           #ROLLING WINDOW PER DCCA
    
    # print("\n Lista rendimenti:\n", lista_rend_t)
    # lista_rend_t_detrended = pd.DataFrame(signal.detrend(lista_rend_t, axis = 1))                 #Prova detrending
    # print("\n Lista rendimenti detrended: ", lista_rend_t_detrended.shape)
    # lista_rend_t_detrended.to_csv("Detrending asse 1.csv")

    print("Rendimenti medi dei titoli", lista_rend_t.mean())                             #statistiche descrittive dei titoli
                                                                      

    #2) compute and plot correl matrix 
    
    dcca_data = np.array(lista_rend_t)
    dcca = pd.DataFrame(compute_dpcca_others(dcca_data, rolling_window))
    dcca.columns = lista_rend_t.columns             #Brutto modo per fare il rename delle colonne e righe
    dcca = dcca.T
    dcca.columns = lista_rend_t.columns
    
    corr = dcca                                     #DCCA UTILIZZATO COME MATRICE CORR

    cov = lista_rend_t.cov()
    corra = lista_rend_t.corr()
    plotCorrMatrix('Heat1.png', corr, labels = corr.columns)  

    plotCorrMatrix('HRP3_dcca_beforeCluster.png', corr, labels = corr.columns)  
    plotCorrMatrix('HRP3_corr_beforeCluster.png', corra, labels = cov.columns) 

    #3) cluster 
    dist = correlDist(corr) 
    link = sch.linkage(dist, 'ward') 
    sortIx = getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist() # recover labels  
    
    print("sortix", sortIx)
    df0 = corr.loc[sortIx, sortIx] # reorder        #Matrice di correlazioni tra clusters
    plotCorrMatrix('Heat2.png', df0, labels = df0.columns)

    # sch.dendrogram(link, labels = lista_rend_t.columns)
    # mpl.savefig("Dendrogramma.png")

    #4) Capital allocation 
    hrp = getRecBipart(cov, sortIx)         #Peso in portafoglio assegnato ad ogni cluster somma = 1 come vincolo
    print("\nI pesi assegnati (HRP), sum~1: \n", hrp)    
    #print(sum(hrp))   

    #5) Strumenti per il portafoglio
    std_p = getStdPortafoglio(hrp, cov)
    rend_p = getRendimentoPortafoglio(hrp, lista_rend_t)
    sr_p = getSharpeRatio(rend_p, std_p, -0.00498)     #risk free arbitrario
   
    print("\nStd, Rend, Sharpe ratio: (daily basis)", std_p, rend_p, sr_p)  


    # ################################################## HISTO skew NFLX
    # skewness = skew(lista_rend_t.NFLX)
    # mpl.hist(lista_rend_t.NFLX, color = "r", bins = 50, label = "NFLX")
    # # mpl.plot(lista_rend_t.NFLX, color = "r", label = "NFLX returns")
    # plt.legend()
    # plt.xlabel("Rendimenti")
    # plt.ylabel("Frequenza")
    # plt.title("Distribuzione rendimenti")
    # # mpl.show()
    # mpl.savefig("Skew_NFLX")
    # print(skewness)
    # ################################################## HISTO skew NFLX

   
    
    # # lista_plot = [lista_rend_t.NFLX, lista_rend_t.MCD, lista_rend_t.TSLA]
    # # pd_plot = pd.DataFrame(lista_plot)
    # graph = lista_rend_t.plot.kde()
    # graph.set_xlim([-0.25, +0.25])
    # plt.xlabel("Rendimenti")
    # plt.ylabel("Densità")
    # plt.title("Distribuzione dei rendimenti")
    # kurt = kurtosis(lista_rend_t, fisher = True)
    # print("KURTOSIS", kurt)
    # # plt.show()
    # plt.savefig("Kurt_NFLX_TSLA")
    
    return 


if __name__=='__main__':main()



#TODO Non tiene conto dei frazionamenti azionari e dividendi pagati
#TODO Come interpretre gli indici ottenuti? Rendimento medio giornaliero?
