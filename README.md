# Hierarchical Risk Parity for portfolio management (work in progress)
This work is a part of my thesis (Ottimizzazione della gestione del portafoglio tramite tecniche di clustering gerarchico) in collaboration with Stefano Ferretti (https://www.unibo.it/sitoweb/s.ferretti) starting with the idea of an algorithm that can automatically develops weight's for the asset composition in an arbitrary stock portfolio. The algorithm was introduced for the first time by Marcos Lopez De Prado in 2016 with "Building diversified portfolios that outperform out of sample", SSRN-id2708678. A big part of the code is referred to his work, the substantial difference is regarding the correlation matrix construction (using DCCA (1)) and an upgrade concerning a wider option like a in-real-time stock market data(choosing a pool of assets or using random combination (2)). 

(__1__) DCCA is a method for estimating correlation matrix holding the requirement of cross-correlation between time series.
* "DCCA cross-correlation coefficient with sliding windows approach", E.F. Guedes, G.F. Zebende, www.elsevier.com/locate/physa.
* "Detrending moving-average cross-correlation coefficient:Measuring cross-correlations between non-stationary series", Ladislav Kristoufek, www.elsevier.com/locate/physa.
* "A sliding window approach to detrended fluctuation analysis of heart rate variability", Daniel L. F. Almeida, Fabiano A. Soares, and Joao L. A. Carvalho, ISBN:978-1-4577-0216-7.

(__2__) Data is taken from API (https://github.com/alvarobartt/investpy) facing one of the most famous financial data platform Investing (https://it.investing.com/).

## First Steps

![Inputs](/Input_vars.png)

* __lista_investing__, use the ISIN of the asset you want to add into the algo. This var is not used if the __input_method__ is _"combo"_.
* __rolling_window__, is the size of the rolling window used into the DCCA method. This var in not used if the __corr_est_method__ is _"normale"_.
* __riskfree__, is the yield of an arbitrary term structure used in the computation of Sharpe Ration.
* __n_asset_port__, is referred to the number of asset you want into portfolio picked randomly (rn is not random, just the header of all title retrieved by API) in the pool of __n_asset_mkt__(__3__).

![Combinations](/Combination.png)

* __corr_est_method__, you can choose between _"normale"_ as a classic correlation matrix estimate and _"dcca"_(__1__).
* __input_method__, you can choose between _"investing"_ where the algo takes in input a list of asset of your choice(ISIN) or _"combo"_ representing the random way for picking assets from market.
* __linkage_method__, i used to work with _"ward"_ method, btw you can follow different paths (https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html)

## How does it works
After yuo set up all the variables described before the algo start working, it takes few seconds, the longest part is downloading data from API.
* First of all the algo gives IVP composition, and then HRP composition in terms of % weight over the total budget. About the HRP output there's some easy measures like Standard Deviation, Return and Sharpe Ration. These measures are on daily basis.
