# Hierarchical Risk Parity for portfolio management
This work is a part of my thesis (Ottimizzazione della gestione del portafoglio tramite tecniche di clustering gerarchico) in collaboration with Stefano Ferretti (https://www.unibo.it/sitoweb/s.ferretti) starting with the idea of an algorithm that can automatically develops weight's for the asset composition in an arbitrary stock portfolio. The raw algorithm was introduced for the first time by Marcos Lopez De Prado in 2016 with "Building diversified portfolios that outperform out of sample", SSRN-id2708678. A big part of the code is referred to his work, the substantial difference is regarding the correlation matrix construction (using DCCA (__1__)) and an upgrade concerning a wider option like a in-real-time stock market data(choosing a pool of assets or using random combination (__2__) and a maximisation of Sharpe Ratio as a performance measure). 

## References

(__1__) DCCA is a method for estimating correlation matrix holding the requirement of cross-correlation between time series.
* "DCCA cross-correlation coefficient with sliding windows approach", E.F. Guedes, G.F. Zebende, www.elsevier.com/locate/physa.
* "Detrending moving-average cross-correlation coefficient:Measuring cross-correlations between non-stationary series", Ladislav Kristoufek, www.elsevier.com/locate/physa.
* "A sliding window approach to detrended fluctuation analysis of heart rate variability", Daniel L. F. Almeida, Fabiano A. Soares, and Joao L. A. Carvalho, ISBN:978-1-4577-0216-7.
* Code side i'm thankful for the work about DCCA in python (https://gist.github.com/jaimeide/a9cba18192ee904307298bd110c28b14).

(__2__) Data is taken from API (https://github.com/alvarobartt/investpy) facing one of the most famous financial data platform Investing (https://it.investing.com/).

## Assumptions
* Algorithim is built for return the min variance. So the main idea is lower risk and it mean's lower return.
* Returns and measures are in Daily basis.

## How does it works (Wombo_combo.py)
If the aim of the utilization is just a pratic sense of what could do a management algorithm into the market the right way to use it is working with _Wombo_combo.py_ otherwise you can try some theoretical ways using _HRP.py_ and _Montecarlo.py_

After you set up all the variables described before the algo start working, it takes few seconds, the longest part is downloading data from API.
* First of all the algo gives IVP composition, and then HRP composition in terms of % weight over the total budget. About the HRP output there's some easy measures like Standard Deviation, Return and Sharpe Ratio. These measures are on daily basis.
* There are 2 paths to follow, the main one takes in input a list __lista_investing__ with assets the user want to bring into portfolio. The second path regard a composition built with combinations that explain the best Sharpe Ratio among the head __n_asset_mkt__. As i explain in (__3__) is not random for now.

## First Steps (Wombo_combo.py)

![Inputs](/Input_vars.png)

* __lista_investing__, use the ISIN of the asset you want to add into the algo. This var is not used if the __input_method__ is _"combo"_.
* __rolling_window__, is the size of the rolling window used into the DCCA method. This var in not used if the __corr_est_method__ is _"normale"_.
* __riskfree__, is the yield of an arbitrary term structure used in the computation of Sharpe Ration.
* __n_asset_port__, is referred to the number of asset you want into portfolio picked randomly (rn is not random, just the header of all title retrieved by API) in the pool of __n_asset_mkt__(__3__).

![Combinations](/Combination.png)

* __corr_est_method__, you can choose between _"normale"_ as a classic correlation matrix estimate and _"dcca"_(__1__).
* __input_method__, you can choose between _"investing"_ where the algo takes in input a list of asset of your choice(ISIN) or _"combo"_ representing the random way for picking assets from market.
* __linkage_method__, i used to work with _"ward"_ method, btw you can follow different paths (https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html)

## What about others files?
* _HRP.py_ file is the main work of De Prado. It computes the HRP and IVP using a normal distribution + an inducted correlation.
* _DPCCA.py_ is the code regarding the DCCA process (__1__) adapted for the structure used in main files.
* _Montecarlo.py_ is used to compute all the MC simulations and then validating the Out of Sample thesis.

If you want to change the method for simulations in _Montecarlo.py_.
![Montecarlo_method](/Montecarlo_method.png)


## The output? (Wombo_combo.py)

![HRP_out](/HRP_output.png)
![IVP_out](/IVP_output.png)

