# Hierarchical Risk Parity for portfolio management (work in progress)
This work is a part of my thesis (Ottimizzazione della gestione del portafoglio tramite tecniche di clustering gerarchico) in collaboration with Stefano Ferretti (https://www.unibo.it/sitoweb/s.ferretti) starting with the idea of an algorithm that can automatically develops weight's for the asset composition in an arbitrary stock portfolio. The algorithm was introduced for the first time by Marcos Lopez De Prado in 2016 with "Building diversified portfolios that outperform out of sample", SSRN-id2708678. A big part of the code is referred to his work, the substantial difference is regarding the correlation matrix construction (using DCCA (1)) and an upgrade concerning a wider option like a in-real-time stock market data(choosing a pool of assets or using random combination (2)). 

(1) DCCA is a method for estimating correlation matrix holding the requirement of cross-correlation between time series.
* "DCCA cross-correlation coefficient with sliding windows approach", E.F. Guedes, G.F. Zebende, www.elsevier.com/locate/physa.
* "Detrending moving-average cross-correlation coefficient:Measuring cross-correlations between non-stationary series", Ladislav Kristoufek, www.elsevier.com/locate/physa.
* "A sliding window approach to detrended fluctuation analysis of heart rate variability", Daniel L. F. Almeida, Fabiano A. Soares, and Joao L. A. Carvalho, ISBN:978-1-4577-0216-7.

(2) Data is taken from API (https://github.com/alvarobartt/investpy) facing one of the most famous financial data platform Investing (https://it.investing.com/).

* First Steps


