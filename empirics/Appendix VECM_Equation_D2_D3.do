* Title: On a Pecuniary Externality of Competitive Banking through Goods Pricing Dispersion
* Copyright: Timothy Kam, Hyungsuk Lee, Junsang Lee, and Sam Ng
* Contact: walden0230@gmail.com

* This file analyzes using VECM (Vector Error Correction Model). Unit root analysis and Information criteria (AIC) were conducted separately. The required file is data.dta.

clear

use data

* 1) log of markup, log of real GDP, consumer credit–to–GDP, and log of real exchange rate; 
* We set the p(lag) = 5 based on information criteria in empirical specifications.

vec lnmarkup CCDL lREER_BIS lnRGDP, lags(5) rank(1) 

* 2) markup dispersion, log of real GDP, consumer credit–to–GDP, and log of real exchange rate
* We set the p(lag) = 4 based on information criteria in empirical specifications.

vec lnmarkup_d CCDL lREER_BIS lnRGDP, lags(4) rank(1) 

