* Title: On a Pecuniary Externality of Competitive Banking through Goods Pricing Dispersion
* Copyright: Timothy Kam, Hyungsuk Lee, Junsang Lee, and Sam Ng
* Contact: walden0230@gmail.com

* Note: This file generates summary statistics and produces Table 3. The required file is data.dta.

clear
cd "C:\Users\User\Dropbox\Hyoungsuk_Junsang\Paper\A.BJ+BCW\Data\E. US_Data"

use data

gen lnRGDP_d = d.lnRGDP

estpost summarize markup Markup_Dispersion CCDL cpi_py lnRGDP_d dtfp RWS BIS_REER RC, detail  

matrix S= [e(count)\e(mean)\e(p50)\e(sd)]'
matrix list S
clear

* Store in new dataset
svmat2 S, rnames(variable)
rename S1 Nobs
rename S2 Mean
rename S3 Median
rename S4 SD 
order variable Nobs Mean Median SD 
tempfile d1_sumstats
save    `d1_sumstats'

export excel using "D:\Dropbox\Hyoungsuk_Junsang\Paper\A.BJ+BCW\Table\Table1_Data_sources_and_Coverage.xlsx" , firstrow(variables) replace