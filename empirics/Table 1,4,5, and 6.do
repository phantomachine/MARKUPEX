* Title: On a Pecuniary Externality of Competitive Banking through Goods Pricing Dispersion
* Copyright: Timothy Kam, Hyungsuk Lee, Junsang Lee, and Sam Ng
* Contact: walden0230@gmail.com

* Note: This file conducts an empirical analysis on the relationship between Markup (its dispersion) and the consumer credit to GDP ratio using OLS (Ordinary Least Squares). Executing it generates Table 1, Table 4, Table 5, and Table 6. The required file is data.dta.

clear

use data

** Table 1: OLS results: Markup and dispersion **

eststo clear 

eststo: reg lnmarkup  CCDL cpi_py lnRGDP dtfp RWS lREER_BIS RC  , vce(robust)

eststo: reg lnmarkup_d CCDL cpi_py lnRGDP dtfp RWS lREER_BIS RC , vce(robust)

#delimit ;
esttab  using "Table1.tex", 
	   keep(CCDL cpi_py lnRGDP dtfp RWS lREER_BIS RC) 
	   replace compress b(a3) se(a3)  star(+ 0.10 * 0.05 ** 0.01 ) noconstant 
	   mgroups( "Dependent variable: \( log(Markup) \)" , pattern(1 0 0 0 0) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span}))  
	   obslast label
	   booktabs   substitute(\_ _) nonotes
	   scalars("r2 \(R^2\)") ;
#delimit cr

** Table 4: OLS results: Markup and Markup Dispersion: lag of dependent variable **

eststo clear 

eststo: reg lnmarkup  L(1/1).lnmarkup CCDL cpi_py lnRGDP dtfp RWS lREER_BIS RC , vce(robust)

eststo: reg lnmarkup_d L(1/1).lnmarkup_d CCDL cpi_py lnRGDP dtfp RWS lREER_BIS RC , vce(robust)

#delimit ;      
esttab  using "Table4.tex", 
	   keep(CCDL cpi_py lnRGDP dtfp RWS lREER_BIS RC) 
	   replace compress b(a3) se(a3)  star(+ 0.10 * 0.05 ** 0.01 ) noconstant 
	   mgroups( "Dependent variable: \( log(Markup) \)" , pattern(1 0 0 0 0) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span}))  
	   obslast label
	   booktabs   substitute(\_ _) nonotes
	   scalars("r2 \(R^2\)") ;
#delimit cr

** Table 5: OLS results: Markup and Markup Dispersion: Different measure **

eststo clear 

eststo: reg lnmarkup  CCD cpi_py lnRGDP dtfp RWS lREER_BIS RC, vce(robust)

eststo: reg lnmarkup_d CCD cpi_py lnRGDP dtfp RWS lREER_BIS RC, vce(robust)

eststo: reg lnmarkup  COLL cpi_py lnRGDP dtfp RWS lREER_BIS RC, vce(robust)

eststo: reg lnmarkup_d COLL cpi_py lnRGDP dtfp RWS lREER_BIS RC , vce(robust)

#delimit ;    
esttab  using "Table5.tex", 
	   keep(CCD COLL cpi_py lnRGDP dtfp RWS lREER_BIS RC) 
	   replace compress b(a3) se(a3)  star(+ 0.10 * 0.05 ** 0.01 ) noconstant 
	   mgroups( "Dependent variable: \( log(Markup) \)" , pattern(1 0 0 0 0) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span}))  
	   obslast label
	   booktabs   substitute(\_ _) nonotes
	   scalars("r2 \(R^2\)") ;
#delimit cr

** Table 6: OLS results: Markup and Markup Dispersion: Volcker Dummy **

eststo clear 

eststo: reg lnmarkup  CCDL cpi_py lnRGDP dtfp RWS lREER_BIS RC volcker , vce(robust)

eststo: reg lnmarkup_d CCDL cpi_py lnRGDP dtfp RWS lREER_BIS RC volcker, vce(robust)

#delimit ;    
esttab  using "Table6.tex", 
	   keep(CCDL CCDL cpi_py lnRGDP dtfp RWS lREER_BIS RC) 
	   replace compress b(a3) se(a3)  star(+ 0.10 * 0.05 ** 0.01 ) noconstant 
	   mgroups( "Dependent variable: \( log(Markup) \)" , pattern(1 0 0 0 0) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span}))  
	   obslast label
	   booktabs   substitute(\_ _) nonotes
	   scalars("r2 \(R^2\)") ;
#delimit cr


