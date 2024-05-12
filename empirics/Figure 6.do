* Title: On a Pecuniary Externality of Competitive Banking through Goods Pricing Dispersion
* Copyright: Timothy Kam, Hyungsuk Lee, Junsang Lee, and Sam Ng
* Contact: walden0230@gmail.com

*Note: This file generates Figure 6, which includes (a) Credit and average markup, (b) Credit and markup dispersion, and (c) Average markup and the effective Federal Funds rate.

clear

use data

gen year = floor(time)

collapse (mean) markup Markup_Dispersion CCDL cpi_py FFDR, by(year)

gen time = year

twoway (line markup time, yaxis(2) lc(red) ylabel(,format(%9.2f) axis(2))) ///
       (line CCDL time, lc(black) ylabel(,format(%9.1f) axis(1))) ///
       , ytitle(" ", size(medium) axis(1)) ytitle(" ", size(medium) axis(2)) ///
       title("" , size(large)) xtitle("Year", size(medium)) ///
       scheme(s1color) plotregion(style(none) margin(zero)) graphregion(margin(small)) ///
       legend(order (1 "Markup(left)" 2 "Consumer Credit to GDP (%, right)") region(lw(none) color(none)) ring(1) position(11) size(medium)) ///
       name(RPV_Bank_line, replace) ysize(3) xsize(5) 
graph export "Fig6a_Credit and average markup.png", replace

twoway (line Markup_Dispersion time, yaxis(2) lc(blue) ylabel(,format(%9.2f) axis(2))) ///
       (line CCDL time, lc(black) ylabel(,format(%9.1f) axis(1))) ///
       , ytitle(" ", size(medium) axis(1)) ytitle(" ", size(medium) axis(2)) ///
       title("" , size(large)) xtitle("Year", size(medium)) ///
       scheme(s1color) plotregion(style(none) margin(zero)) graphregion(margin(small)) ///
       legend(order (1 "Markup Dispersion(left)" 2 "Consumer Credit to GDP (%, right)") region(lw(none) color(none)) ring(1) position(11) size(medium)) ///
       name(RPV_Bank_line, replace) ysize(3) xsize(5) 
graph export "Fig6b_Credit and markups dispersion.png", replace

twoway line markup time,yaxis(2) ytitle(" ", size(medium) axis(1))  lc(red)||line FFDR time,  title("" ,  size(large))  xtitle("Year", size(medium)) ytitle(" ", size(medium) axis(2)) lc(green)  ///
scheme(s1color) plotregion(sty(none) margin(zero)) graphregion(margin(small)) ///
legend(order (1 "Markup(left)" 2 "FFER(%, right)" ) region(lw(none) color(none)) ring(1) position(11) size(medium)) name(Markup_Bank_line, replace) ysize(3) xsize(5) 
graph export "Fig6c_Average markup and the effective Federal Funds rate.png",replace

