# MARKUPEX
Source codes for the paper [*On a Pecuniary Externality of Competitive Banking through Goods Pricing Dispersion*](https://github.com/phantomachine/MARKUPEX/blob/main/main.pdf) by Timothy **Kam**, Hyungsuk **Lee**, Junsang **Lee**, Ieng Man (Sam) **Ng**.

There are two sets of programs. 

* Model calibration and analyses are done using ``Python``. See directory `model`.

* Empirical work is done in `STATA`. See directory `empirics`.

## Numerical example

Dependencies:

* `Python (> 3.7)`

* `STATA`

### Stationary monetary equilibrium 
* `sme_example.ipynb` 
* `sme_welfare_decompose.ipynb`
* This two files generate figures shown in the paper

### Calibration
* `calibration.ipynb`
* This file documents the calibration of the model parameter

## Empirical analysis (Deprecated - not included in the final paper)
This documentation covers the empirical analysis conducted for the study of the relationship between credit and markup (and its dispersion) in the United States from 1980 to 2007. The analysis utilizes `STATA` for generating tables and figures as per the findings discussed in our paper.

### Data description
- `data.dta`: Contains the dataset used for the empirical analysis spanning from 1980 to 2007 in the United States. The variables used and their sources are detailed in Appendix C "Data and Measurement" of our paper.

### Code structure
The empirical analysis is supported by four `STATA` `.do` files, each serving a unique purpose in the generation of tables and figures as outlined in the paper:

1. `Table 1, 4, 5, and 6.do`

    This script conducts the empirical analysis on the relationship between credit and markup (and its dispersion). It generates:
    
    * Table 1: OLS results: Markup and dispersion (Section 4.2 Empirical evidence)
    
    * Tables 4, 5, and 6 in Appendix D Robustness

    The tables are output in $\LaTeX$ format.

2. `Table 3.do`

    Generates Table 3: Data sources and summary statistics, providing an overview of the data utilized in our analysis.

3. `Figure 6.do`

    Produces Figure 6: Time series of credit-to-GDP ratio and markup statistics for Section 4.2 Empirical evidence. It creates three figures illustrating:
    
    * Credit and average markup

    * Credit and markups dispersion

    * Average markup and the effective Federal Funds rate

#### 4. `Appendix VECM_Equation_D2_D3.do`

Outputs the VECM results discussed in Appendix D Robustness, examining the relationships involving:
  
1. log of markup, log of real GDP, consumer credit-to-GDP, and log of real exchange rate

2. markup dispersion, log of real GDP, consumer credit-to-GDP, and log of real exchange rate

Focuses on Equations (D.2) and (D.3) in paper ``main.pdf``.

### Usage

To run any of the `.do` files, navigate to the directory containing the files and execute the following command in `STATA`:

# Sources

`Python` code classes and observational data.

## Numerical example

* `bj_bcw.py`: Baseline model primitives and equilibrium solver
    * A monetary model **with** banks 
    * Model setup combines and encompasses two important limits: 
    
      * [Berentsen, Camera and Waller (*J. Econ. Theory*, 2007)](https://doi.org/10.1016/j.jet.2006.03.016) 
      * [Head, Liu, Menzio and Wright (*J. Eur. Econ. Assoc.*, 2012)](https://www.jstor.org/stable/23251208)

* `calibration.ipynb`: Use for calibration  

* `hlmw.py`: Model primitives and equilibrium solver
    * A monetary model **without** banks
    * Model setup: [Head, Liu, Menzio and Wright (*J. Eur. Econ. Assoc.*, 2012)](https://www.jstor.org/stable/23251208)
    
## Data file

* `Calibration_data.xlsx`
    * For model calibration
* `data.dta`
    * For empirical analysis
