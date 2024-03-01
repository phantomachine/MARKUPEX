import numpy as np
import scipy as sp
from textwrap import dedent
from scipy.optimize import brentq, fsolve
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sys
import time
import collections as collections
from cycler import cycler

class baseline_mod(object):
    """Model primitives and methods"""
    def __init__(self, 
                 β = 0.9804, σ_CM = 1.0, σ_DM = 0.4225,  
                 Abar = 1.0, Ubar_CM = 1.0, Ubar_DM = 1.0, 
                 λ = 0.65,
                 c = 1.0, τ_b = 0.0, τ_max = 0.08525, τgrid_size = 50, 
                 z_min = 1e-10, z_max = 1.0, zgrid_size = 100, 
                 N_reimann = 3000, Tol=1e-10
                ):
        #Model parameters
        #-------------------------------             # Default parameter settings:
        self.β = β                                   # Discount factor
        
        """Preferences"""
        self.σ_CM = σ_CM                             # CRRA CM (market 2)   
        self.σ_DM = σ_DM                             # CRRA DM (market 1)
        self.Abar = Abar                             # Labor disutility scale -Abar*h  
        self.Ubar_CM = Ubar_CM                       # DM utility scale Ubar_DM*u(q)
        self.Ubar_DM = Ubar_DM                       # CM utility scale Ubar_CM*u(x)
        
        """Matching"""
        self.λ = λ
        self.α_0 = (1.0-λ)**2.0                      # prob. no firm contact 
        self.α_1 = 2.0*(1.0-λ)*λ                     # Prob. of 1 firm contact 
        self.α_2 = λ**2.0                            # Prob. of 2 firm contacts: residual
        
        
        x_star = self.invU_C_CM(Abar)                # Optimal CM consumption: from FoC 
        self.x_star = x_star                         # in CM: quasilinear preference         
        """Production"""
        self.c = c                                   # Real marginal cost of production
                
        """Transfer/tax/inflation rate"""
        self.τ_b = τ_b                               # Transfer/tax to active DM buyers       
        τ_min = -0.00965                             # data τ min
        self.τ_min = τ_min                           # Lower bound on inflation rate experiment
        self.τ_max = τ_max                           # Upper bound on inflation rate experiment
        
        #Approximation - state space settings
        self.z_min = z_min                           # Loewr bound on z
        self.z_max = z_max                           # Upper bound on z
                
        self.zgrid_size = zgrid_size                 # Grid size on array of z
        self.τgrid_size = τgrid_size                 # Grid size on array of τ
        
        #Array of grids on z
        self.z_grid = np.linspace(z_min, z_max, zgrid_size)      
        
        
        #Array of grids on τ
        self.τ_grid = np.linspace(τ_min, τ_max, τgrid_size) 
       
        # Reimann integral domain partition
        self.N_reimann = N_reimann
        
        self.Tol = Tol
        
    ##-----------Model Primitive Functions--------------------------------## 
    ##-----------CM preference-------------------##         
    def U_CM(self, x):
        """CM per-period utility"""
        if self.σ_CM == 1.0:
            u = np.log(x)
        else:
            u = ( x**( 1.0 - self.σ_CM ) ) / ( 1.0 - self.σ_CM )
        return self.Ubar_CM*u
    
    def mu_CM(self, x):
        """CM per-period marginal utility"""
        if self.σ_CM == 1.0:
            MU = self.Ubar_CM / x
        else: 
            MU = x**( -self.σ_CM )
        return MU
    
    def invU_C_CM(self, marginal_value):
        """Inverse of dU/dx function, Uprime of CM good"""
        return ( marginal_value/self.Ubar_CM )**( -1.0 / self.σ_CM )  
    
    def h(self, labor):
        """Linear CM labor supply function"""
        return self.Abar*labor
    
    def production_CM(self, labor):
        """Linear CM production function"""
        Y = labor
        return Y
    
    ##-----------DM preference------------------##  
    def u_DM(self, q):
        """DM per-period utility"""
        if self.σ_DM == 1.0:
            u = np.log(q)
        else:
            u = ( q**( 1.0 - self.σ_DM ) ) / ( 1.0 - self.σ_DM )
        return self.Ubar_DM*u
    
    def mu_DM(self, q):
        """DM marginal utility"""
        if self.σ_DM == 1.0:
            mu = 1.0 / q
        else:
            mu = q**( -self.σ_DM )
        return mu    
    
    def invu_q_DM(self, marginal_value):
        """Inverse of du/dq function, Uprime of DM good"""
        return ( ( marginal_value/self.Ubar_DM )**( -1.0 / self.σ_DM ) )
    
    def cost_DM(self, q):
        """DM production technology - linear cost of producing a unit of q"""
        cost = q
        return cost
    
    def invcost_DM(self, q):
        """Inverse of DM production function - output in DM q 
        associated with given level of cost"""
        q = cost
        return q        
    ##-------------------Gross inflation and CM transfer-------------##
    def GrossInflation(self, τ):
        """Gross inflation"""
        γ = 1.0 + τ
        return γ
    
    def τ_CM(self, τ):
        """Tax/transfer to CM households"""
        τ1 = (self.α_0+self.α_1+self.α_2)*self.τb 
        τ2 = τ - τ1
        return τ2    
        
    def i_policy(self, τ):
        """
        Nominal policy interest rate in steady state
        """
        γ = self.GrossInflation(τ)
        policy_rate = γ / self.β - 1.0
        return policy_rate    
    
    ##----------------DM goods, loans demand and cut-off functions----------------------##    
    def ρ_hat_func(self, z):
        """price that max profits with constrained buyer"""
        ρ_hat_value = z**( self.σ_DM / ( self.σ_DM - 1.0 ) ) 
        return ρ_hat_value
    
    def ρ_tilde_func(self, z, i):
        """cut-off price where constrained buyer will borrow"""
        ρ_tilde_value = self.ρ_hat_func(z) * ( (1.0 + i)**(1.0/(self.σ_DM-1.0)) ) 
        return ρ_tilde_value
       
    def q_demand_func(self, ρ, i, z):
        """DM goods demand function"""
        ρ_hat = self.ρ_hat_func(z)
        ρ_tilde = self.ρ_tilde_func(z, i)

        if ρ <= ρ_tilde:
            """liquidity constrained and borrowing unconstrained buyer"""
            q = ( ρ*(1.0 + i ) )**(-1.0/self.σ_DM)
            
        elif ρ_tilde < ρ < ρ_hat:
            """liquidity constrained and zero borrowing buyer"""
            q = z / ρ
        
        else:
            """liquidity unconstrained and zero borrowing buyer"""
            q = ρ**(-1.0/self.σ_DM)
            
        return q
    
    def ξ_demand_func(self, ρ, i, z):
        """DM loans demand function"""
        ρ_hat = self.ρ_hat_func(z)
        ρ_tilde = self.ρ_tilde_func(z, i)

        if ρ <= ρ_tilde:
            """liquidity constrained and borrowing unconstrained buyer"""
            loan = ρ * ( ( ρ*(1.0 + i ) )**(-1.0/self.σ_DM) ) - z
            
        elif ρ_tilde < ρ < ρ_hat:
            """liquidity constrained and zero borrowing buyer"""
            loan = 0.0
        
        else:
            """liquidity unconstrained and zero borrowing buyer"""
            loan = 0.0
 
        return loan
    
    def Total_loan_func(self, i, z, τ):
        """Aggregate loan"""
        ρ_grid = self.support_grid_func(z, i, τ)

        pdf_grid = self.dF_normalization_func(z, i, τ)

        em_func = lambda ρ: self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(ρ, i, z) ) 

        em = np.array( [em_func(ρ) for ρ in ρ_grid] )

        loan_expost = np.array( [self.ξ_demand_func(ρ, i, z) for ρ in ρ_grid] )

        integrand_values = em * loan_expost * pdf_grid        
        total_loan = np.trapz(integrand_values, ρ_grid)       

        return total_loan
    
    def i_rate_obj(self, i, z, τ):
        """Loan market clearing condition"""
        LHS = self.α_0 * z # Total deposit
        
        # Total loan
        ρ_grid = self.support_grid_func(z, i, τ)

        pdf_grid = self.dF_normalization_func(z, i, τ)

        em_func = lambda ρ: self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(ρ, i, z) ) 

        em = np.array( [em_func(ρ) for ρ in ρ_grid] )

        loan_expost = np.array( [self.ξ_demand_func(ρ, i, z) for ρ in ρ_grid] )

        integrand_values = em * loan_expost * pdf_grid        
        total_loan = np.trapz(integrand_values, ρ_grid)
        
        # Objective
        net = LHS - total_loan     
        return net
    
    ##-----------------DM goods expenditure and sellers' profit margin------------##
    def q_expenditure(self, ρ, i, z):
        """expenditure := ρq"""        
        ρ_hat = self.ρ_hat_func(z)
        ρ_tilde = self.ρ_tilde_func(z, i)

        if ρ <= ρ_tilde:
            """liquidity constrained and borrowing unconstrained buyer"""
            ρq = ρ * ( ( ρ*(1.0 + i ) )**(-1.0/self.σ_DM) )
            
        elif ρ_tilde < ρ < ρ_hat:
            """liquidity constrained and zero borrowing buyer"""
            ρq = z
        
        else:
            """liquidity unconstrained and zero borrowing buyer"""
            ρq = ρ * ( ρ**(-1.0/self.σ_DM) )
            
        return ρq
    
    def R_func(self, ρ, i, z):
        """seller's ex-post profit per customer served"""
        ρ_hat = self.ρ_hat_func(z)
        ρ_tilde = self.ρ_tilde_func(z, i)
        if ρ <= ρ_tilde:
            """profit margin from serving
            liquidity constrained and borrowing unconstrained buyer
            """
            qb = ( ρ*(1.0 + i ) )**(-1.0/self.σ_DM)
            val = qb * ( ρ - self.c ) 
        
        elif ρ_tilde < ρ < ρ_hat:
            """profit margin from serving 
            liquidity constrained and zero borrowing buyer
            """
            qb = z / ρ
            val = qb * ( ρ - self.c )
        
        else:
            """profit margin from serving 
            liquidity unconstrained and zero borrowing buyer
            """
            qb = ρ**(-1.0/self.σ_DM)
            val = qb * ( ρ - self.c )
            
        return val
    
    def dq_dρ_func(self, ρ, i, z):
        """dq/dρ"""
        ρ_hat = self.ρ_hat_func(z)
        ρ_tilde = self.ρ_tilde_func(z, i)
        
        if ρ <= ρ_tilde:
            term_a = ( -1.0 / self.σ_DM ) * ( ρ**( (-1.0/self.σ_DM) -1.0 ) )
            
            term_b = ( 1.0 + i )**( -1.0 / self.σ_DM )
            
            val = term_a * term_b
        
        elif ρ_tilde < ρ < ρ_hat:
            
            val = -z/(ρ**2.0)
        
        else:
            val = ( -1.0 / self.σ_DM ) * ( ρ**( (-1.0/self.σ_DM) -1.0 ) )
            
        return val    
    
    ##-----------------bounds on DM goods prices------------##
    def ρ_max_func(self, z):
        """Equilibrium upper bound on the support of F: ρ_{\max}"""
        
        ρ_hat = self.ρ_hat_func(z)
        ρ_constant = self.c / (1.0 - self.σ_DM)
        ρ_ub = np.max( [ ρ_hat, ρ_constant ] ) #monopoly price in general
        
        #Note: Case 1: ρ_ub = ρ_hat implies all liquidity constrained
        #Note: Case 2: ρ_ub = ρ_constant implies mixture of liquidity constrained and unconstrained
        
        return ρ_ub
    
    def ρ_min_func(self, z, i, τ):
        """Equal expected profit condition to back out the lower bound
        on the support of F given R(ρ_{\max})
        """ 
        
        ρ_max = self.ρ_max_func(z)
        
        noisy_search = self.α_1  / ( self.α_1 + 2.0 * self.α_2 ) 

        LHS = lambda ρ: self.q_demand_func(ρ, i, z) * ( ρ - self.c )

        RHS = noisy_search * self.R_func(ρ_max, i,  z)
        
        vals_obj = lambda ρ: RHS - LHS(ρ)
        
        ρ_lb = brentq(vals_obj, self.c, ρ_max)
       
        return ρ_lb
    
    ##-----------DM goods price distribution: CDF and pdf in terms of aggregate variables-------##
    def F_func(self, ρ, i, z):
        """Posted price distribution (CDF): F(ρ, i, z)
        Note: i := i(ρ, z)
        """
        ρ_max = self.ρ_max_func(z)

        noisy_search = self.α_1 / ( 2.0 * self.α_2 ) 
        
        R_max = self.R_func(ρ_max, i, z)
        
        R = self.R_func(ρ, i, z)
        
        CDF_value =  1.0 - noisy_search * ( ( R_max / R ) - 1.0 ) 
        
        return CDF_value    
    
    def dF_func(self, ρ, i, z):
        """Density of the price distribution (PDF): dF(ρ, i, z)/dρ"""
        ρ_max = self.ρ_max_func(z)

        noisy_search = self.α_1 / ( 2.0 * self.α_2 ) 
        
        R_max = self.R_func(ρ_max, i, z)
        
        R = self.R_func(ρ, i, z)
        
        dR_dρ = self.dq_dρ_func(ρ, i, z)*(ρ - self.c) + self.q_demand_func(ρ, i, z)
        
        pdf_value = noisy_search*( ( R_max / ( R**2.0 ) ) * dR_dρ )
        
        return pdf_value
    
    def support_grid_func(self, z, i, τ):
        a = self.ρ_min_func(z, i, τ)
        b = self.ρ_max_func(z)       
        supports = np.linspace(a, b, self.N_reimann)
        return supports
    
    def dF_normalization_func(self, z, i, τ):
        ρ_grid = self.support_grid_func(z, i, τ)
        dF = np.array( [ self.dF_func(ρ, i, z) for ρ in ρ_grid] )
        
        a = self.ρ_min_func(z, i, τ)
        b = self.ρ_max_func(z) 
        w = ρ_grid[1] - ρ_grid[0]
        dF_sum = np.sum(w*dF)
        dF_nor = dF/ dF_sum
        return dF_nor           
    
    ##-----------------Money demand-------------------------------##
    def money_euler_rhs(self, z, i, τ):
        """Money demand net expected benefit"""
        if τ > self.β - 1.0:
        
            ρ_hat = self.ρ_hat_func(z)

            ρ_grid = self.support_grid_func(z, i, τ)

            dF_grid = self.dF_normalization_func(z, i, τ)


            """depositor"""
            mu_depositor = self.α_0 * i

            """
            Buyer: can be liquidity constrained + borrowing or liquidity constrained + zero borrowing

            """

            em_func = lambda ρ: self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(ρ, i, z) )

            em = np.array( [em_func(ρ) for ρ in ρ_grid[ρ_hat>=ρ_grid] ] )

            LP_func = lambda ρ: ( self.mu_DM( self.q_demand_func(ρ, i, z) ) / ρ ) -1.0

            liquidity_premium = np.array( [LP_func(ρ) for ρ in ρ_grid[ρ_hat>=ρ_grid]] )
       
            mu_buyer = em * liquidity_premium * dF_grid[ρ_hat>=ρ_grid]

            value = mu_depositor + np.trapz(mu_buyer,  ρ_grid[ρ_hat>=ρ_grid])
        
        else:
            value = z**(-self.σ_DM) - 1.0
        
        return value
    
    def money_euler_obj(self, z, i, τ):
        """Money demand objective"""
        LHS = self.i_policy(τ)
        RHS = self.money_euler_rhs(z, i, τ)
        net = LHS - RHS
        return net
    
    def system_func(self, initial_guess, τ):
        """System of two equations for z (money demand) and i (loan market clearing)"""
        z = initial_guess[0]
        i = initial_guess[1]
        
        z_obj = self.money_euler_obj(z, i, τ)
        i_obj = self.i_rate_obj(i, z, τ)
        
        return [z_obj, i_obj]
    
    def solve_z_i(self, z_guess, i_guess, τ):
        """Solution for z* and i* """
        
        x0 = [z_guess, i_guess]
        
        x = fsolve(self.system_func, x0, xtol=1e-5, args=(τ), full_output=False)

        z = x[0] # solution
        i = x[1] # solution
        
        if i < self.i_policy(τ):
            i = 0.0
        else:
            i = i
        return z, i                        
    

    ##------------------SME and Stat---------------------------------##
    def Total_q_func(self, z, i, τ):
        """Total DM goods demand"""
        ρ_grid = self.support_grid_func(z, i, τ)
        a = self.ρ_min_func(z, i, τ)
        b = self.ρ_max_func(z) 
        w = (b-a)/ self.N_reimann
        pdf_grid = self.dF_normalization_func(z, i, τ)
        
        em_func = lambda ρ: self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(ρ, i, z) ) 
            
        em = np.array( [em_func(ρ) for ρ in ρ_grid] )
        
        q_expost = np.array( [self.q_demand_func(ρ, i, z) for ρ in ρ_grid] )
        
        integrand_values = em * q_expost * pdf_grid
        
        total_q = np.trapz(integrand_values, ρ_grid)
        
        return total_q
    
    def Total_ρq_func(self, z, i, τ):
        """Total DM goods expenditure"""
        ρ_grid = self.support_grid_func(z, i, τ)
     
        pdf_grid = self.dF_normalization_func(z, i, τ)
        
        em_func = lambda ρ: self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(ρ, i, z) ) 
            
        em = np.array( [em_func(ρ) for ρ in ρ_grid] )
        
        ρq_expost = np.array( [self.q_expenditure(ρ, i, z) for ρ in ρ_grid] )
        
        integrand_values = em * ρq_expost * pdf_grid
        
        total_ρq = np.trapz(integrand_values, ρ_grid)
        
        return total_ρq    
    
    def firm_profit_func(self, z, i, τ):
        """Firms expected transacted profit"""
        ρ_grid = self.support_grid_func(z, i, τ)
    
        pdf_grid = self.dF_normalization_func(z, i, τ)
        
        em_func = lambda ρ: self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(ρ, i, z) ) 
            
        em = np.array( [em_func(ρ) for ρ in ρ_grid] )
        
        profit_margin = np.array([self.R_func(ρ, i, z ) for ρ in ρ_grid])
        
        integrand_values = em * profit_margin * pdf_grid
        
        firm_profit = np.trapz(integrand_values, ρ_grid)        
        
        return firm_profit
    
    def markup_func(self, z, i, τ):
        """Consumption weighted transactred markup"""  
        ρ_grid = self.support_grid_func(z, i, τ)
      
        pdf_grid = self.dF_normalization_func(z, i, τ)
        
        em_func = lambda ρ: self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(ρ, i, z) ) 
            
        em = np.array( [em_func(ρ) for ρ in ρ_grid] )
        
        expost_markup_func = lambda ρ: ρ / self.c
        
        markup_expost = np.array([expost_markup_func(ρ) for ρ in ρ_grid])
        
        # DM consumption share
        q_share = self.Total_q_func(z, i, τ) / (self.Total_q_func(z, i, τ) + self.x_star)
        
        # CM consumption share
        x_share = self.x_star / (self.Total_q_func(z, i, τ) + self.x_star)
        
        # normalized
        nor_share = q_share / x_share
        
        # dm_consumption_share * dm_markup + cm_consumption_share * cm_markup(=1, CM p=mc) 
        # normalized by x_share to get consumption weighted markup
        markup = np.trapz(nor_share * (markup_expost * pdf_grid), ρ_grid) + 1.0
        
        return markup
    
    def DM_utility(self, z, i, τ):
        """
        DM utility 
        """        
        # those who matches with a lending bnak
        ρ_grid = self.support_grid_func(z, i, τ)
       
        pdf_grid = self.dF_normalization_func(z, i, τ)
        
        em_func = lambda ρ: self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(ρ, i, z) ) 
            
        em = np.array( [em_func(ρ) for ρ in ρ_grid] )
        
        dm_net_func = lambda ρ: self.u_DM(self.q_demand_func(ρ,i,z)) - self.cost_DM(self.q_demand_func(ρ,i,z)) 
        
        expost_utility_α1_α2 = np.array( [ dm_net_func(ρ) for ρ in ρ_grid] )
        
        utility = np.trapz(em * expost_utility_α1_α2 * pdf_grid, ρ_grid ) 
                    
        return utility    
    
    def DM_utility_delta(self, z, i, τ, delta):
        """
        DM utility change by delta
        """        
        ρ_grid = self.support_grid_func(z, i, τ)
        
        pdf_grid = self.dF_normalization_func(z, i, τ)
        
        em_func = lambda ρ: self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(ρ, i, z) ) 
            
        em = np.array( [em_func(ρ) for ρ in ρ_grid] )
        
        dm_net_func = lambda ρ: self.u_DM(self.q_demand_func(ρ,i,z)*delta) - self.cost_DM(self.q_demand_func(ρ,i,z))  
        
        expost_utility_α1_α2 = np.array( [ dm_net_func(ρ) for ρ in ρ_grid] )
        
        utility = np.trapz( em * expost_utility_α1_α2 * pdf_grid, ρ_grid )
                    
        return utility      
    
    
    def welfare_func(self, z, i, τ):
        """Total welfare"""
        discount = ( 1.0 / ( 1.0 - self.β ) )
        if τ == self.β - 1.0:
            
            #allocation under Friedman rule
            qb_FR = z

            DM_segment = (self.α_1 + self.α_2) * ( self.u_DM(qb_FR)- self.cost_DM(qb_FR) ) 
            
            labor_CM = self.x_star 
            
            CM_segment = self.U_CM(self.x_star) - self.h(labor_CM)
            
            lifetime_utility = discount*(DM_segment + CM_segment )
        
        else:
            
            DM_segment = self.DM_utility(z, i, τ) 
            
            labor_CM = self.x_star 
            
            CM_segment = self.U_CM(self.x_star) -  self.h(labor_CM) 
            
            lifetime_utility = discount*(DM_segment + CM_segment )
            
        return lifetime_utility  
    
    def welfare_func_delta(self, z, i, τ, delta):
        """Economy with perfectly competitive banks
        for calculating CEV
        """
        discount = ( 1.0 / ( 1.0 - self.β ) )
        if τ == self.β - 1.0:
            
            #allocation under Friedman rule
            qb_FR = z

            DM_segment = (self.α_1 + self.α_2) * ( self.u_DM(qb_FR*delta) - self.cost_DM(qb_FR) )
            
            labor_CM = self.x_star 
            
            CM_segment = self.U_CM(self.x_star*delta) - self.h(labor_CM)
            
            lifetime_utility = discount*(DM_segment + CM_segment )
        
        else:
            
            DM_segment = self.DM_utility_delta(z, i, τ, delta) 
            
            labor_CM = self.x_star 
            
            CM_segment = self.U_CM(self.x_star*delta) -  self.h(labor_CM) 
            
            lifetime_utility = discount*(DM_segment + CM_segment )
            
        return lifetime_utility      
    
##---Begin Special case: BCW optimal quantity derived from their Money demand equation------##
    def q_BCW(self, τ):
        power = ( -1.0 / self.σ_DM )
        i_d = self.i_policy(τ)
        q =  (i_d + 1.0)**(power) 
        return q
    
    def loan_gdp_bcw_func(self, τ):
        ρ = 1.0
        i = self.i_policy(τ)
        total_q_exp = ρ * (self.α_1 + self.α_2) * self.q_BCW(τ)
        z_bcw = total_q_exp
        loan = (self.α_1 + self.α_2) * self.ξ_demand_func(ρ, i, z_bcw)
        loan_gdp = loan / (total_q_exp + self.x_star)
        return loan_gdp    
    
    def welfare_bcw_func(self, τ):
        """Economy with perfectly competitive banks and competitive pricing in goods market"""
        discount = 1.0 / ( 1.0 - self.β ) 
        
        labor_CM = self.x_star 
        
        CM_segment = self.U_CM(self.x_star) - self.h(labor_CM)
        
        DM_segment = (self.α_1 + self.α_2)*( self.u_DM(self.q_BCW(τ)) - self.cost_DM(self.q_BCW(τ)) )
        
        lifetime_utility = discount*(DM_segment + CM_segment )
        
        return lifetime_utility   
    
    def welfare_bcw_func_delta(self, τ, delta):
        """Economy with perfectly competitive banks and competitive pricing in goods market
        for calculating CEV
        """
        discount = 1.0 / ( 1.0 - self.β ) 
        
        labor_CM = self.x_star 
        
        CM_segment = self.U_CM(self.x_star*delta) - self.h(labor_CM)
        
        DM_segment = (self.α_1+self.α_2)*( self.u_DM(self.q_BCW(τ)*delta) - self.cost_DM(self.q_BCW(τ)) )
        
        lifetime_utility = discount*(DM_segment + CM_segment )
        
        return lifetime_utility     
                                                 
    def SME_stat_func(self, z, i, τ):
        ρ_grid = self.support_grid_func(z, i, τ)
     
        pdf_normalized_grid = self.dF_normalization_func(z, i, τ)
        
        em_func = lambda ρ: self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(ρ, i, z) ) 
            
        em = np.array( [em_func(ρ) for ρ in ρ_grid] )   
        
        #calculate (posted) price dispersion
        price_mean = np.trapz(ρ_grid * pdf_normalized_grid, ρ_grid)
        y_1 = pdf_normalized_grid * ( (ρ_grid  - price_mean)**2.0 )
        price_sd = np.sqrt( np.trapz(y_1, ρ_grid) )
        price_cv = price_sd / price_mean    
        
        ## calculate aggregate markup
        Y = self.x_star + self.Total_q_func(z, i, τ)
        
        DM_share = self.Total_q_func(z, i, τ) / Y
        
        CM_share = self.x_star / Y
        
        nor_share = DM_share / CM_share
        
        markup_deviation = ρ_grid / self.c  
        
        markup_mean = self.markup_func(z, i, τ) 
        
        y_2 =  pdf_normalized_grid * ( (markup_deviation - markup_mean)**2.0 ) * (nor_share**2.0)  + 0.0 # CM_dispersion = 0
        
        markup_sd = np.sqrt( np.trapz(y_2, ρ_grid) ) 
        
        markup_cv = markup_sd / markup_mean

        mpy = z / Y
        
        loan_gdp = self.Total_loan_func(i, z, τ) / Y
        
        stat_result = {  'price_mean': price_mean,
                         'price_sd': price_sd,
                         'price_cv': price_cv,
                         'markup_mean': markup_mean,
                         'markup_sd': markup_sd,
                         'markup_cv': markup_cv,     
                         'mpy': mpy,
                         'loan_gdp': loan_gdp
                    }
        
        return stat_result
    
    def SME_stat(self, z_guess, i_guess):  
        """SME solver for various inflation rate"""
        tic = time.time()
        
        zstar = np.zeros(self.τ_grid.size)
        istar = zstar.copy()
        
        ξstar = zstar.copy()
        Dstar = zstar.copy()
        qstar = zstar.copy()
        
        π_firm_star = zstar.copy()
        
        DM_surplus = zstar.copy()

        price_mean = zstar.copy()
        price_sd = zstar.copy()
        price_cv = zstar.copy()
                
        markup_mean = zstar.copy()
        markup_sd = zstar.copy()
        markup_cv = zstar.copy()

        mpy_star = zstar.copy()
        
        w_star = zstar.copy()
        w_bcw = zstar.copy()
        
        FFR = zstar.copy()
        
        credit_gdp = zstar.copy()
        credit_gdp_bcw = zstar.copy()  
        
        for idx_τ, τ in enumerate(self.τ_grid):

            zstar[idx_τ], istar[idx_τ] = self.solve_z_i(z_guess, i_guess, τ)
            
            ξstar[idx_τ] = self.Total_loan_func(istar[idx_τ], zstar[idx_τ], τ)
            
            Dstar[idx_τ] = self.α_0 * zstar[idx_τ]
            
            qstar[idx_τ] = self.Total_q_func(zstar[idx_τ], istar[idx_τ], τ)
            
            π_firm_star[idx_τ] = self.firm_profit_func(zstar[idx_τ], istar[idx_τ], τ)
            
            DM_surplus[idx_τ] = self.DM_utility(zstar[idx_τ], istar[idx_τ], τ)
                
            w_star[idx_τ] = self.welfare_func(zstar[idx_τ], istar[idx_τ], τ)
            
            w_bcw[idx_τ] = self.welfare_bcw_func(τ)
            
            price_mean[idx_τ] = self.SME_stat_func(zstar[idx_τ], istar[idx_τ], τ)['price_mean']
            
            price_sd[idx_τ] = self.SME_stat_func(zstar[idx_τ], istar[idx_τ], τ)['price_sd']
            
            price_cv[idx_τ] = self.SME_stat_func(zstar[idx_τ], istar[idx_τ], τ)['price_cv']
            
            markup_mean[idx_τ] = self.SME_stat_func(zstar[idx_τ], istar[idx_τ], τ)['markup_mean']
            
            markup_sd[idx_τ] = self.SME_stat_func(zstar[idx_τ], istar[idx_τ], τ)['markup_sd']
            
            markup_cv[idx_τ] = self.SME_stat_func(zstar[idx_τ], istar[idx_τ], τ)['markup_cv']
       
            mpy_star[idx_τ] = self.SME_stat_func(zstar[idx_τ], istar[idx_τ], τ)['mpy']
            
            credit_gdp[idx_τ] = self.SME_stat_func(zstar[idx_τ], istar[idx_τ], τ)['loan_gdp']
            
            credit_gdp_bcw[idx_τ] = self.loan_gdp_bcw_func(τ)
            
            FFR[idx_τ] = self.i_policy(τ)
            
        allocation_grid = {'zstar': zstar,
                           'istar': istar,
                           'ξstar': ξstar,
                           'Dstar': Dstar,
                           'qstar': qstar,
                           'π_firm_star': π_firm_star,
                           'DM_surplus': DM_surplus,
                           'w_star': w_star,
                           'w_bcw': w_bcw
                          }
        
        stat_grid = {'price_mean': price_mean,
                     'price_sd': price_sd,
                     'price_cv': price_cv,
                     'markup_mean': markup_mean,
                     'markup_sd': markup_sd,
                     'markup_cv': markup_cv,                   
                     'mpy_star': mpy_star,
                     'FFR': FFR,
                     'credit_gdp': credit_gdp,
                     'credit_gdp_bcw': credit_gdp_bcw                     
                    }
        
        result = {'allocation_grid': allocation_grid,
                  'stat_grid': stat_grid
                 }
        
        toc = time.time() - tic
        print("Elapsed time of solving SME:", toc, "seconds")
        
        return result                   