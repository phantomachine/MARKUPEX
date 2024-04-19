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

class hlmw_mod(object):
    """Head-Liu-Menzio-Wright (2012): Model primitives and methods"""
    def __init__(self, 
                 β = 0.9804, σ_CM = 1.0, σ_DM = 0.4225, 
                 Abar = 1.0, Ubar_CM = 1.0, Ubar_DM = 1.0, 
                 λ = 0.65,  
                 c = 1.0, τ_b = 0.0, τ_max = 0.1, τgrid_size = 200,
                 z_min = 1e-10, z_max = 1.0, zgrid_size = 120, 
                 N_reimann = 10000, Tol=1e-5, N_local = 20,
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
        τ_min = β-1.0                                # FR
        self.τ_min = τ_min                           # Lower bound on inflation rate experiment
        self.τ_max = τ_max                           # Upper bound on inflation rate experiment
        
        #Approximation - state space settings
        self.z_min = z_min                           # Loewr bound on z
        self.z_max = z_max                           # Upper bound on z
                
        self.zgrid_size = zgrid_size                 # Grid size on array of z
        self.τgrid_size = τgrid_size                 # Grid size on array of τ
        
        #Array of grids on z - add more points on left side
        self.z_grid = np.linspace(z_min, z_max, zgrid_size)   
        
        #Array of grids on τ
        #self.τ_grid = np.arange(β-1.0, 0.08+0.0025, 0.0025) #np.linspace(τ_min, τ_max, τgrid_size) 
        self.τ_grid = np.linspace(τ_min, τ_max, τgrid_size) 
        # τ_grid = np.linspace(τ_min, τ_max, τgrid_size - N_local)    
        # τ_l = 1.001*β - 1.0
        # τ_u = 1.01*β - 1.0
        # τ_local = np.linspace(τ_l, τ_u, N_local)
        # self.τ_grid = np.sort(np.append(τ_grid, τ_local))
        
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
        policy_rate = ( γ / self.β ) - 1.0
        return policy_rate    
    
    ##----------------DM goods, loans demand and cut-off functions----------------------##    
    def ρ_hat_func(self, z):
        """price that max profits with constrained buyer"""
        ρ_hat_value = z**( self.σ_DM / ( self.σ_DM - 1.0 ) ) 
        return ρ_hat_value
       
    def q_demand_func(self, ρ, z):
        """DM goods demand function"""
        ρ_hat = self.ρ_hat_func(z)
        
        if ρ < ρ_hat:
            """liquidity constrained"""
            q = z / ρ
        
        else:
            """liquidity unconstrained"""
            q = ρ**(-1.0/self.σ_DM)
            
        return q
    
    ##-----------------DM goods expenditure and sellers' profit margin------------##
    def q_expenditure(self, ρ, z):
        """expenditure := ρq"""        
        ρ_hat = self.ρ_hat_func(z)
 
        if ρ < ρ_hat:
            """liquidity constrained"""
            ρq = z
        
        else:
            """liquidity unconstrained"""
            ρq = ρ * ( ρ**(-1.0/self.σ_DM) )
            
        return ρq
    
    def R_func(self, ρ, z):
        """seller's ex-post profit per customer served"""
        ρ_hat = self.ρ_hat_func(z)
        
        if ρ < ρ_hat:
            """profit margin from serving 
            liquidity constrained buyer
            """
            qb = z / ρ
            val = qb * ( ρ - self.c )
        
        else:
            """profit margin from serving 
            liquidity unconstrained buyer
            """
            qb = ρ**(-1.0/self.σ_DM)
            val = qb * ( ρ - self.c )
            
        return val
    
    def dq_dρ_func(self, ρ, z):
        """dq/dρ"""
        ρ_hat = self.ρ_hat_func(z)
        
        if ρ < ρ_hat:
            
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
    
    def ρ_min_func(self, z, τ):
        """Equal expected profit condition to back out the lower bound
        on the support of F given R(ρ_{\max})
        """ 
        
        ρ_max = self.ρ_max_func(z)
        
        noisy_search = self.α_1  / ( self.α_1 + 2.0 * self.α_2 ) 

        LHS = lambda ρ: self.q_demand_func(ρ, z) * ( ρ - self.c )

        RHS = noisy_search * self.R_func(ρ_max, z)
        
        vals_obj = lambda ρ: RHS - LHS(ρ)
        
        ρ_lb = brentq(vals_obj, self.c, ρ_max)
       
        return ρ_lb
    
    ##-----------DM goods price distribution: CDF and pdf in terms of aggregate variables-------##
    def F_func(self, ρ, z):
        """Posted price distribution (CDF): F(ρ, z)
        """
        ρ_max = self.ρ_max_func(z)

        noisy_search = self.α_1 / ( 2.0 * self.α_2 ) 
        
        R_max = self.R_func(ρ_max, z)
        
        R = self.R_func(ρ, z)
        
        CDF_value =  1.0 - noisy_search * ( ( R_max / R ) - 1.0 ) 
        
        return CDF_value    
    
    def dF_func(self, ρ, z):
        """Density of the price distribution (PDF): dF(ρ, z)/dρ"""
        ρ_max = self.ρ_max_func(z)

        noisy_search = self.α_1 / ( 2.0 * self.α_2 ) 
        
        R_max = self.R_func(ρ_max, z)
        
        R = self.R_func(ρ, z)
        
        dR_dρ = self.dq_dρ_func(ρ, z)*(ρ - self.c) + self.q_demand_func(ρ, z)
        
        pdf_value = noisy_search*( ( R_max / ( R**2.0 ) ) * dR_dρ )
        
        return pdf_value
    
    def support_grid_func(self, z, τ):
        a = self.ρ_min_func(z, τ)
        b = self.ρ_max_func(z)       
        supports = np.linspace(a, b, self.N_reimann)
        return supports
    
    def dF_normalization_func(self, z, τ):
        ρ_grid = self.support_grid_func(z, τ)
        dF = np.array( [ self.dF_func(ρ, z) for ρ in ρ_grid] )
        w = ρ_grid[1] - ρ_grid[0]
        dF_sum = np.sum(w*dF)
        dF_nor = dF/ dF_sum
        return dF_nor          
    
    ##-----------------Money demand-------------------------------##
    def money_euler_rhs(self, z, τ):
        """Money demand net expected benefit"""
        if τ > self.β - 1.0:
        
            ρ_hat = self.ρ_hat_func(z)

            ρ_grid = self.support_grid_func(z, τ)

            dF_grid = self.dF_normalization_func(z, τ)

            """
            Buyer: can be liquidity constrained + borrowing or liquidity constrained + zero borrowing

            """

            em_func = lambda ρ: self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(ρ, z) )

            em = np.array( [em_func(ρ) for ρ in ρ_grid[ρ_hat>=ρ_grid] ] )

            LP_func = lambda ρ: ( self.mu_DM( self.q_demand_func(ρ, z) ) / ρ ) -1.0

            liquidity_premium = np.array( [LP_func(ρ) for ρ in ρ_grid[ρ_hat>=ρ_grid]] )

            mu_buyer = (em * liquidity_premium * dF_grid[ρ_hat>=ρ_grid])

            value = np.trapz(mu_buyer,  ρ_grid[ρ_hat>=ρ_grid])
        
        else:
            value = z**(-self.σ_DM) - 1.0
        
        return value
    
    def money_euler_obj(self, z, τ):
        """Money demand objective"""
        LHS = self.i_policy(τ)
        RHS = self.money_euler_rhs(z, τ)
        net = LHS - RHS
        return net
    
    def z_solver(self, τ):
        z_sol = brentq(self.money_euler_obj, self.z_min, self.z_max, args=(τ))
        return z_sol

    ##------------------SME and Stat---------------------------------##
    def Total_q_func(self, z, τ):
        """Total DM goods"""
        ρ_grid = self.support_grid_func(z, τ)
        
        pdf_grid = self.dF_normalization_func(z, τ)
        
        em_func = lambda ρ: self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(ρ, z) ) 
            
        em = np.array( [em_func(ρ) for ρ in ρ_grid] )
        
        q_expost = np.array( [self.q_demand_func(ρ, z) for ρ in ρ_grid] )
        
        integrand_values = em * q_expost * pdf_grid
        
        total_q = np.trapz(integrand_values, ρ_grid)
        
        return total_q
    
    def Total_ρq_func(self, z, τ):
        """Total DM expenditure"""        
        ρ_grid = self.support_grid_func(z, τ)
        
        pdf_grid = self.dF_normalization_func(z, τ)
        
        em_func = lambda ρ: self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(ρ, z) ) 
            
        em = np.array( [em_func(ρ) for ρ in ρ_grid] )
        
        ρq_expost = np.array( [self.q_expenditure(ρ, z) for ρ in ρ_grid] )
        
        integrand_values = em * ρq_expost * pdf_grid
        
        total_ρq = np.trapz(integrand_values, ρ_grid)
        
        return total_ρq    
    
    def firm_profit_func(self, z, τ):
        """Firms' expected transacted profit"""        
        ρ_grid = self.support_grid_func(z, τ)
        
        pdf_grid = self.dF_normalization_func(z, τ)
        
        em_func = lambda ρ: self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(ρ, z) ) 
            
        em = np.array( [em_func(ρ) for ρ in ρ_grid] )
        
        profit_margin = np.array([self.R_func(ρ, z ) for ρ in ρ_grid])
        
        integrand_values = em * profit_margin * pdf_grid
        
        firm_profit = np.trapz(integrand_values, ρ_grid) 
        
        return firm_profit
    
    def markup_func(self, z, τ):
        """Consumption weighted transactred markup"""        
        ρ_grid = self.support_grid_func(z, τ)
        
        pdf_grid = self.dF_normalization_func(z, τ)
        
        em_func = lambda ρ: self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(ρ, z) ) 
            
        em = np.array( [em_func(ρ) for ρ in ρ_grid] )
        
        expost_markup_func = lambda ρ: ρ / self.c
        
        markup_expost = np.array([expost_markup_func(ρ) for ρ in ρ_grid])
        
        # DM consumption share
        q_share = self.Total_q_func(z, τ) / (self.Total_ρq_func(z, τ) + self.x_star)
        
        # CM consumption share
        x_share = self.x_star / (self.Total_ρq_func(z, τ) + self.x_star)
        
        # normalized
        nor_share = q_share / x_share
        
        # dm_consumption_share * dm_markup + cm_consumption_share * cm_markup(=1, CM p=mc) 
        # normalized by x_share to get consumption weighted markup
        markup = np.trapz(nor_share * (markup_expost * pdf_grid), ρ_grid) + 1.0
        
        return markup
    
    def DM_utility(self, z, τ):
        """
        DM utility 
        """        
        ρ_grid = self.support_grid_func(z, τ)
        
        pdf_grid = self.dF_normalization_func(z, τ)
        
        em_func = lambda ρ: self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(ρ, z) ) 
            
        em = np.array( [em_func(ρ) for ρ in ρ_grid] )
        
        dm_net_func = lambda ρ: self.u_DM(self.q_demand_func(ρ,z)) - self.cost_DM(self.q_demand_func(ρ,z))
        
        expost_utility_α1_α2 = np.array( [ dm_net_func(ρ) for ρ in ρ_grid] )
        
        utility = np.trapz(em * expost_utility_α1_α2 * pdf_grid, ρ_grid ) 
                    
        return utility    
    
    def DM_utility_delta(self, z, τ, delta):
        """
        DM utility change by delta
        """        
        ρ_grid = self.support_grid_func(z, τ)
        
        pdf_grid = self.dF_normalization_func(z, τ)
        
        em_func = lambda ρ: self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(ρ, z) ) 
            
        em = np.array( [em_func(ρ) for ρ in ρ_grid] )
        
        dm_net_func = lambda ρ: self.u_DM(self.q_demand_func(ρ,z)*delta) - self.cost_DM(self.q_demand_func(ρ,z))
        
        expost_utility_α1_α2 = np.array( [ dm_net_func(ρ) for ρ in ρ_grid] )
        
        utility = np.trapz( em * expost_utility_α1_α2 * pdf_grid, ρ_grid )
                    
        return utility      
    
    def welfare_func(self, z, τ):
        """Total welfare"""
        discount = ( 1.0 / ( 1.0 - self.β ) )
        if τ == self.β - 1.0:
            
            #allocation under Friedman rule
            qb_FR = z

            DM_segment = (self.α_1 + self.α_2) * ( self.u_DM(qb_FR) - self.cost_DM(qb_FR) )
            
            labor_CM = self.x_star
            
            CM_segment = self.U_CM(self.x_star) - self.h(labor_CM)
            
            lifetime_utility = discount*(DM_segment + CM_segment )
        
        else:
            
            DM_segment = self.DM_utility(z, τ) 
            
            labor_CM = self.x_star 
            
            CM_segment = self.U_CM(self.x_star) -  self.h(labor_CM) 
            
            lifetime_utility = discount*(DM_segment + CM_segment )
            
        return lifetime_utility        
    
    def welfare_func_delta(self, z, τ, delta):
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
            
            DM_segment = self.DM_utility_delta(z, τ, delta) 
            
            labor_CM = self.x_star 
            
            CM_segment = self.U_CM(self.x_star*delta) -  self.h(labor_CM) 
            
            lifetime_utility = discount*(DM_segment + CM_segment )
            
        return lifetime_utility      
    
    def SME_stat_func(self, z, τ):
        ρ_grid = self.support_grid_func(z, τ)
        
        pdf_normalized_grid = self.dF_normalization_func(z, τ)
        
        em_func = lambda ρ: self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(ρ, z) ) 
            
        em = np.array( [em_func(ρ) for ρ in ρ_grid] )   
        
        #calculate (posted) price dispersion
        price_mean = np.trapz(ρ_grid * pdf_normalized_grid, ρ_grid)
        y_1 = pdf_normalized_grid * ( (ρ_grid  - price_mean)**2.0 )
        price_sd = np.sqrt( np.trapz(y_1, ρ_grid) )
        price_cv = price_sd / price_mean    
        
        ## calculate aggregate markup
        Y = self.x_star + self.Total_q_func(z, τ)
        
        DM_share = self.Total_q_func(z, τ) / Y
        
        CM_share = self.x_star / Y
        
        nor_share = DM_share / CM_share
        
        markup_deviation = ρ_grid / self.c  
        
        markup_mean = self.markup_func(z, τ) 
        
        y_2 =  pdf_normalized_grid * ( (markup_deviation - markup_mean)**2.0 ) * (nor_share**2.0)  + 0.0 # CM_dispersion = 0
        
        markup_sd = np.sqrt( np.trapz(y_2, ρ_grid) ) 
        
        markup_cv = markup_sd / markup_mean

        mpy = z / Y
        
        stat_result = {  'price_mean': price_mean,
                         'price_sd': price_sd,
                         'price_cv': price_cv,
                         'markup_mean': markup_mean,
                         'markup_sd': markup_sd,
                         'markup_cv': markup_cv,     
                         'mpy': mpy,
                    }
        
        return stat_result
    
    def SME_stat(self, ):    
        """SME solver for various inflation rate"""
        tic = time.time()
        
        zstar = np.zeros(self.τ_grid.size)

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
                
        for idx_τ, τ in enumerate(self.τ_grid):

            zstar[idx_τ] = self.z_solver(τ)
            
            qstar[idx_τ] = self.Total_q_func(zstar[idx_τ], τ)
            
            π_firm_star[idx_τ] = self.firm_profit_func(zstar[idx_τ], τ)
            DM_surplus[idx_τ] = self.DM_utility(zstar[idx_τ], τ)
            
            price_mean[idx_τ] = self.SME_stat_func(zstar[idx_τ], τ)['price_mean']
            price_sd[idx_τ] = self.SME_stat_func(zstar[idx_τ], τ)['price_sd']
            price_cv[idx_τ] = self.SME_stat_func(zstar[idx_τ], τ)['price_cv']
            
            markup_mean[idx_τ] = self.SME_stat_func(zstar[idx_τ], τ)['markup_mean']
            markup_sd[idx_τ] = self.SME_stat_func(zstar[idx_τ], τ)['markup_sd']
            markup_cv[idx_τ] = self.SME_stat_func(zstar[idx_τ], τ)['markup_cv']
       
            mpy_star[idx_τ] = self.SME_stat_func(zstar[idx_τ], τ)['mpy']
        
            w_star[idx_τ] = self.welfare_func(zstar[idx_τ], τ)
                    
        allocation_grid = {'zstar': zstar,
                           'qstar': qstar,
                           'π_firm_star': π_firm_star,
                           'DM_surplus': DM_surplus,
                           'w_star': w_star
                          }
        
        stat_grid = {'price_mean': price_mean,
                     'price_sd': price_sd,
                     'price_cv': price_cv,
                     'markup_mean': markup_mean,
                     'markup_sd': markup_sd,
                     'markup_cv': markup_cv,                   
                     'mpy_star': mpy_star,
                    }
        
        result = {'allocation_grid': allocation_grid,
                  'stat_grid': stat_grid
                 }
        
        toc = time.time() - tic
        print("Elapsed time of solving SME:", toc, "seconds")
        
        return result          