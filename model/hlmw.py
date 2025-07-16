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
from scipy.spatial import ConvexHull
import setops as setops # Custom module by TK, from CSM
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.interpolate import pchip, Akima1DInterpolator

class hlmw_mod(object):
    """Head-Liu-Menzio-Wright (2012): Model primitives and methods"""
    def __init__(self, 
                 β = 0.9804, σ_CM = 1.0, σ_DM = 0.4225, 
                 Abar = 1.0, Ubar_CM = 1.0, Ubar_DM = 1.0, 
                 c = 1.0, τ_b = 0.0, τ_max = 0.1, τgrid_size = 120,
                 z_min = 1e-2, z_max = 1.0, zgrid_size = 120, 
                 N_reimann = 1000, Tol=1e-12, N_local = 20,τ_min=0.01,
                 α_1=0.3,n=0.8
                ):
        #Model parameters
        #-------------------------------             # Default parameter settings:
        self.β = β                                   # Discount factor
        
        """Preferences"""
        
        #HS Log.

        #define new variable measure of active byuer n 
        
        self.n    =   n                              # Measure of active byuer 
        self.σ_CM = σ_CM                             # CRRA CM (market 2)   
        self.σ_DM = σ_DM                             # CRRA DM (market 1)
        self.Abar = Abar                             # Labor disutility scale -Abar*h  
        self.Ubar_CM = Ubar_CM                       # DM utility scale Ubar_DM*u(q)
        self.Ubar_DM = Ubar_DM                       # CM utility scale Ubar_CM*u(x)
        
        """Matching"""
        self.α_1 = α_1                               # Prob. of 1 firm contact 
        self.α_2 = 1-self.α_1                        # Prob. of 2 firm contacts: residual
        
        
        x_star = self.invU_C_CM(Abar)                # Optimal CM consumption: from FoC 
        self.x_star = x_star                         # in CM: quasilinear preference         
        """Production"""
        self.c = c                                   # Real marginal cost of production
                
        """Transfer/tax/inflation rate"""
        self.τ_b = τ_b                               # Transfer/tax to active DM buyers       
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
        
        
    ## Interpolation tool ------------------------------------------------##
    def InterpFun1d(self, xdata, ydata, InterpMethod='slinear'):
        """ Interpolate 1D functions given data points (xdata, ydata). 
        Returns instance of class: funfit. NOTE: funfit.derivative() will 
        provide the derivative functions of particular smooth 
        approximant---depends on class of interpolating function. 
        See SCIPY.INTERPOLATE subclasses for more detail.   """

        # xdata = np.sort(xdata)
        if InterpMethod == 'slinear':
            funfit = spline(xdata, ydata, k=1)  # B-spline 1st order
        elif InterpMethod == 'squadratic':
            funfit = spline(xdata, ydata,  k=2)  # instantiate B-spline interp
        elif InterpMethod == 'scubic':
            funfit = spline(xdata, ydata, k=3)  # instantiate B-spline interp
        elif InterpMethod == 'squartic':
            funfit = spline(xdata, ydata, k=4)  # instantiate B-spline interp
        elif InterpMethod == 'squintic':
            funfit = spline(xdata, ydata,  k=5)  # instantiate B-spline interp
        elif InterpMethod == 'pchip':
            # Shape preserving Piecewise Cubic Hermite Interp Polynomial splines
            funfit = pchip(xdata, ydata)
        elif InterpMethod == 'akima':
            funfit = Akima1DInterpolator(xdata, ydata)
        return funfit  # instance at m point(s)
        
        
        #HS_LOG
        
        # modify the return part
        # back out just R_gird  
        
    def R(self, p_grid, Rex, DropFakes=True):
        """Find convex hull of graph(Rex) to get R - See Appendix B.3
        Taken from https://github.com/phantomachine/csm/
        """
        Rex_graph = np.column_stack((p_grid, Rex))
        
        # Step 0:   Find Convex hull of { ρ, Rex(ρ) }
        mpoint = np.array([p_grid.max(), Rex.min()])  # SE corner
        graph = np.vstack((Rex_graph, mpoint))
        chull = ConvexHull(graph)
        extreme_pts = graph[chull.vertices, :]
        
        # Step 1: Intersection between original function Rex and ext. pts.
        v_intersect_graph, ind = setops.intersect(Rex_graph, extreme_pts)[0:2]
        
        # Step 2: First difference the index locations
        idiff = np.diff(ind, n=1, axis=0)
        
        # Step 3: if idiff contains elements > 1 then we know exist one/more line 
        # segment (i.e., more than one set of lotteries played). Location where 
        # straddles more than 1 step
        idz = np.where(idiff > 1)[0]
       
        # Step 4: Given the jumps, we have the end points defining each lottery!
        idz = np.column_stack((idz, idz+1))
        
        # Step 5: Store lottery supports
        lottery_supports = v_intersect_graph[idz, 0]
        lottery_payoffs = v_intersect_graph[idz, 1]
        
        # print(lottery_supports)
        # Step 6: Interpolate to approximate R
        R_fit = self.InterpFun1d(v_intersect_graph[:, 0],
                            v_intersect_graph[:, 1])
        
        # Step 7: Eliminate Fake Lotteries (below-tolerance numerical imprecisions)
        if DropFakes:
            selector = []
            for idx_lot in range(lottery_supports.shape[0]):
                # Lottery prizes for at current lottery segment, idx_lot
                p_lo, p_hi = lottery_supports[idx_lot, :]
                # Conditions: points between current lottery segment
                find_condition = (p_grid > p_lo) & (p_grid < p_hi)
                # Value of Vtilde between current lottery segment
                Rex_p_temp = Rex[find_condition]
                # Value of V between current lottery segment
                R_p_temp = R_fit(p_grid[find_condition])
                # Check: Is R "significantly different" from Rex at lottery segment?
                gap = np.absolute(R_p_temp - Rex_p_temp).max()
                if gap > 1e-10:
                    selector.append(idx_lot)  # Keep, if "YES"
                # Update definition of the set of lotteries
            lottery_supports_temp = lottery_supports[selector, :]
            lottery_payoffs_temp = lottery_payoffs[selector, :]
            # if lottery_supports_temp.size == 0:
                #     lottery_supports = lottery_supports[0, :]
                #     lottery_payoffs = lottery_payoffs[0, :]
                # else:
            if lottery_supports_temp.size > 0:
                lottery_supports = lottery_supports_temp
                lottery_payoffs = lottery_payoffs_temp
        # print(lottery_supports)
        # Step 8: Store R as evaluated on finite set p_grid
        R_grid = R_fit(p_grid)
        return R_grid,R_fit
        
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
        return self.Ubar_CM*MU
    
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
        return self.Ubar_DM*mu    
    
    def invu_q_DM(self, marginal_value):
        """Inverse of du/dq function, Uprime of DM good"""
        return ( ( marginal_value/self.Ubar_DM )**( -1.0 / self.σ_DM ) )
    
    def cost_DM(self, q):
        """DM production technology - linear cost of producing a unit of q"""
        cost = q
        return cost
    
    def invcost_DM(self, ):
        """Inverse of DM production function - output in DM q 
        associated with given level of cost"""
        q = self.c
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
        
        if ρ <= ρ_hat:
            """liquidity constrained"""
            q = z / ρ
        
        else:
            """liquidity unconstrained"""
            q = ρ**(-1.0/self.σ_DM)
            
        return q

    def R_ex_scalar(self, ρ, z):
        """seller's ex-post profit per customer served"""
        ρ_hat = self.ρ_hat_func(z)
        if  ρ <= ρ_hat:
            """profit margin from serving 
            liquidity constrained and zero borrowing buyer
            """
            qb = z / ρ
            val = qb * ( ρ - self.c )
        
        else:
            qb = ρ**(-1.0/self.σ_DM)
            val = qb * ( ρ - self.c )
            
        return val
    
    def R_ex(self, ρ, z):
        """seller's ex-post profit per customer served.
        ρ is a 1D array, (i,z) are scalars. This function is 
        the vectorized version of R_ex_scalar() above."""
        
        # Pricing cutoffs
        ρ_hat = self.ρ_hat_func(z)
        
        qb = np.zeros(ρ.shape)
        val = qb.copy()

        # liquidity constrained and zero borrowing buyer
        bool =  (ρ <= ρ_hat)
        qb[bool] = z / ρ[bool]
        val[bool] = qb[bool] * (ρ[bool] - self.c)
        
        # money unconstrained
        bool = (ρ > ρ_hat)
        qb[bool] = ρ[bool]**(-1.0/self.σ_DM)
        val[bool] = qb[bool] * (ρ[bool] - self.c)

        return val
    
    ##-----------------DM goods expenditure and sellers' profit margin------------##
    def q_expenditure(self, ρ, z):
        """expenditure := ρq"""        
        ρ_hat = self.ρ_hat_func(z)
 
        if ρ <= ρ_hat:
            """liquidity constrained"""
            ρq = z
        
        else:
            """liquidity unconstrained"""
            ρq = ρ * ( ρ**(-1.0/self.σ_DM) )
            
        return ρq
        
    ##-----------------bounds on DM goods prices------------##
    def ρ_max_func(self, z, τ):
        """Equilibrium upper bound on the support of F: ρ_{\max}"""
        if τ > self.β-1.0:
            ρ_hat = self.ρ_hat_func(z)
            ρ_constant = self.c / (1.0 - self.σ_DM)
            ρ_ub = np.max( [ ρ_hat, ρ_constant ] ) #monopoly price in general

            #Note: Case 1: ρ_ub = ρ_hat implies all liquidity constrained
            #Note: Case 2: ρ_ub = ρ_constant implies mixture of liquidity constrained and unconstrained
        else:
            ρ_ub = self.c
        return ρ_ub
    
    def ρ_min_func(self, z, τ):
        """Equal expected profit condition to back out the lower bound
        on the support of F given R(ρ_{\max})
        """ 
        if τ > self.β - 1.0:
            ρ_max = self.ρ_max_func(z, τ)

            ρ_range= np.linspace(self.c, ρ_max, self.N_reimann)  

            ## Temporarily, start by setting rho_max and begin with c at the minimum.

            Rex = self.R_ex(ρ_range, z)

            R_grid,R_fit=self.R(ρ_range, Rex, DropFakes=True)

            noisy_search = self.α_1  / ( self.α_1 + 2.0 * self.α_2 ) 

            LHS = lambda ρ: self.q_demand_func(ρ, z) * ( ρ - self.c )

            RHS = noisy_search * R_fit(ρ_max)  

            vals_obj = lambda ρ: RHS - LHS(ρ)

            ρ_lb = brentq(vals_obj, self.c, ρ_max)
            
        else:
            ρ_lb = self.c
        return ρ_lb
    
    ##-----------DM goods price distribution: CDF and pdf in terms of aggregate variables-------##
    def F_func(self, z, τ):
        """Posted price distribution (CDF): F(ρ, z)
        """
        
        ρ_max = self.ρ_max_func(z, τ)
        
        ρ_range= self.support_grid_func(z, τ)
        
        Rex = self.R_ex(ρ_range, z)
        
        R_grid,R_fit=self.R(ρ_range, Rex, DropFakes=True)

        noisy_search = self.α_1 / ( 2.0 * self.α_2 ) 
        
        R_max = R_fit(ρ_max)
        
        R =R_grid
        
        CDF_value =  1.0 - noisy_search * ( ( R_max / R ) - 1.0 ) 
        
        return CDF_value    
    
    def dF_func(self, z,τ):
        """Density of the price distribution (PDF): dF(ρ, z)/dρ"""
        
        ρ_max = self.ρ_max_func(z, τ)
        
        ρ_range= self.support_grid_func(z, τ)
        
        noisy_search = self.α_1 / ( 2.0 * self.α_2 ) 
        
        Rex = self.R_ex(ρ_range, z)
        
        R_grid,R_fit=self.R(ρ_range, Rex, DropFakes=True)
        
        R_max =R_fit(ρ_max)
        
        R =R_grid
        
        dR_dρ = R_fit.derivative()(ρ_range)
        
        pdf_value = noisy_search*( ( R_max / ( R**2.0 ) ) * dR_dρ )
                
        return pdf_value
      
    def support_grid_func(self, z, τ):
        a = self.ρ_min_func(z, τ)
        b = self.ρ_max_func(z, τ)       
        supports = np.linspace(a, b, self.N_reimann)
        return supports
    
    def dF_normalization_func(self, z, τ):
        
        ρ_grid = self.support_grid_func(z, τ)
        dF = self.dF_func(z, τ)
        width = (ρ_grid[-1]-ρ_grid[0])/(len(ρ_grid)-1)
        
        dF_sum = np.sum(width * dF)
        dF_nor = dF / dF_sum
        return dF_nor     
    #HS_log 
    # Times measure of active buyers (n)
    
    ##-----------------Money demand-------------------------------##
    def money_euler_rhs(self, z, τ):
        """Money demand net expected benefit"""
        if τ > self.β - 1.0:
            
            ρ_grid = self.support_grid_func(z, τ)

            dF_grid = self.dF_normalization_func(z, τ)
            
            ρ_hat = self.ρ_hat_func(z)

            em_func = self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(z, τ))
            
            ρ_hat_index = np.argmax(ρ_grid >= ρ_hat)

            em = em_func[:ρ_hat_index]

            LP_func = lambda ρ: ( self.mu_DM( self.q_demand_func(ρ, z) ) / ρ ) -1.0

            liquidity_premium = np.array( [LP_func(ρ) for ρ in ρ_grid[:ρ_hat_index]] )

            mu_buyer = (em * liquidity_premium * dF_grid[:ρ_hat_index])

            value = self.n*np.trapz(mu_buyer,  ρ_grid[:ρ_hat_index])  # plz, check this.
        
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
        z_sol= brentq(self.money_euler_obj, self.z_min, self.z_max, args=(τ))
        return z_sol


    '''
    Below part, we may time active measure (n)
    plz, check
    '''

    ##------------------SME and Stat---------------------------------##
    def Total_q_func(self, z, τ):
        """Total DM goods"""
        if τ > self.β-1.0:
            ρ_grid = self.support_grid_func(z, τ)

            pdf_grid = self.dF_normalization_func(z, τ)

            em = self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(z, τ))

            q_expost = np.array( [self.q_demand_func(ρ, z) for ρ in ρ_grid] )

            integrand_values = em * q_expost * pdf_grid

            total_q = self.n*np.trapz(integrand_values, ρ_grid)
        else:
            q_FR = z/self.c
            
            total_q = self.n*q_FR
        return total_q
    
    def Total_ρq_func(self, z, τ):
        """Total DM expenditure"""        
        if τ > self.β-1.0:
            ρ_grid = self.support_grid_func(z, τ)

            pdf_grid = self.dF_normalization_func(z, τ)

            em = self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(z, τ))

            ρq_expost = np.array( [self.q_expenditure(ρ, z) for ρ in ρ_grid] )

            integrand_values = em * ρq_expost * pdf_grid

            total_ρq = self.n*np.trapz(integrand_values, ρ_grid)
        else:
            q_FR = z/self.c
            total_ρq = self.n*q_FR*self.c
        return total_ρq    
    
    def firm_profit_func(self, z, τ):
        """Firms' expected transacted profit"""    
        if τ>self.β-1.0:
            ρ_grid = self.support_grid_func(z, τ)

            pdf_grid =  self.dF_normalization_func(z, τ)

            em = self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(z, τ))

            ρ_range=self.support_grid_func(z, τ)

            Rex = self.R_ex(ρ_range, z)

            R_grid,R_fit=self.R(ρ_range, Rex, DropFakes=True)

            profit_margin = R_grid

            integrand_values = em * profit_margin * pdf_grid

            firm_profit = np.trapz(integrand_values, ρ_grid)   
        else:
            firm_profit = 0.0
        return firm_profit
    
    def markup_func(self, z, τ):
        """Consumption weighted transactred markup"""        
        if τ>self.β-1.0:
            ρ_grid = self.support_grid_func(z, τ)

            pdf_grid = self.dF_normalization_func(z, τ)

            expost_markup_func = lambda ρ: ρ / self.c

            markup_expost = np.array([expost_markup_func(ρ) for ρ in ρ_grid])

            # DM consumption share
            q_share = self.n* self.Total_q_func(z, τ) / (self.Total_ρq_func(z, τ) + self.x_star)

            # CM consumption share
            x_share = self.n* self.x_star / (self.Total_ρq_func(z, τ) + self.x_star)

            # normalized
            nor_share = q_share / x_share

            # dm_consumption_share * dm_markup + cm_consumption_share * cm_markup(=1, CM p=mc) 
            # normalized by x_share to get consumption weighted markup
            markup = np.trapz(nor_share * (markup_expost * pdf_grid), ρ_grid) + 1.0
        else:
            markup = 1.0
        return markup
    
    def DM_utility(self, z, τ):
        """
        DM utility 
        """        
        if τ>self.β-1.0:
            ρ_grid = self.support_grid_func(z, τ)

            pdf_grid = self.dF_normalization_func(z, τ)
            
            em = self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(z, τ))

            dm_net_func = lambda ρ: self.u_DM(self.q_demand_func(ρ,z)) - self.cost_DM(self.q_demand_func(ρ,z))

            expost_utility_α1_α2 = np.array( [ dm_net_func(ρ) for ρ in ρ_grid] )

            utility = self.n*np.trapz(em * expost_utility_α1_α2 * pdf_grid, ρ_grid ) 
        else:
            qb_FR = z

            utility = self.n*(self.α_1 + self.α_2) * ( self.u_DM(qb_FR)- self.cost_DM(qb_FR) ) 
        return utility    
    
    def DM_utility_delta(self, z, τ, delta):
        """
        DM utility change by delta
        """        
        if τ>self.β-1.0:
            ρ_grid = self.support_grid_func(z, τ)

            pdf_grid = self.dF_normalization_func(z, τ)

            em = self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(z, τ))

            dm_net_func = lambda ρ: self.u_DM(self.q_demand_func(ρ,z)*delta) - self.cost_DM(self.q_demand_func(ρ,z))

            expost_utility_α1_α2 = np.array( [ dm_net_func(ρ) for ρ in ρ_grid] )

            utility = self.n*np.trapz( em * expost_utility_α1_α2 * pdf_grid, ρ_grid )
        else:
            qb_FR = z

            utility = self.n*(self.α_1 + self.α_2) * ( self.u_DM(qb_FR*delta)- self.cost_DM(qb_FR) ) 
        return utility      
    
    def welfare_func(self, z, τ):
        """Total welfare"""
        discount = ( 1.0 / ( 1.0 - self.β ) )
        if τ == self.β - 1.0:
            
            #allocation under Friedman rule
            qb_FR = z

            DM_segment = self.n*(self.α_1 + self.α_2) * ( self.u_DM(qb_FR) - self.cost_DM(qb_FR) )
            
            labor_CM = self.x_star
            
            CM_segment = self.U_CM(self.x_star) - self.h(labor_CM)
            
            lifetime_utility =  discount*(DM_segment + CM_segment )
        
        else:
            
            DM_segment = self.DM_utility(z, τ) 
            
            labor_CM = self.x_star 
            
            CM_segment =  (self.U_CM(self.x_star) -  self.h(labor_CM))
            
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

            DM_segment = self.n*(self.α_1 + self.α_2) * ( self.u_DM(qb_FR*delta) - self.cost_DM(qb_FR) )
            
            labor_CM = self.x_star
            
            CM_segment = self.U_CM(self.x_star*delta) - self.h(labor_CM)
            
            lifetime_utility =  discount*(DM_segment + CM_segment )
        
        else:
            
            DM_segment = self.DM_utility_delta(z, τ, delta) 
            
            labor_CM = self.x_star 
            
            CM_segment =  (self.U_CM(self.x_star*delta) -  self.h(labor_CM))
            
            lifetime_utility = discount*(DM_segment + CM_segment )
            
        return lifetime_utility      
    
    def SME_stat_func(self, z, τ):
        ## calculate aggregate markup
        Y = self.x_star + self.Total_q_func(z, τ)
        
        DM_share = self.Total_q_func(z, τ) / Y
        
        CM_share = self.x_star / Y
        
        nor_share = DM_share / CM_share
        
        if τ>self.β-1.0:
        
            ρ_grid = self.support_grid_func(z, τ)

            pdf_normalized_grid = self.dF_normalization_func(z, τ)

            #calculate (posted) price dispersion
            price_mean = np.trapz(ρ_grid * pdf_normalized_grid, ρ_grid)
            y_1 = pdf_normalized_grid * ( (ρ_grid  - price_mean)**2.0 )
            price_sd = np.sqrt( np.trapz(y_1, ρ_grid) )
            price_cv = price_sd / price_mean    



            markup_deviation = ρ_grid / self.c  

            markup_mean = self.markup_func(z, τ) 

            y_2 =  pdf_normalized_grid * ( (markup_deviation - markup_mean)**2.0 ) * (nor_share**2.0)  + 0.0 # CM_dispersion = 0

            markup_sd = np.sqrt( np.trapz(y_2, ρ_grid) ) 

            markup_cv = markup_sd / markup_mean
        else:
            price_mean = self.c
            price_sd = 0.0
            price_cv = 0.0
            markup_mean = 1.0
            markup_sd = 0.0
            markup_cv = 0.0

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
            
#            print(τ)

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
"""
σ_DM=0.5
Ubar_CM=1.9
n=0.65
α_1=0.1    

N_reimann = 300

model =hlmw_mod(σ_DM=σ_DM, Ubar_CM=Ubar_CM, n=n,τ_max = 0.00,α_1=α_1,τgrid_size = 20,N_reimann = N_reimann,Tol=1e-5)
        

τ = 0
z= 0.7

ρ_range= model.support_grid_func(z, τ)

plt.plot(ρ_range,model.F_func(z,τ))
plt.plot(ρ_range,model.dF_func(z,τ))
plt.plot(ρ_range,model.dF_normalization_func(z,τ))

integral = np.trapz(model.dF_func(z,τ), ρ_range)

print(integral)


ρ_grid = model.support_grid_func(z, τ)

dF_grid = model.dF_normalization_func(z, τ)


Buyer: can be liquidity constrained + borrowing or liquidity constrained + zero borrowing


ρ_hat = model.ρ_hat_func(z)

em_func = model.α_1 + 2.0*model.α_2*( 1.0 - model.F_func(z, τ))

ρ_hat_index = np.argmax(ρ_grid > ρ_hat)

em = em_func[:ρ_hat_index]

LP_func = lambda ρ: ( model.mu_DM( model.q_demand_func(ρ, z) ) / ρ ) -1.0

liquidity_premium = np.array( [LP_func(ρ) for ρ in ρ_grid[:ρ_hat_index]] )

mu_buyer = (em * liquidity_premium * dF_grid[:ρ_hat_index])

value = model.n*np.trapz(mu_buyer,  ρ_grid[:ρ_hat_index])  # plz, check this.

print(value)
tic = time.time()
print(model.z_solver(τ))
toc = time.time() - tic
print("Elapsed time of solving SME:", toc, "seconds")

model_robustness =hlmw_robustness.hlmw_mod(σ_DM=σ_DM, Ubar_CM=Ubar_CM, n=n,τ_max = 0.00,α_1=α_1,τgrid_size = 20,N_reimann = N_reimann,Tol=1e-5)
   

tic = time.time()
print(model_robustness.z_solver(τ))
toc = time.time() - tic
print("Elapsed time of solving SME:", toc, "seconds")

result_hlmw = model.SME_stat() # this uses brentq only


σ_DM=0.5
Ubar_CM=1.9
n=0.65
α_1=0.1    
model_hlmw_convexfication =hlmw_mod(σ_DM=σ_DM, Ubar_CM=Ubar_CM, n=n,τ_max = 0.00,α_1=α_1,τgrid_size = 20,N_reimann = 300,Tol=1e-5)

result_hlmw_convexfication = model_hlmw_convexfication.SME_stat() # this uses brentq only

import hlmw as hlmw

model_hlmw =hlmw.hlmw_mod(σ_DM=σ_DM, Ubar_CM=Ubar_CM, n=n,τ_max = 0.00,α_1=α_1,τgrid_size = 20,N_reimann = 300,Tol=1e-5)
result_hlmw = model_hlmw.SME_stat() # this uses brentq only


#plt.plot(result_hlmw['allocation_grid']['zstar'])
#plt.plot(result_hlmw_convexfication['allocation_grid']['zstar'])



#temp=result_hlmw_convexfication['allocation_grid']['zstar']-result_hlmw['allocation_grid']['zstar']

"""