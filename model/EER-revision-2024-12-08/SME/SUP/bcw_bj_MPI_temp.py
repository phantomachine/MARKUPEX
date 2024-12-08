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
import setops as setops # Custom module by TK, from CSM

from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.interpolate import pchip, Akima1DInterpolator
import scipy.interpolate as interp
#from IPython.display import display, Math
from scipy.spatial import ConvexHull

from mpi4py import MPI
import hlmw as hlmw

class baseline_mod(object):
    """Model primitives and methods"""
    def __init__(self, 
              β = 0.9804, σ_CM = 1.0, σ_DM = 0.4225,  
                 Abar = 1.0, Ubar_CM = 1.0, Ubar_DM = 1.0, 
                 c = 1.0, τ_b = 0.0, τ_max = 0.1, τgrid_size = 120, 
                 z_min = 1e-10, z_max = 1.0, zgrid_size = 120, 
                 N_reimann = 500, Tol=1e-5,
                 α_1=0.3, n=0.8
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
        
        
        #HS Log.

        #Removed alpha_0.
        #Chose an arbitrary alpha_1 value of 0.3.
        
        """Matching"""
        self.α_1 = α_1                               # Prob. of 1 firm contact 
        self.α_2 = 1-self.α_1                        # Prob. of 2 firm contacts: residual
        

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
        
        #Array of grids on z
        self.z_grid = np.linspace(z_min, z_max, zgrid_size)      
        
        
        #Array of grids on τ
        #self.τ_grid = np.arange(β-1.0, 0.08+0.0025, 0.0025)  #np.linspace(τ_min, τ_max, τgrid_size) 
        self.τ_grid = np.linspace(τ_min, τ_max, τgrid_size) 
        # Reimann integral domain partition
        self.N_reimann = N_reimann
        
        self.Tol = Tol
        
        
    def z_cutoffs(self, i):
        """Define the four cutoff levels for z as in Lemmata 1-3:
            0 < z_tilde_i < zhat < z_prime < ∞."""

        z_hat = (self.c/(1-self.σ_DM))**((self.σ_DM-1)/self.σ_DM)
        z_prime = z_hat*(1/(1-self.σ_DM))**(-(self.σ_DM-1)/self.σ_DM)
        z_tilde = z_hat*(1+i)**(-1/self.σ_DM)
        # z_breve = (1-self.c) + z_hat*self.σ_DM
        # z_breve =((1+self.σ_DM*z_tilde)/self.c)**(1-self.σ_DM)
    
        # Sanity check!
        # zcut = np.diff([z_tilde, z_hat])
        if z_tilde <= z_hat:
            #print("🥳 Congrats: Your cutoffs are in order!")
            z_cutoffs_str = '\\tilde{z}_{i} < \\hat{z} < z^{\\prime}'
#            display(
#                Math(z_cutoffs_str)
#                )
            z_cutoffs = [z_tilde, z_hat, z_prime] 
            #print(['%.3f' % x for x in z_cutoffs])
#        else:
#            print("💩 You have a problem with your cutoff ordering!")
            
        return z_tilde, z_hat, z_prime 
        
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
        q = self.c
        return q        
    ##-------------------Gross inflation and CM transfer-------------##
    def GrossInflation(self, τ):
        """Gross inflation"""
        γ = 1.0 + τ
        return γ
    
    def τ_CM(self, τ):
        """Tax/transfer to CM households"""
        τ1 = (self.α_1+self.α_2)*self.τb 
        τ2 = τ - τ1
        return τ2    
        
    def i_policy(self, τ):
        """
        Nominal policy interest rate in steady state
        """
        γ = self.GrossInflation(τ)
        policy_rate = γ / self.β - 1.0
        return policy_rate    
    
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
            
        elif ρ_tilde < ρ <= ρ_hat:
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
            
        elif ρ_tilde < ρ <= ρ_hat:
            """liquidity constrained and zero borrowing buyer"""
            loan = 0.0
        
        else:
            """liquidity unconstrained and zero borrowing buyer"""
            loan = 0.0
 
        return loan
    
    #HS_LOG
    # Times measure of active buyer (n)
    
    def Total_loan_func(self, i, z, τ):        
        ρ_grid = self.support_grid_func(z, i, τ)

        pdf_grid = self.dF_normalization_func(z, i, τ)

        em_func = lambda ρ: self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(ρ, i, z,τ) ) 

        em = np.array( [em_func(ρ) for ρ in ρ_grid] )

        loan_expost = np.array( [self.ξ_demand_func(ρ, i, z) for ρ in ρ_grid] )

        integrand_values = em * loan_expost * pdf_grid        
        total_loan = self.n*np.trapz(integrand_values, ρ_grid)  ## Plz, Check this part

        return total_loan
    
        #HS Log.

        # Define new δ deposit function 
    
    def δ_demand_func(self, ρ, i, z):
        """DM loans demand function"""
        ρ_hat = self.ρ_hat_func(z)
        ρ_tilde = self.ρ_tilde_func(z, i)

        if ρ <= ρ_tilde:
            """liquidity constrained and zero deposit buyer"""
            deposit = 0.0
            
        elif ρ_tilde < ρ <= ρ_hat:
            """liquidity constrained and zero deposit buyer"""
            deposit = 0.0
        
        else:
            """liquidity unconstrained and deposit buyer"""
            deposit = z - ρ*ρ**(-1.0/self.σ_DM)
 
        return deposit
    
        #HS Log.

        # Define new total δ deposit function 
        # Change measure of inactive buyer:  α_0 -> 1-n
        
    def Total_deposit_func(self, i, z, τ):        
        ρ_grid = self.support_grid_func(z, i, τ)

        pdf_grid = self.dF_normalization_func(z, i, τ)

        em_func = lambda ρ: self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(ρ, i, z,τ) ) 

        em = np.array( [em_func(ρ) for ρ in ρ_grid] )

        deposit_expost = np.array( [self.δ_demand_func(ρ, i, z) for ρ in ρ_grid] )

        integrand_values = em * deposit_expost * pdf_grid        
        #total_loan =  (1-self.n) * z + self.n *np.trapz(integrand_values, ρ_grid)
        
        # S
        total_deposit  =  (1-self.n) * z + self.n *np.trapz(integrand_values, ρ_grid)
        return total_deposit #total_loan
    
    def i_rate_obj(self, i, z, τ):
        """"""
        
        # Total deposit
        LHS = self.Total_deposit_func(i,z,τ) 
        
        # Total loan
        RHS = self.Total_loan_func(i,z,τ) 
        net = LHS - RHS
 
        return net
    
    ##-----------------DM goods expenditure and sellers' profit margin------------##
    def q_expenditure(self, ρ, i, z):
        """expenditure := ρq"""        
        ρ_hat = self.ρ_hat_func(z)
        ρ_tilde = self.ρ_tilde_func(z, i)

        if ρ <= ρ_tilde:
            """liquidity constrained and borrowing unconstrained buyer"""
            ρq = ρ * ( ( ρ*(1.0 + i ) )**(-1.0/self.σ_DM) )
            
        elif ρ_tilde < ρ <= ρ_hat:
            """liquidity constrained and zero borrowing buyer"""
            ρq = z
        
        else:
            """liquidity unconstrained and zero borrowing buyer"""
            ρq = ρ * ( ρ**(-1.0/self.σ_DM) )
            
        return ρq
    
    def R_ex_scalar(self, ρ, i, z):
        """seller's ex-post profit per customer served"""
        ρ_hat = self.ρ_hat_func(z)
        ρ_tilde = self.ρ_tilde_func(z, i)
        if ρ <= ρ_tilde:
            """profit margin from serving
            liquidity constrained and borrowing unconstrained buyer
            """
            qb = ( ρ*(1.0 + i ) )**(-1.0/self.σ_DM)
            val = qb * ( ρ - self.c ) 
        
        elif ρ_tilde < ρ <= ρ_hat:
            """profit margin from serving 
            liquidity constrained and zero borrowing buyer
            """
            qb = z / ρ
            val = qb * ( ρ - self.c )
        
        else:
            qb = ρ**(-1.0/self.σ_DM)
            val = qb * ( ρ - self.c )
            
        return val
    
    def R_ex(self, ρ, i, z):
        """seller's ex-post profit per customer served.
        ρ is a 1D array, (i,z) are scalars. This function is 
        the vectorized version of R_ex_scalar() above."""
        
        # Pricing cutoffs
        ρ_hat = self.ρ_hat_func(z)
        ρ_tilde = self.ρ_tilde_func(z, i)
        
        qb = np.zeros(ρ.shape)
        val = qb.copy()
        
        # liquidity constrained and borrowing unconstrained buyer
        bool = (ρ <= ρ_tilde)
        qb[bool] = (ρ[bool]*(1.0 + i))**(-1.0/self.σ_DM)
        val[bool] = qb[bool] * (ρ[bool] - self.c)

        # liquidity constrained and zero borrowing buyer
        bool = (ρ_tilde < ρ) & (ρ <= ρ_hat)
        qb[bool] = z / ρ[bool]
        val[bool] = qb[bool] * (ρ[bool] - self.c)
        
        # money unconstrained
        bool = (ρ > ρ_hat)
        qb[bool] = ρ[bool]**(-1.0/self.σ_DM)
        val[bool] = qb[bool] * (ρ[bool] - self.c)

        return val

    def G1(self, ρ, i, z):
        """seller's ex-post profit per customer served
        profit from serving liquidity constrained and 
        borrowing unconstrained buyer"""
        qb = ( ρ*(1.0 + i ) )**(-1.0/self.σ_DM)
        val = qb * ( ρ - self.c ) 
                    
        return val

    def G2(self, ρ, i, z):
        """seller's ex-post profit per customer served"""
        qb = z / ρ
        val = qb * ( ρ - self.c )
        
        return val

    def G3(self, ρ, i, z):
        qb = ρ**(-1.0/self.σ_DM)
        val = qb * ( ρ - self.c )
            
        return val
    
    #HS_LOG
    
    # set ρ_max by following Tim's argument 
    # call z_cutoofs function
    # set ρ_max range 
    
    ##-----------------bounds on DM goods prices------------##
    def ρ_max_func(self, z,i):
        """Equilibrium upper bound on the support of F: ρ_{max}"""
        
        ρ_hat = self.ρ_hat_func(z)
        ρ_constant = self.c / (1.0 - self.σ_DM)
        
        z_tilde, z_hat, z_prime =self.z_cutoffs(i)
        
        if z>z_hat:
            ρ_ub = ρ_constant
        elif z>z_tilde and z<z_hat: 
            ρ_ub = ρ_hat
        else:
            ρ_ub = ρ_constant
        
        return ρ_ub
    
    #HS_LOG
    
    # add revised R function(from Tim)
    
    def ρ_min_func(self, z, i, τ):
        """Equal expected profit condition to back out the lower bound
        on the support of F given R(ρ_{max})
        """ 
        
        ρ_max = self.ρ_max_func(z,i)
                
        ρ_range= np.linspace(self.c, ρ_max, self.N_reimann)  
        
        ## Temporarily, start by setting rho_max and begin with c at the minimum.
        
        Rex = self.R_ex(ρ_range, i, z)
        
        R_grid,R_fit=self.R(ρ_range, Rex, DropFakes=True)
        
        noisy_search = self.α_1  / ( self.α_1 + 2.0 * self.α_2 ) 

        LHS = lambda ρ: self.q_demand_func(ρ, i, z) * ( ρ - self.c )

        RHS = noisy_search * R_fit(ρ_max)  
        
        vals_obj = lambda ρ: RHS - LHS(ρ)
        
        ρ_lb = brentq(vals_obj, self.c, ρ_max)
       
        return ρ_lb
    
    ##-----------DM goods price distribution: CDF and pdf in terms of aggregate variables-------##
    def F_func(self, ρ, i, z, τ):
        """Posted price distribution (CDF): F(ρ, i, z)
        Note: i := i(ρ, z)
        """
        
        ρ_max = self.ρ_max_func(z,i)
        
        ρ_range=self.support_grid_func(z, i, τ)
        
        Rex = self.R_ex(ρ_range, i, z)
        
        R_grid,R_fit=self.R(ρ_range, Rex, DropFakes=True)

        noisy_search = self.α_1 / ( 2.0 * self.α_2 ) 
        
        R_max = R_fit(ρ_max)
        
        R =R_fit(ρ)
        
        CDF_value =  1.0 - noisy_search * ( ( R_max / R ) - 1.0 ) 
        
        return CDF_value    
    
    def dF_func(self, ρ, i, z, τ):
        """Density of the price distribution (PDF): dF(ρ, i, z)/dρ"""
        
        ρ_max = self.ρ_max_func(z,i)
        
        ρ_range=self.support_grid_func(z, i, τ)
        
        noisy_search = self.α_1 / ( 2.0 * self.α_2 ) 
        
        Rex = self.R_ex(ρ_range, i, z)
        
        R_grid,R_fit=self.R(ρ_range, Rex, DropFakes=True)
        
        R_max =R_fit(ρ_max)
        
        R =R_fit(ρ)
        
        dR_dρ = R_fit.derivative()(ρ)
        
        pdf_value = noisy_search*( ( R_max / ( R**2.0 ) ) * dR_dρ )
        
        return pdf_value
    
    def support_grid_func(self, z, i, τ):
        a = self.ρ_min_func(z, i, τ)
        b = self.ρ_max_func(z,i)       
        supports = np.linspace(a, b, self.N_reimann)
        return supports
    
    def dF_normalization_func(self, z, i, τ):
        ρ_grid = self.support_grid_func(z, i, τ)
        dF = np.array( [ self.dF_func(ρ, i, z,τ) for ρ in ρ_grid] )
        w = ρ_grid[1] - ρ_grid[0]
        dF_sum = np.sum(w*dF)
        dF_nor = dF/ dF_sum
        return dF_nor           
    
    
    #HS_log 
    # Add uncontrained depositor - two stage approach 
    # Times measure of active buyers (n)
    
    ##-----------------Money demand-------------------------------##
    def money_euler_rhs(self, z, i, τ):
        
        if τ > self.β - 1.0:
        
            ρ_hat = self.ρ_hat_func(z)

            ρ_grid = self.support_grid_func(z, i, τ)

            dF_grid = self.dF_normalization_func(z, i, τ)


            """
            Buyer: can be liquidity constrained + borrowing or liquidity constrained + zero borrowing

            """

            em_func = lambda ρ: self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(ρ, i, z, τ) )

            em = np.array( [em_func(ρ) for ρ in ρ_grid[ρ_hat>=ρ_grid] ] )

            LP_func = lambda ρ: ( self.mu_DM( self.q_demand_func(ρ, i, z) ) / ρ ) -1.0

            liquidity_premium = np.array( [LP_func(ρ) for ρ in ρ_grid[ρ_hat>=ρ_grid]] )

            mu_buyer = em * liquidity_premium * dF_grid[ρ_hat>=ρ_grid]
            
            """ 
            
            inactive depositor + unconstrained depositor 
            
            """
            
            ### plz, check ### - OK, checked by TK, 2024-11-03
            
            em_deposit = np.array( [em_func(ρ) for ρ in ρ_grid[ρ_hat<ρ_grid] ] )
            
            deposit_premium = np.array( [ i for ρ in ρ_grid[ρ_hat<ρ_grid]] )
            
            mu_depositor_1 = (1-self.n) * i 
            
            mu_depositor_2 = em_deposit * deposit_premium * dF_grid[ρ_hat<ρ_grid]

            value = mu_depositor_1 +self.n*( np.trapz(mu_buyer,  ρ_grid[ρ_hat>=ρ_grid]) \
            + np.trapz(mu_depositor_2,  ρ_grid[ρ_hat<ρ_grid]))
                    
        else:
            value = z**(-self.σ_DM) - 1.0
        
        return value
    
    def money_euler_obj(self, z, i, τ):
        LHS = self.i_policy(τ)
        RHS = self.money_euler_rhs(z, i, τ)
        net = LHS - RHS
        return net
    
    def system_func(self, initial_guess, τ):
        z = initial_guess[0]
        i = initial_guess[1]
        
        z_obj = self.money_euler_obj(z, i, τ)
        i_obj = self.i_rate_obj(i, z, τ)
        
        print(z)
        print(i)
        
        return [z_obj, i_obj]
    
    def solve_z_i(self, z_guess, i_guess, τ):
        
        if τ > self.β - 1.0:
        
        
            x0 = [z_guess, i_guess]
        
            x = fsolve(self.system_func, x0, xtol=1e-5, args=(τ), full_output=False)

            z = x[0] # solution
            i = x[1] # solution
        else:
            
            z= 1
            i= 0
        
        if i < self.i_policy(τ):
            i = 0.0
        else:
            i = i
        
        return z, i                        
    
    '''
    Below part, we may time active measure (n)
    plz, check
    '''

    ##------------------SME and Stat---------------------------------##
    def Total_q_func(self, z, i, τ):
        ρ_grid = self.support_grid_func(z, i, τ)

        pdf_grid = self.dF_normalization_func(z, i, τ)
        
        em_func = lambda ρ: self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(ρ, i, z, τ) ) 
            
        em = np.array( [em_func(ρ) for ρ in ρ_grid] )
        
        q_expost = np.array( [self.q_demand_func(ρ, i, z) for ρ in ρ_grid] )
        
        integrand_values = em * q_expost * pdf_grid
        
        total_q = self.n*np.trapz(integrand_values, ρ_grid)
        
        return total_q
    
    def Total_ρq_func(self, z, i, τ):
        ρ_grid = self.support_grid_func(z, i, τ)
   
        pdf_grid = self.dF_normalization_func(z, i, τ)
        
        em_func = lambda ρ: self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(ρ, i, z, τ) ) 
            
        em = np.array( [em_func(ρ) for ρ in ρ_grid] )
        
        ρq_expost = np.array( [self.q_expenditure(ρ, i, z) for ρ in ρ_grid] )
        
        integrand_values = em * ρq_expost * pdf_grid
        
        total_ρq = self.n*np.trapz(integrand_values, ρ_grid)
        
        return total_ρq    
    
    def firm_profit_func(self, z, i, τ):
        ρ_grid = self.support_grid_func(z, i, τ)
      
        pdf_grid = self.dF_normalization_func(z, i, τ)
        
        em_func = lambda ρ: self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(ρ, i, z , τ) ) 
            
        em = np.array( [em_func(ρ) for ρ in ρ_grid] )
        
        noisy_search = self.α_1 / ( 2.0 * self.α_2 ) 
        
        ρ_range=self.support_grid_func(z, i, τ)
        
        Rex = self.R_ex(ρ_range, i, z)
        
        R_grid,R_fit=self.R(ρ_range, Rex, DropFakes=True)
        
        profit_margin = np.array([R_fit(ρ) for ρ in ρ_grid])
        
        integrand_values = em * profit_margin * pdf_grid
        
        firm_profit = np.trapz(integrand_values, ρ_grid)        
        return firm_profit
    
    def markup_func(self, z, i, τ):
        ρ_grid = self.support_grid_func(z, i, τ)
      
        pdf_grid = self.dF_normalization_func(z, i, τ)
        
        em_func = lambda ρ: self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(ρ, i, z, τ) ) 
            
        em = np.array( [em_func(ρ) for ρ in ρ_grid] )
        
        expost_markup_func = lambda ρ: ρ / self.c
        
        markup_expost = np.array([expost_markup_func(ρ) for ρ in ρ_grid])
        
        q_share = self.n*self.Total_q_func(z, i, τ) / (self.Total_q_func(z, i, τ) + self.x_star)
        
        x_share = self.n*self.x_star / (self.Total_q_func(z, i, τ) + self.x_star)
        
        nor_share = q_share / x_share
        
        markup = np.trapz(nor_share * (markup_expost * pdf_grid), ρ_grid) + 1.0
        
        return markup
    
    def DM_utility(self, z, i, τ):
        """
        DM utility 
        """        
        ρ_grid = self.support_grid_func(z, i, τ)
    
        pdf_grid = self.dF_normalization_func(z, i, τ)
        
        em_func = lambda ρ: self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(ρ, i, z, τ) ) 
            
        em = np.array( [em_func(ρ) for ρ in ρ_grid] )
        
        dm_net_func = lambda ρ: self.u_DM(self.q_demand_func(ρ,i,z)) - self.cost_DM(self.q_demand_func(ρ,i,z)) 
        
        expost_utility_α1_α2 = np.array( [ dm_net_func(ρ) for ρ in ρ_grid] )
        
        utility = self.n*np.trapz(em * expost_utility_α1_α2 * pdf_grid, ρ_grid ) 
                    
        return utility    
    
    def DM_utility_delta(self, z, i, τ, delta):
        """
        DM utility change by delta
        """        
        ρ_grid = self.support_grid_func(z, i, τ)
        
        pdf_grid = self.dF_normalization_func(z, i, τ)
        
        em_func = lambda ρ: self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(ρ, i, z, τ) ) 
            
        em = np.array( [em_func(ρ) for ρ in ρ_grid] )
        
        dm_net_func = lambda ρ: self.u_DM(self.q_demand_func(ρ,i,z)*delta) - self.cost_DM(self.q_demand_func(ρ,i,z))  
        
        expost_utility_α1_α2 = np.array( [ dm_net_func(ρ) for ρ in ρ_grid] )
        
        utility = self.n*np.trapz( em * expost_utility_α1_α2 * pdf_grid, ρ_grid )
                    
        return utility      
    
    
    def welfare_func(self, z, i, τ):
        discount = ( 1.0 / ( 1.0 - self.β ) )
        if τ == self.β - 1.0:
            
            #allocation under Friedman rule
            qb_FR = z

            DM_segment = (self.α_1 + self.α_2) * ( self.u_DM(qb_FR)- self.cost_DM(qb_FR) ) 
            
            labor_CM = self.x_star 
            
            CM_segment = self.U_CM(self.x_star) - self.h(labor_CM)
            
            lifetime_utility = self.n*discount*(DM_segment + CM_segment )
        
        else:
            
            DM_segment = self.DM_utility(z, i, τ) 
            
            labor_CM = self.x_star 
            
            CM_segment = self.n*(self.U_CM(self.x_star) -  self.h(labor_CM) )
            
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
            
            lifetime_utility = self.n*discount*(DM_segment + CM_segment )
        
        else:
            
            DM_segment = self.DM_utility_delta(z, i, τ, delta) 
            
            labor_CM = self.x_star 
            
            CM_segment = self.n*(self.U_CM(self.x_star*delta) -  self.h(labor_CM))
            
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
        
        em_func = lambda ρ: self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(ρ, i, z, τ) ) 
            
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
        tic = time.time()
                
        # MPI setting
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        τ_per_process = len(self.τ_grid) // size
        start_idx = rank * τ_per_process
        end_idx = start_idx + τ_per_process if rank != size-1 else len(self.τ_grid)
        local_τ_grid = self.τ_grid[start_idx:end_idx]
        
        local_zstar = np.zeros(len(local_τ_grid))
        local_istar = local_zstar.copy()
        
        local_ξstar = local_zstar.copy()
        local_Dstar = local_zstar.copy()
        local_qstar = local_zstar.copy()
        
        local_π_firm_star = local_zstar.copy()
        
        local_DM_surplus = local_zstar.copy()

        local_price_mean = local_zstar.copy()
        local_price_sd = local_zstar.copy()
        local_price_cv = local_zstar.copy()
                
        local_markup_mean = local_zstar.copy()
        local_markup_sd = local_zstar.copy()
        local_markup_cv = local_zstar.copy()

        local_mpy_star = local_zstar.copy()
        
        local_w_star = local_zstar.copy()
        local_w_bcw = local_zstar.copy()
        
        local_FFR = local_zstar.copy()
        
        local_credit_gdp = local_zstar.copy()
        local_credit_gdp_bcw = local_zstar.copy()  
        
        for idx_τ, τ in enumerate(local_τ_grid):
            print(f"Process {rank} processing tau = {τ}")
            
            local_zstar[idx_τ], local_istar[idx_τ] = self.solve_z_i(z_guess, i_guess, τ)
            
            local_ξstar[idx_τ] = self.Total_loan_func(local_istar[idx_τ], local_zstar[idx_τ], τ)
            
            local_Dstar[idx_τ] = self.Total_deposit_func(local_istar[idx_τ], local_zstar[idx_τ], τ)
            
            local_qstar[idx_τ] = self.Total_q_func(local_zstar[idx_τ], local_istar[idx_τ], τ)
            
            local_π_firm_star[idx_τ] = self.firm_profit_func(local_zstar[idx_τ], local_istar[idx_τ], τ)
            
            local_DM_surplus[idx_τ] = self.DM_utility(local_zstar[idx_τ], local_istar[idx_τ], τ)
                
            local_w_star[idx_τ] = self.welfare_func(local_zstar[idx_τ], local_istar[idx_τ], τ)
            
            local_w_bcw[idx_τ] = self.welfare_bcw_func(τ)
            
            local_price_mean[idx_τ] = self.SME_stat_func(local_zstar[idx_τ], local_istar[idx_τ], τ)['price_mean']
            
            local_price_sd[idx_τ] = self.SME_stat_func(local_zstar[idx_τ], local_istar[idx_τ], τ)['price_sd']
            
            local_price_cv[idx_τ] = self.SME_stat_func(local_zstar[idx_τ], local_istar[idx_τ], τ)['price_cv']
            
            local_markup_mean[idx_τ] = self.SME_stat_func(local_zstar[idx_τ], local_istar[idx_τ], τ)['markup_mean']
            
            local_markup_sd[idx_τ] = self.SME_stat_func(local_zstar[idx_τ], local_istar[idx_τ], τ)['markup_sd']
            
            local_markup_cv[idx_τ] = self.SME_stat_func(local_zstar[idx_τ], local_istar[idx_τ], τ)['markup_cv']
       
            local_mpy_star[idx_τ] = self.SME_stat_func(local_zstar[idx_τ], local_istar[idx_τ], τ)['mpy']
            
            local_credit_gdp[idx_τ] = self.SME_stat_func(local_zstar[idx_τ], local_istar[idx_τ], τ)['loan_gdp']
            
            local_credit_gdp_bcw[idx_τ] = self.loan_gdp_bcw_func(τ)
            
            local_FFR[idx_τ] = self.i_policy(τ)
        
        # 결과 수집
        if rank == 0:
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
            
        else:
            zstar = None
            istar = None
        
            ξstar = None
            Dstar = None
            qstar = None
        
            π_firm_star = None
        
            DM_surplus = None

            price_mean = None
            price_sd = None
            price_cv = None
                
            markup_mean = None
            markup_sd = None
            markup_cv = None

            mpy_star = None
        
            w_star = None
            w_bcw = None
        
            FFR = None
            
            credit_gdp = None
            credit_gdp_bcw =None
            
            
        comm.Gatherv(local_zstar, [zstar, (np.array([len(local_τ_grid)]*size), None), MPI.DOUBLE], root=0)
        comm.Gatherv(local_istar, [istar, (np.array([len(local_τ_grid)]*size), None), MPI.DOUBLE], root=0)
        comm.Gatherv(local_ξstar, [ξstar, (np.array([len(local_τ_grid)]*size), None), MPI.DOUBLE], root=0)
        comm.Gatherv(local_Dstar, [Dstar, (np.array([len(local_τ_grid)]*size), None), MPI.DOUBLE], root=0)
        comm.Gatherv(local_qstar, [qstar, (np.array([len(local_τ_grid)]*size), None), MPI.DOUBLE], root=0)
        comm.Gatherv(local_π_firm_star, [π_firm_star, (np.array([len(local_τ_grid)]*size), None), MPI.DOUBLE], root=0)
        comm.Gatherv(local_DM_surplus, [DM_surplus, (np.array([len(local_τ_grid)]*size), None), MPI.DOUBLE], root=0)
        comm.Gatherv(local_w_star, [w_star, (np.array([len(local_τ_grid)]*size), None), MPI.DOUBLE], root=0)
        comm.Gatherv(local_w_bcw, [w_bcw, (np.array([len(local_τ_grid)]*size), None), MPI.DOUBLE], root=0)
        comm.Gatherv(local_price_mean, [price_mean, (np.array([len(local_τ_grid)]*size), None), MPI.DOUBLE], root=0)
        comm.Gatherv(local_price_sd, [price_sd, (np.array([len(local_τ_grid)]*size), None), MPI.DOUBLE], root=0)
        comm.Gatherv(local_price_cv, [price_cv, (np.array([len(local_τ_grid)]*size), None), MPI.DOUBLE], root=0)
        comm.Gatherv(local_markup_mean, [markup_mean, (np.array([len(local_τ_grid)]*size), None), MPI.DOUBLE], root=0)
        comm.Gatherv(local_markup_cv, [markup_cv, (np.array([len(local_τ_grid)]*size), None), MPI.DOUBLE], root=0)
        comm.Gatherv(local_markup_sd, [markup_sd, (np.array([len(local_τ_grid)]*size), None), MPI.DOUBLE], root=0)
        comm.Gatherv(local_mpy_star, [mpy_star, (np.array([len(local_τ_grid)]*size), None), MPI.DOUBLE], root=0)
        comm.Gatherv(local_credit_gdp, [credit_gdp, (np.array([len(local_τ_grid)]*size), None), MPI.DOUBLE], root=0)
        comm.Gatherv(local_credit_gdp_bcw, [credit_gdp_bcw, (np.array([len(local_τ_grid)]*size), None), MPI.DOUBLE], root=0)
        comm.Gatherv(local_FFR, [FFR, (np.array([len(local_τ_grid)]*size), None), MPI.DOUBLE], root=0)

        if rank == 0:        

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
        
        return None             

tic = time.time()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    σ_DM = 0.45
    Ubar_CM = 1.90
    n = 0.65
    α_1 = 0.1
    model = baseline_mod(σ_DM=σ_DM, Ubar_CM=Ubar_CM, n=n, α_1=α_1,τ_max = 0.1, τgrid_size = 120, N_reimann=500)
    z_guess = 0.5
    i_guess = 0.01
    params = {
        'model': model,
        'z_guess': z_guess,
        'i_guess': i_guess
    }
else:
    params = None

params = comm.bcast(params, root=0)
model = params['model']
z_guess = params['z_guess']
i_guess = params['i_guess']

result = model.SME_stat(z_guess, i_guess)

if rank == 0:  # rank 0만 결과 처리 및 시각화 수행
    model_hlmw = hlmw.hlmw_mod(σ_DM=σ_DM, Ubar_CM=Ubar_CM, n=n, 
                              α_1=α_1,τ_max = 0.1, τgrid_size = 120, N_reimann=500)
    
    # 결과 처리
    z = result['allocation_grid']['zstar']
    q = result['allocation_grid']['qstar']
    ξ = result['allocation_grid']['ξstar']
    i = result['allocation_grid']['istar']
    DM_surplus = result['allocation_grid']['DM_surplus']
    W = result['allocation_grid']['w_star']
    W_BCW = result['allocation_grid']['w_bcw']

    mpy = result['stat_grid']['mpy_star']
    markup = result['stat_grid']['markup_mean']
    markup_cv = result['stat_grid']['markup_cv']
    loan_gdp = result['stat_grid']['credit_gdp']
    loan_gdp_bcw = result['stat_grid']['credit_gdp_bcw']

    result_hlmw = model_hlmw.SME_stat() # this uses brentq only

    z_hlmw = result_hlmw['allocation_grid']['zstar']
    q_hlmw = result_hlmw['allocation_grid']['qstar']
    W_hlmw = result_hlmw['allocation_grid']['w_star']
    DM_surplus_hlmw = result_hlmw['allocation_grid']['DM_surplus']

    mpy_hlmw = result_hlmw['stat_grid']['mpy_star']
    markup_hlmw = result_hlmw['stat_grid']['markup_mean']
    markup_cv_hlmw = result_hlmw['stat_grid']['markup_cv']

    τ_grid = model.τ_grid # inflation rate grid
    i_grid = model.i_policy(τ_grid)

    font = {'family' : 'serif','weight':'normal',
            'size'   : 10}
    plt.rc('font', **font)

    #plt.style.use(style='default')
    W = result['allocation_grid']['w_star']
    W_hlmw = result_hlmw['allocation_grid']['w_star']
    DW = W - W_hlmw # Baseline vs. HLMW
    DW_2 = W_BCW - W # BCW vs. Baseline
    DW_3 = W_BCW - W_hlmw # BCW vs. HLMW
    
    # 시각화
    plt.figure(facecolor='white')
    plt.plot(i_grid*100, DW, label='Baseline vs. HLMW', color='r')
    plt.axhline(y=0, color="black", linestyle='--', linewidth=1.0)
    plt.ylabel("DW")
    plt.xlabel("Nominal interest rate (%)")
    plt.legend()
    plt.savefig(f"DW_s{model.σ_DM}_ub{model.Ubar_CM}_n{model.n}_α_1{model.α_1}.png")
    plt.show()

toc = time.time()-tic