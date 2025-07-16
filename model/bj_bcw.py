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

#from mpi4py import MPI

class baseline_mod(object):
    """Model primitives and methods"""
    def __init__(self, 
                 β = 0.9804, σ_CM = 1.0, σ_DM = 0.4225,  
                 Abar = 1.0, Ubar_CM = 1.0, Ubar_DM = 1.0, 
                 c = 1.0, τ_b = 0.0, τ_max = 0.1, τgrid_size = 50, 
                 z_min = 1e-2, z_max = 1.0, zgrid_size = 120, 
                 N_reimann = 1000, Tol=1e-12,τ_min=0.0,
                 α_1=0.3, n=0.8, chi = 0.053, i_r = 0.0
                ):
        #Model parameters
        #-------------------------------             # Default parameter settings:
        self.β = β                                   # Discount factor
        
        """Preferences"""
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
        
        #Array of grids on z
        self.z_grid = np.linspace(z_min, z_max, zgrid_size)      
        
        #Array of grids on τ
        self.τ_grid = np.linspace(τ_min, τ_max, τgrid_size) 
        
        # Reimann integral domain partition
        self.N_reimann = N_reimann
        
        
        #############################  New  ###################################
        
        self.chi = chi                              # reserve requirement ratio
        self.i_r = i_r                              # reserve rate
        
        self.Tol = Tol
    
        ####################################################################### 
    
    def z_cutoffs(self, i):
        """Define the four cutoff levels for z as in Lemmata 1-3:
            0 < z_tilde_i < zhat < z_prime < ∞."""
        i_d = self.i_d(i)
        z_hat_id = (self.c/(1-self.σ_DM)*(1+i_d))**((self.σ_DM-1)/self.σ_DM)
        z_prime = z_hat_id*(1/(1-self.σ_DM))**(-(self.σ_DM-1)/self.σ_DM)
        z_tilde = z_hat_id*(1+i)**(-1/self.σ_DM)
        # z_breve = (1-self.c) + z_hat*self.σ_DM
        # z_breve =((1+self.σ_DM*z_tilde)/self.c)**(1-self.σ_DM)
    
        # Sanity check!
        # zcut = np.diff([z_tilde, z_hat])
        if z_tilde <= z_hat_id:
            #print("🥳 Congrats: Your cutoffs are in order!")
            z_cutoffs_str = '\\tilde{z}_{i} < \\hat{z} < z^{\\prime}'
#            display(
#                Math(z_cutoffs_str)
#                )
            z_cutoffs = [z_tilde, z_hat_id, z_prime] 
            #print(['%.3f' % x for x in z_cutoffs])
#        else:
#            print("💩 You have a problem with your cutoff ordering!")
            
        return z_tilde, z_hat_id, z_prime 
    
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
    
    def R(self, p_grid, Rex, DropFakes=True):
        """Find convex hull of graph(Rex) to get R - See Appendix B.3
        NOW RETURNS: R_grid, R_fit, lottery_supports, lottery_payoffs
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
        lottery_supports = v_intersect_graph[idz, 0] if len(idz) > 0 else np.array([]).reshape(0, 2)
        lottery_payoffs = v_intersect_graph[idz, 1] if len(idz) > 0 else np.array([]).reshape(0, 2)
    
        # Step 6: Interpolate to approximate R
        R_fit = self.InterpFun1d(v_intersect_graph[:, 0],
                            v_intersect_graph[:, 1])
    
        # Step 7: Eliminate Fake Lotteries (below-tolerance numerical imprecisions)
        if DropFakes and lottery_supports.size > 0:
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
            if len(selector) > 0:
                lottery_supports = lottery_supports[selector, :]
                lottery_payoffs = lottery_payoffs[selector, :]
            else:
                lottery_supports = np.array([]).reshape(0, 2)
                lottery_payoffs = np.array([]).reshape(0, 2)
        
        # Step 8: Store R as evaluated on finite set p_grid
        R_grid = R_fit(p_grid)
        return R_grid, R_fit, lottery_supports, lottery_payoffs
    
    def get_lottery_probabilities(self, ρ, lottery_supports, lottery_payoffs, R_fit):
        """Calculate π1,ρ and π2,ρ for a specific price ρ
        Solves the system:
        π1*R(ρ1) + π2*R(ρ2) = R(ρ)
        π1*ρ1 + π2*ρ2 = ρ
        where π2 = 1 - π1
        """
        π1, π2 = 1.0, 0.0  # Default: no lottery
        
        if lottery_supports.size == 0:
            return π1, π2
            
        # Check if ρ is in any lottery segment
        for i in range(lottery_supports.shape[0]):
            ρ1, ρ2 = lottery_supports[i, :]
            if ρ1 <= ρ <= ρ2:
                R1, R2 = lottery_payoffs[i, :]
                R_ρ = R_fit(ρ)
                
                # Set up system: A * [π1, π2]' = b
                A = np.array([[R1, R2], [ρ1, ρ2]])
                b = np.array([R_ρ, ρ])
                
                try:
                    π1, π2 = np.linalg.solve(A, b)
                    break
                except np.linalg.LinAlgError:
                    # Singular matrix, keep default values
                    continue
                    
        return π1, π2
    
    def omega_weighting_function(self, ρ_grid, lottery_supports, lottery_payoffs, R_fit):
        """Weighting function ω(ρℓ,ρ, z, s) from equation (0.1)
        For each ρ, returns [ω1, ω2] where:
        ωi = 1{|π1,ρ - π2,ρ| > 0} × πi,ρ + 1{|π1,ρ - π2,ρ| = 0}
        """
        omega1 = np.ones_like(ρ_grid)
        omega2 = np.ones_like(ρ_grid)
        
        for j, ρ in enumerate(ρ_grid):
            π1, π2 = self.get_lottery_probabilities(ρ, lottery_supports, lottery_payoffs, R_fit)
            
            if np.abs(π1) != 1.0:  # There is a meaningful lottery
                omega1[j] = π1  # Weight for ρ1 price
                omega2[j] = π2  # Weight for ρ2 price  
            else:
                omega1[j] = 1.0  # No lottery case
                omega2[j] = 0.0  # No lottery case 
                
        return omega1, omega2
        
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
        τ1 = (self.α_1+self.α_2)*self.τb 
        τ2 = τ - τ1
        return τ2    
        
    def i_d(self, i):
        """
        deposit rate
        """
        i_d = i*(1-self.chi)+self.i_r*self.chi 
        return i_d
    
    def i_policy(self, τ):
        """
        Nominal policy interest rate in steady state
        """
        γ = self.GrossInflation(τ)
        policy_rate = γ / self.β - 1.0
        return policy_rate    

############################# New function ####################################
    def ρ_hat_func(self, z):
        """price that max profits with constrained buyer"""
        ρ_hat_value = z**( self.σ_DM / ( self.σ_DM - 1.0 ) ) 
        return ρ_hat_value
    
    def ρ_hat_id_func(self, z, i):
        """cut-off price where constrained buyer neither borrow nor save """
        i_d = self.i_d(i)
        ρ_hat_value = self.ρ_hat_func(z) * ( (1.0 + i_d)**(1.0/(self.σ_DM-1.0)) ) 
        return ρ_hat_value

    def ρ_tilde_func(self, z, i):
        """cut-off price where constrained buyer will borrow"""
        ρ_tilde_value = self.ρ_hat_func(z) * ( (1.0 + i)**(1.0/(self.σ_DM-1.0)) ) 
        return ρ_tilde_value
       
    def q_demand_func(self, ρ, i, z):
        """DM goods demand function"""
        if self.i_r < i:
            """Reserve requirement binds"""
            i_d = self.i_d(i)

            ρ_hat_id = self.ρ_hat_id_func(z,i_d)

            ρ_tilde = self.ρ_tilde_func(z, i)

            if ρ <= ρ_tilde:
                """liquidity constrained and borrowing unconstrained buyer"""
                q = ( ρ*(1.0 + i ) )**(-1.0/self.σ_DM)

            elif ρ_tilde < ρ <= ρ_hat_id:
                """liquidity constrained and neither borrow nor save"""
                q = z / ρ

            else:
                """liquidity unconstrained and save"""
                q = ( ρ*(1.0 + i_d ) )**(-1.0/self.σ_DM)
        else:
            """
            Reserve requirement slacks, only one type of buyers
            In this case, z has to be re-solved later (and thus the pricing cut-offs) using i=i_r and i_d = i
            """
            q = ( ρ*(1.0 + i ) )**(-1.0/self.σ_DM)
            
        return q

    def ξ_demand_func(self, ρ, i, z):
        """DM loans demand function"""
        if self.i_r < i:
            """Reserve requirement binds"""
            ρ_hat_id = self.ρ_hat_id_func(z, i)

            ρ_tilde = self.ρ_tilde_func(z, i)

            if ρ <= ρ_tilde:
                """liquidity constrained and borrowing unconstrained buyer"""
                loan = ρ * ( ( ρ*(1.0 + i ) )**(-1.0/self.σ_DM) ) - z

            elif ρ_tilde < ρ <= ρ_hat_id:
                """liquidity constrained and neither borrow nor save"""
                loan = 0.0

            else:
                """liquidity unconstrained and save"""
                loan = 0.0
        else:
            """
            Reserve requirement slacks, only one type of buyers
            In this case, z has to be re-solved later (and thus the pricing cut-offs) using i=i_r and i_d = i
            """
            loan = ρ * ( ( ρ*(1.0 + i ) )**(-1.0/self.σ_DM) ) - z
 
        return loan
        
    def δ_demand_func(self, ρ, i, z):
        """DM deposits supply function coming from the active buyers"""
        if self.i_r < i:
            """Reserve requirement binds"""
            i_d = self.i_d(i)

            ρ_hat_id = self.ρ_hat_id_func(z,i)

            ρ_tilde = self.ρ_tilde_func(z, i)

            if ρ <= ρ_tilde:
                """liquidity constrained and zero deposit buyer"""
                deposit = 0.0

            elif ρ_tilde < ρ <= ρ_hat_id:
                """liquidity constrained and zero deposit buyer"""
                deposit = 0.0

            else:
                """liquidity unconstrained and deposit buyer"""
                deposit = z - ρ*( ρ*(1.0 + i_d ) )**(-1.0/self.σ_DM)
        else:
            """Reserve requirement slacks
            There is only one type of buyers, no active buyers deposit
            """
            deposit = 0.0
        return deposit
            
    def Total_loan_func(self, i, z, τ):  
        if τ > self.β-1.0:
            if self.i_r < i:
                """Reserve requirement binds"""
                ρ_grid = self.support_grid_func(z, i, τ)
                pdf_grid = self.dF_normalization_func(z, i, τ)
                em_func = self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(i, z, τ) )
                em = em_func
                
                # Get lottery information
                ρ_range = self.support_grid_func(z, i, τ)
                Rex = self.R_ex(ρ_range, i, z)
                R_grid, R_fit, lottery_supports, lottery_payoffs = self.R(ρ_range, Rex)
                
                # Calculate expected loans with lottery weighting
                loan_expost_weighted = np.zeros_like(ρ_grid)
                
                for j, ρ in enumerate(ρ_grid):
                    π1, π2 = self.get_lottery_probabilities(ρ, lottery_supports, lottery_payoffs, R_fit)
                    
                    if np.abs(π1) != 1.0:  # Lottery case
                        # Find lottery segment prices
                        ρ1, ρ2 = ρ, ρ  # Default
                        for k in range(lottery_supports.shape[0]):
                            if lottery_supports[k, 0] <= ρ <= lottery_supports[k, 1]:
                                ρ1, ρ2 = lottery_supports[k, :]
                                break
                        print(π1)
                        print(π2)
                        # Expected loans: π1×ξ(ρ1) + π2×ξ(ρ2)
                        loan1 = self.ξ_demand_func(ρ1, i, z)
                        loan2 = self.ξ_demand_func(ρ2, i, z)
                        loan_expost_weighted[j] = π1 * loan1 + π2 * loan2
                    else:  # No lottery case
                        loan_expost_weighted[j] = self.ξ_demand_func(ρ, i, z)
                
                integrand_values = em * loan_expost_weighted * pdf_grid        
                total_loan = self.n * np.trapz(integrand_values, ρ_grid)  
            else:
                """
                Reserve requirement slakcs
                In this case, z has to be re-solved later (and thus the pricing cut-offs) using i=i_r and i_d = i
                """
                ρ_grid = self.support_grid_func(z, i, τ)
                
                pdf_grid = self.dF_normalization_func(z, i, τ)

                em =self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(i, z, τ) )

                loan_expost = np.array( [self.ξ_demand_func(ρ, i, z) for ρ in ρ_grid] )

                integrand_values = em * loan_expost * pdf_grid        
                
                total_loan = self.n*np.trapz(integrand_values, ρ_grid)  
        else:
            total_loan = 0.0

        return total_loan
    
    def Total_deposit_func(self, i, z, τ):  
        if τ > self.β-1.0:
            if self.i_r <= i:
                """Reserve requirement binds"""
                i_d = self.i_d(i)
                
                ρ_grid = self.support_grid_func(z, i, τ)
                pdf_grid = self.dF_normalization_func(z, i, τ)
                em_func = self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(i, z, τ) )
                em = em_func
                
                # Get lottery information
                ρ_range = self.support_grid_func(z, i, τ)
                Rex = self.R_ex(ρ_range, i, z)
                R_grid, R_fit, lottery_supports, lottery_payoffs = self.R(ρ_range, Rex)
                
                # Calculate expected deposits with lottery weighting
                deposit_expost_weighted = np.zeros_like(ρ_grid)
                
                for j, ρ in enumerate(ρ_grid):
                    π1, π2 = self.get_lottery_probabilities(ρ, lottery_supports, lottery_payoffs, R_fit)
                    
                    if np.abs(π1) != 1.0:  # Lottery case
                        # Find lottery segment prices
                        ρ1, ρ2 = ρ, ρ  # Default
                        for k in range(lottery_supports.shape[0]):
                            if lottery_supports[k, 0] <= ρ <= lottery_supports[k, 1]:
                                ρ1, ρ2 = lottery_supports[k, :]
                                break
                        
                        # Expected deposits: π1×δ(ρ1) + π2×δ(ρ2)
                        deposit1 = self.δ_demand_func(ρ1, i, z)
                        deposit2 = self.δ_demand_func(ρ2, i, z)
                        deposit_expost_weighted[j] = π1 * deposit1 + π2 * deposit2
                    else:  # No lottery case
                        deposit_expost_weighted[j] = self.δ_demand_func(ρ, i, z)
                
                integrand_values = em * deposit_expost_weighted * pdf_grid        
                # Total deposits from inactive buyers and active depositors
                total_deposit = (1-self.n) * z + self.n * np.trapz(integrand_values, ρ_grid)
            else:
                """Reserve requirement slacks"""
                total_deposit = (1-self.n) * z 
        else:
            total_deposit = 0.0
            
        return total_deposit 
    
    def i_rate_obj(self, i, z, τ):
        """
        Loan rate objective, which is equivalent to $\bar{i}_{r}$
        """
        
        # Total deposit
        LHS = (1-self.chi)*self.Total_deposit_func(i,z,τ) 
        
        # Total loan
        RHS = self.Total_loan_func(i,z,τ) 
        
        net = LHS - RHS
 
        return net
    
    ##-----------------DM goods expenditure and sellers' profit margin------------##
    def q_expenditure(self, ρ, i, z):
        """expenditure := ρq"""    
        if self.i_r < i:
            """Reserve requirement bind"""
            i_d = self.i_d(i)
            ρ_hat_id = self.ρ_hat_id_func(z,i)
            ρ_tilde = self.ρ_tilde_func(z, i)

            if ρ <= ρ_tilde:
                """liquidity constrained and borrowing unconstrained buyer"""
                ρq = ρ * ( ( ρ*(1.0 + i ) )**(-1.0/self.σ_DM) )

            elif ρ_tilde < ρ <= ρ_hat_id:
                """liquidity constrained and neither borrow nor save"""
                ρq = z

            else:
                """liquidity unconstrained and save"""
                ρq = ρ *( ρ*(1.0 + i_d ) )**(-1.0/self.σ_DM)
        else:
            """
            Reserve requirement slacks
            In this case, z has to be re-solved later (and thus the pricing cut-offs) using i=i_r and i_d = i
            """
            ρq = ρ * ( ( ρ*(1.0 + i ) )**(-1.0/self.σ_DM) )
        return ρq
    
    def R_ex_scalar(self, ρ, i, z):
        """seller's ex-post profit per customer served"""
        if self.i_r < i:
            """Reserve requirement bind"""
            i_d = self.i_d(i)
            ρ_hat_id = self.ρ_hat_id_func(z,i)
            ρ_tilde = self.ρ_tilde_func(z, i)
            if ρ <= ρ_tilde:
                """profit margin from serving
                liquidity constrained and borrowing unconstrained buyer
                """
                qb = ( ρ*(1.0 + i ) )**(-1.0/self.σ_DM)
                val = qb * ( ρ - self.c ) 

            elif ρ_tilde < ρ <= ρ_hat_id:
                """profit margin from serving 
                liquidity constrained and neither borrow nor save
                """
                qb = z / ρ
                val = qb * ( ρ - self.c )

            else:
                """liquidity unconstrained buyer and save"""
                qb =  ( ρ*(1.0 + i_d ) )**(-1.0/self.σ_DM)
                val = qb * ( ρ - self.c )
        else:
            """
            Reserve requirement slakcs
            In this case, z has to be re-solved later (and thus the pricing cut-offs) using i=i_r and i_d = i
            """
            qb = ( ρ*(1.0 + i ) )**(-1.0/self.σ_DM)
            val = qb * ( ρ - self.c ) 
        return val
    
    def R_ex(self, ρ, i, z):
        """seller's ex-post profit per customer served.
        ρ is a 1D array, (i,z) are scalars. This function is 
        the vectorized version of R_ex_scalar() above."""
        if self.i_r < i:
            """Reserve requirement bind"""
            
            # Pricing cutoffs
            i_d = self.i_d(i)
            ρ_hat_id = self.ρ_hat_id_func(z,i)
            ρ_tilde = self.ρ_tilde_func(z, i)

            qb = np.zeros(ρ.shape)
            val = qb.copy()

            # liquidity constrained and borrowing unconstrained buyer
            bool = (ρ <= ρ_tilde)
            qb[bool] = (ρ[bool]*(1.0 + i))**(-1.0/self.σ_DM)
            val[bool] = qb[bool] * (ρ[bool] - self.c)

            # liquidity constrained and zero borrowing buyer
            bool = (ρ_tilde < ρ) & (ρ <= ρ_hat_id)
            qb[bool] = z / ρ[bool]
            val[bool] = qb[bool] * (ρ[bool] - self.c)

            # money unconstrained
            bool = (ρ > ρ_hat_id)
            qb[bool] = (ρ[bool]*(1.0 + i_d))**(-1.0/self.σ_DM)
            val[bool] = qb[bool] * (ρ[bool] - self.c)
        else:
            """
            Reserve requirement slakcs
            In this case, z has to be re-solved later (and thus the pricing cut-offs) using i=i_r and i_d = i
            """
            # Pricing cutoffs
            i_d = self.i_r
            i = self.i_r
            ρ_hat_id = self.ρ_hat_id_func(z,i)
            ρ_tilde = self.ρ_tilde_func(z, i)

            qb = np.zeros(ρ.shape)
            val = qb.copy()
            
            ρ_constant = self.c / (1.0 - self.σ_DM)
            
            # liquidity constrained and borrowing unconstrained buyer
            bool = (ρ <=ρ_constant)
            qb[bool] = (ρ[bool]*(1.0 + i))**(-1.0/self.σ_DM)
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
        i_d = self.i_d(i)
        qb = ( ρ*(1.0 + i_d ) )**(-1.0/self.σ_DM)
        val = qb * ( ρ - self.c )
            
        return val
    
    ##-----------------bounds on DM goods prices------------##
    def ρ_max_func(self, z,i):
        """Equilibrium upper bound on the support of F: ρ_{max}"""
        
        ρ_constant = self.c / (1.0 - self.σ_DM)
        
        if i >0:
            """Away from the Friedman rule"""
            if self.i_r < i:
                """Reserve requirement binds"""
                i_d = self.i_d(i)
                ρ_hat_id = self.ρ_hat_id_func(z,i)
#                ρ_constant = self.c / (1.0 - self.σ_DM)

                z_tilde, z_hat_id, z_prime =self.z_cutoffs(i)

                if z>z_hat_id:
                    ρ_ub = ρ_constant
                elif z>z_tilde and z<z_hat_id: 
                    ρ_ub = ρ_hat_id
                else:
                    ρ_ub = ρ_constant
            else:
                
                ρ_ub = ρ_constant
        elif i <0:
            ρ_ub = ρ_constant
        else:
            """At the Friedman rule, price degenerate at marginal cost"""
            ρ_ub = self.c
        return ρ_ub
    
    def ρ_min_func(self, z, i, τ):
        """Equal expected profit condition to back out the lower bound
        on the support of F given R(ρ_{max})
        """ 
        if τ > self.β-1.0:
            """Away from the Friedman rule"""
            i_d = self.i_d(i)
            ρ_max = self.ρ_max_func(z,i)

            ρ_range= np.linspace(self.c, ρ_max, self.N_reimann)  

            ## Temporarily, start by setting rho_max and begin with c at the minimum.

            Rex = self.R_ex(ρ_range, i, z)

            R_grid,R_fit, lottery_supports, lottery_payoffs=self.R(ρ_range, Rex, DropFakes=True)

            noisy_search = self.α_1  / ( self.α_1 + 2.0 * self.α_2 ) 

            LHS = lambda ρ: self.q_demand_func(ρ, i, z) * ( ρ - self.c )

            RHS = noisy_search * R_fit(ρ_max)  

            vals_obj = lambda ρ: RHS - LHS(ρ)

            ρ_lb = brentq(vals_obj, self.c, ρ_max)
        else:
            """At the Friedman rule, price degenerate at marginal cost"""
            ρ_lb = self.c
        return ρ_lb
    
    ##-----------DM goods price distribution: CDF and pdf in terms of aggregate variables-------##
    def F_func(self, i, z, τ):
        """Posted price distribution (CDF): F(ρ, i, z)
        Note: i := i(ρ, z)
        """
        
        ρ_max = self.ρ_max_func(z,i)
        
        ρ_range=self.support_grid_func(z, i, τ)
        
        Rex = self.R_ex(ρ_range, i, z)
        
        R_grid,R_fit, lottery_supports, lottery_payoffs=self.R(ρ_range, Rex, DropFakes=True)

        noisy_search = self.α_1 / ( 2.0 * self.α_2 ) 
        
        R_max = R_fit(ρ_max)
        
        R =R_grid
        
        CDF_value =  1.0 - noisy_search * ( ( R_max / R ) - 1.0 ) 
        
        return CDF_value    
    
    def dF_func(self, i, z, τ):
        """Density of the price distribution (PDF): dF(ρ, i, z)/dρ"""
        
        ρ_max = self.ρ_max_func(z,i)
        
        ρ_range=self.support_grid_func(z, i, τ)
        
        noisy_search = self.α_1 / ( 2.0 * self.α_2 ) 
        
        Rex = self.R_ex(ρ_range, i, z)
        
        R_grid,R_fit, lottery_supports, lottery_payoffs=self.R(ρ_range, Rex, DropFakes=True)
    
        R_max =R_fit(ρ_max)
        
        R =R_grid
        
        dR_dρ = R_fit.derivative()(ρ_range)
        
        pdf_value = noisy_search*( ( R_max / ( R**2.0 ) ) * dR_dρ )
        
        return pdf_value
    
    def support_grid_func(self, z, i, τ):
        a = self.ρ_min_func(z, i, τ)
        b = self.ρ_max_func(z,i)       
        supports = np.linspace(a, b, self.N_reimann)
        return supports
    
    def dF_normalization_func(self, z, i, τ):
        ρ_grid = self.support_grid_func(z, i, τ)
        dF = self.dF_func(i, z, τ)
        width = (ρ_grid[-1] - ρ_grid[0]) / (len(ρ_grid)-1)
        dF_sum = np.sum(width * dF)
        dF_nor = dF / dF_sum
        return dF_nor 

    def money_euler_rhs(self, z, i, τ):
        """Fixed money Euler RHS with compound lottery weighting
        Implements equation (0.2) from the technical note
        """
        if τ > self.β - 1.0:
            if self.i_r < i:
                # Get pricing information with lottery details
                ρ_hat_id = self.ρ_hat_id_func(z, i)
                ρ_grid = self.support_grid_func(z, i, τ)
                dF_grid = self.dF_normalization_func(z, i, τ)
                
                # Get lottery information
                ρ_range = self.support_grid_func(z, i, τ)
                Rex = self.R_ex(ρ_range, i, z)
                R_grid, R_fit, lottery_supports, lottery_payoffs = self.R(ρ_range, Rex)
                
                # Calculate weighting function - returns [ω1, ω2]
                omega1, omega2 = self.omega_weighting_function(ρ_grid, lottery_supports, lottery_payoffs, R_fit)
                
                # Find threshold index
                ρ_hat_index = np.argmax(ρ_grid >= ρ_hat_id)
                if ρ_hat_index == 0:
                    ρ_hat_index = -1
                
                # Extensive margin
                em_func = self.α_1 + 2.0*self.α_2*(1.0 - self.F_func(i, z, τ))
                em = em_func[:ρ_hat_index]
                
                # Liquidity premium with compound lottery weighting
                # For each ρ, we sum over ℓ ∈ {1,2}: ∑ω(ρℓ,ρ) × [function of ρℓ,ρ]
                liquidity_premium_weighted = np.zeros(len(ρ_grid[:ρ_hat_index]))
                
                for j, ρ in enumerate(ρ_grid[:ρ_hat_index]):
                    π1, π2 = self.get_lottery_probabilities(ρ, lottery_supports, lottery_payoffs, R_fit)
    
                    if np.abs(π1) != 1.0:  # Lottery case
                    
                        # Find the lottery segment and corresponding prices
                        ρ1, ρ2 = ρ, ρ  # Default to same price
                        for k in range(lottery_supports.shape[0]):
                            if lottery_supports[k, 0] <= ρ <= lottery_supports[k, 1]:
                                ρ1, ρ2 = lottery_supports[k, :]
                                break
                        
                        # Weighted sum: ω1×f(ρ1) + ω2×f(ρ2)
                        lp1 = (self.mu_DM(self.q_demand_func(ρ1, i, z)) / ρ1) - 1.0
                        lp2 = (self.mu_DM(self.q_demand_func(ρ2, i, z)) / ρ2) - 1.0
                        
                        liquidity_premium_weighted[j] = π1 * lp1 + π2 * lp2
                    else:  # No lottery case
                        liquidity_premium_weighted[j] = (self.mu_DM(self.q_demand_func(ρ, i, z)) / ρ) - 1.0
                
                mu_buyer = em * liquidity_premium_weighted * dF_grid[:ρ_hat_index]
                
                # Deposit terms
                i_d = self.i_d(i)
                mu_depositor_1 = (1-self.n) * i_d
                
                # Unconstrained buyers with compound lottery weighting
                em_deposit = em_func[ρ_hat_index:]
                deposit_premium_weighted = np.zeros(len(ρ_grid[ρ_hat_index:]))
                
                for j, ρ in enumerate(ρ_grid[ρ_hat_index:]):
                    π1, π2 = self.get_lottery_probabilities(ρ, lottery_supports, lottery_payoffs, R_fit)
                    
                    if np.abs(π1) != 1.0:  # Lottery case
                        # For deposit premium, it's just i_d regardless of price
                        deposit_premium_weighted[j] = π1 * i_d + π2 * i_d  # = i_d
                       
                    else:  # No lottery case
                        deposit_premium_weighted[j] = i_d

                mu_depositor_2 = em_deposit * deposit_premium_weighted * dF_grid[ρ_hat_index:]
                
                value = mu_depositor_1 + self.n*(np.trapz(mu_buyer, ρ_grid[:ρ_hat_index]) + 
                                               np.trapz(mu_depositor_2, ρ_grid[ρ_hat_index:]))
                
            else:
                i = self.i_r
                value = i
        else:
            # At Friedman rule
            value = z**(-self.σ_DM) - 1.0
            
        return value
    
    def money_euler_obj(self, z, i, τ):
        """Fixed money Euler objective using compound lottery weighting"""
        LHS = self.i_policy(τ)
        RHS = self.money_euler_rhs(z, i, τ)
        net = LHS - RHS
        return net
    
    def system_func(self, initial_guess, τ):
        """Fixed system function with compound lottery implementation"""
        z = initial_guess[0]
        i = initial_guess[1]
        
        z_obj = self.money_euler_obj(z, i, τ)
        i_obj = self.i_rate_obj(i, z, τ)  # This can remain unchanged
        
        return [z_obj, i_obj]
    
    def solve_z_i(self, z_guess, i_guess, τ):
        """Solve for equilibrium with compound lottery fixes"""
        if τ > self.β - 1.0:
            x0 = [z_guess, i_guess]
            x = fsolve(self.system_func, x0, xtol=1e-5, args=(τ,), full_output=False)
            
            z = x[0]
            i = x[1]
            
            if self.i_r < i:
                i = i
            else:
                i = self.i_r
                z = None  # Need to resolve this case
        else:
            z = 1
            i = 0.0
            
        return z, i                      

    ##------------------SME and Stat---------------------------------##
    def Total_q_func(self, z, i, τ):
        """Total quantity with lottery probability weighting"""
        if τ > self.β-1.0:
            """Away from the Friedman rule"""
            ρ_grid = self.support_grid_func(z, i, τ)
            pdf_grid = self.dF_normalization_func(z, i, τ)
            em_func = self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(i, z, τ) )
            em = em_func
            
            # Get lottery information
            ρ_range = self.support_grid_func(z, i, τ)
            Rex = self.R_ex(ρ_range, i, z)
            R_grid, R_fit, lottery_supports, lottery_payoffs = self.R(ρ_range, Rex)
            
            # Calculate expected quantity with lottery weighting
            q_expost_weighted = np.zeros_like(ρ_grid)
            
            for j, ρ in enumerate(ρ_grid):
                π1, π2 = self.get_lottery_probabilities(ρ, lottery_supports, lottery_payoffs, R_fit)
                
                if np.abs(π1) != 1.0:  # Lottery case
                    # Find lottery segment prices
                    ρ1, ρ2 = ρ, ρ  # Default
                    for k in range(lottery_supports.shape[0]):
                        if lottery_supports[k, 0] <= ρ <= lottery_supports[k, 1]:
                            ρ1, ρ2 = lottery_supports[k, :]
                            break
                    
                    # Expected quantity: π1×q(ρ1) + π2×q(ρ2)
                    q1 = self.q_demand_func(ρ1, i, z)
                    q2 = self.q_demand_func(ρ2, i, z)
                    q_expost_weighted[j] = π1 * q1 + π2 * q2
                else:  # No lottery case
                    q_expost_weighted[j] = self.q_demand_func(ρ, i, z)
            
            integrand_values = em * q_expost_weighted * pdf_grid
            total_q = self.n * np.trapz(integrand_values, ρ_grid)
        else:
            """At the Friedman rule"""
            q_FR = z/self.c
            total_q = self.n * q_FR
            
        return total_q
    
    def Total_ρq_func(self, z, i, τ):
        if τ > self.β - 1.0:
            """Away from the Friedman rule"""
            ρ_grid = self.support_grid_func(z, i, τ)
            pdf_grid = self.dF_normalization_func(z, i, τ)
            em_func = self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(i, z, τ) )
            em = em_func
            
            # Get lottery information
            ρ_range = self.support_grid_func(z, i, τ)
            Rex = self.R_ex(ρ_range, i, z)
            R_grid, R_fit, lottery_supports, lottery_payoffs = self.R(ρ_range, Rex)
            
            # Calculate expected expenditure with lottery weighting
            ρq_expost_weighted = np.zeros_like(ρ_grid)
            
            for j, ρ in enumerate(ρ_grid):
                π1, π2 = self.get_lottery_probabilities(ρ, lottery_supports, lottery_payoffs, R_fit)
                
                if np.abs(π1) != 1.0:  # Lottery case
                    # Find lottery segment prices
                    ρ1, ρ2 = ρ, ρ  # Default
                    for k in range(lottery_supports.shape[0]):
                        if lottery_supports[k, 0] <= ρ <= lottery_supports[k, 1]:
                            ρ1, ρ2 = lottery_supports[k, :]
                            break
                    
                    # Expected expenditure: π1×(ρ1×q1) + π2×(ρ2×q2)
                    exp1 = self.q_expenditure(ρ1, i, z)
                    exp2 = self.q_expenditure(ρ2, i, z)
                    ρq_expost_weighted[j] = π1 * exp1 + π2 * exp2
                else:  # No lottery case
                    ρq_expost_weighted[j] = self.q_expenditure(ρ, i, z)
            
            integrand_values = em * ρq_expost_weighted * pdf_grid
            total_ρq = self.n * np.trapz(integrand_values, ρ_grid)
        else:
            """At the Friedman rule"""
            # ρ = c
            q_FR = z/self.c
            
            total_ρq = self.n*(self.c*q_FR)
        return total_ρq    
    
    def firm_profit_func(self, z, i, τ):
        if τ>self.β-1.0:
            """Away from the Friedman rule"""
            ρ_grid = self.support_grid_func(z, i, τ)

            pdf_grid = self.dF_normalization_func(z, i, τ)

            em_func =self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(i, z, τ) )

            em = em_func

            ρ_range=self.support_grid_func(z, i, τ)

            Rex = self.R_ex(ρ_range, i, z)

            R_grid,R_fit, lottery_supports, lottery_payoffs=self.R(ρ_range, Rex, DropFakes=True)

            profit_margin = np.array([R_fit(ρ) for ρ in ρ_grid])

            integrand_values = em * profit_margin * pdf_grid

            firm_profit = np.trapz(integrand_values, ρ_grid)      
        else:
            """At the Friedman rule"""
            firm_profit = 0.0
        return firm_profit
    
    def markup_func(self, z, i, τ):
        """Markup calculation with lottery probability weighting"""
        if τ > self.β-1.0:
            """Away from the Friedman rule"""
            ρ_grid = self.support_grid_func(z, i, τ)
            pdf_grid = self.dF_normalization_func(z, i, τ)
            
            # Get lottery information
            ρ_range = self.support_grid_func(z, i, τ)
            Rex = self.R_ex(ρ_range, i, z)
            R_grid, R_fit, lottery_supports, lottery_payoffs = self.R(ρ_range, Rex)
            
            # Calculate expected markup with lottery weighting
            markup_expost_weighted = np.zeros_like(ρ_grid)
            
            for j, ρ in enumerate(ρ_grid):
                π1, π2 = self.get_lottery_probabilities(ρ, lottery_supports, lottery_payoffs, R_fit)
                
                if np.abs(π1) != 1.0:  # Lottery case
                    # Find lottery segment prices
                    ρ1, ρ2 = ρ, ρ  # Default
                    for k in range(lottery_supports.shape[0]):
                        if lottery_supports[k, 0] <= ρ <= lottery_supports[k, 1]:
                            ρ1, ρ2 = lottery_supports[k, :]
                            break
                    
                    # Expected markup: π1×(ρ1/c) + π2×(ρ2/c)
                    markup1 = ρ1 / self.c
                    markup2 = ρ2 / self.c
                    markup_expost_weighted[j] = π1 * markup1 + π2 * markup2
                else:  # No lottery case
                    markup_expost_weighted[j] = ρ / self.c
            
            q_share = self.n * self.Total_q_func(z, i, τ) / (self.Total_q_func(z, i, τ) + self.x_star)
            x_share = self.n * self.x_star / (self.Total_q_func(z, i, τ) + self.x_star)
            nor_share = q_share / x_share
            
            markup = np.trapz(nor_share * (markup_expost_weighted * pdf_grid), ρ_grid) + 1.0
        else:
            """At the Friedman rule"""
            # CM (gross) price markup is one
            markup = 1.0
        return markup
    
    def DM_utility(self, z, i, τ):
        """
        DM utility with compound lottery weighting
        """        
        if τ>self.β-1.0:
            """Away from the Friedman rule"""
            ρ_grid = self.support_grid_func(z, i, τ)
            pdf_grid = self.dF_normalization_func(z, i, τ)
            em_func =self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(i, z, τ) )
            em = em_func
            
            # Get lottery information
            ρ_range = self.support_grid_func(z, i, τ)
            Rex = self.R_ex(ρ_range, i, z)
            R_grid, R_fit, lottery_supports, lottery_payoffs = self.R(ρ_range, Rex)
            
            # Calculate expected utility with compound lottery weighting
            expost_utility_weighted = np.zeros_like(ρ_grid)
            
            for j, ρ in enumerate(ρ_grid):
                π1, π2 = self.get_lottery_probabilities(ρ, lottery_supports, lottery_payoffs, R_fit)
                
                if np.abs(π1) != 1.0:  # Lottery case
                    # Find the lottery segment and corresponding prices
                    ρ1, ρ2 = ρ, ρ  # Default to same price
                    for k in range(lottery_supports.shape[0]):
                        if lottery_supports[k, 0] <= ρ <= lottery_supports[k, 1]:
                            ρ1, ρ2 = lottery_supports[k, :]
                            break
                    
                    # Expected utility: π1×[u(q1) - c(q1)] + π2×[u(q2) - c(q2)]
                    utility1 = self.u_DM(self.q_demand_func(ρ1,i,z)) - self.cost_DM(self.q_demand_func(ρ1,i,z))
                    utility2 = self.u_DM(self.q_demand_func(ρ2,i,z)) - self.cost_DM(self.q_demand_func(ρ2,i,z))
                    expost_utility_weighted[j] = π1 * utility1 + π2 * utility2
                else:  # No lottery case
                    expost_utility_weighted[j] = self.u_DM(self.q_demand_func(ρ,i,z)) - self.cost_DM(self.q_demand_func(ρ,i,z))
            
            utility = self.n*np.trapz(em * expost_utility_weighted * pdf_grid, ρ_grid)
        else:
            qb_FR = z/self.c
            utility = self.n*(self.α_1 + self.α_2) * ( self.u_DM(qb_FR)- self.cost_DM(qb_FR) ) 
        return utility    
    
    def DM_utility_delta(self, z, i, τ, delta):
        """
        DM utility change by delta with compound lottery weighting
        """        
        if τ>self.β-1.0:
            """Away from the Friedman rule"""
            ρ_grid = self.support_grid_func(z, i, τ)
            pdf_grid = self.dF_normalization_func(z, i, τ)
            em_func =self.α_1 + 2.0*self.α_2*( 1.0 - self.F_func(i, z, τ) )
            em = em_func
            
            # Get lottery information
            ρ_range = self.support_grid_func(z, i, τ)
            Rex = self.R_ex(ρ_range, i, z)
            R_grid, R_fit, lottery_supports, lottery_payoffs = self.R(ρ_range, Rex)
            
            # Calculate expected utility with compound lottery weighting
            expost_utility_weighted = np.zeros_like(ρ_grid)
            
            for j, ρ in enumerate(ρ_grid):
                π1, π2 = self.get_lottery_probabilities(ρ, lottery_supports, lottery_payoffs, R_fit)
                
                if np.abs(π1) != 1.0:  # Lottery case
                    # Find the lottery segment and corresponding prices
                    ρ1, ρ2 = ρ, ρ  # Default to same price
                    for k in range(lottery_supports.shape[0]):
                        if lottery_supports[k, 0] <= ρ <= lottery_supports[k, 1]:
                            ρ1, ρ2 = lottery_supports[k, :]
                            break
                    
                    # Expected utility with delta: π1×[u(q1*δ) - c(q1)] + π2×[u(q2*δ) - c(q2)]
                    utility1 = self.u_DM(self.q_demand_func(ρ1,i,z)*delta) - self.cost_DM(self.q_demand_func(ρ1,i,z))
                    utility2 = self.u_DM(self.q_demand_func(ρ2,i,z)*delta) - self.cost_DM(self.q_demand_func(ρ2,i,z))
                    expost_utility_weighted[j] = π1 * utility1 + π2 * utility2
                else:  # No lottery case
                    expost_utility_weighted[j] = self.u_DM(self.q_demand_func(ρ,i,z)*delta) - self.cost_DM(self.q_demand_func(ρ,i,z))
            
            utility = self.n*np.trapz(em * expost_utility_weighted * pdf_grid, ρ_grid)
        else:
            qb_FR = z/self.c
            utility = self.n*(self.α_1 + self.α_2) * ( self.u_DM(qb_FR*delta)- self.cost_DM(qb_FR) ) 

        return utility
    
    def welfare_func(self, z, i, τ):
        """Welfare function with compound lottery weighting"""
        discount = ( 1.0 / ( 1.0 - self.β ) )
        if τ == self.β - 1.0:
            
            #allocation under Friedman rule
            qb_FR = z/self.c

            DM_segment = self.n*(self.α_1 + self.α_2) * ( self.u_DM(qb_FR)- self.cost_DM(qb_FR) ) 
            
            labor_CM = self.x_star 
            
            CM_segment = self.U_CM(self.x_star) - self.h(labor_CM)
            
            lifetime_utility = discount*(DM_segment + CM_segment )
        
        else:
            
            DM_segment = self.DM_utility(z, i, τ) 
            
            labor_CM = self.x_star 
            
            CM_segment = (self.U_CM(self.x_star) -  self.h(labor_CM) )
            
            lifetime_utility = discount*(DM_segment + CM_segment )
            
        return lifetime_utility
    
    def welfare_func_delta(self, z, i, τ, delta):
        """Welfare function with delta and compound lottery weighting"""
        discount = ( 1.0 / ( 1.0 - self.β ) )
        if τ == self.β - 1.0:
            
            #allocation under Friedman rule
            qb_FR = z/self.c

            DM_segment = self.n*(self.α_1 + self.α_2) * ( self.u_DM(qb_FR*delta) - self.cost_DM(qb_FR) )
            
            labor_CM = self.x_star 
            
            CM_segment = self.U_CM(self.x_star*delta) - self.h(labor_CM)
            
            lifetime_utility = discount*(DM_segment + CM_segment )
        
        else:
            
            DM_segment = self.DM_utility_delta(z, i, τ, delta) 
            
            labor_CM = self.x_star 
            
            CM_segment = (self.U_CM(self.x_star*delta) -  self.h(labor_CM))
            
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
        total_q_exp = self.n*ρ * (self.α_1 + self.α_2) * self.q_BCW(τ)
        z_bcw = total_q_exp
        loan = self.n*(self.α_1 + self.α_2) * self.ξ_demand_func(ρ, i, z_bcw)
        loan_gdp = loan / (total_q_exp + self.x_star)
        return loan_gdp    
    
    def welfare_bcw_func(self, τ):
        """Economy with perfectly competitive banks and competitive pricing in goods market"""
        discount = 1.0 / ( 1.0 - self.β ) 
        
        labor_CM = self.x_star 
        
        CM_segment = self.U_CM(self.x_star) - self.h(labor_CM)
        
        DM_segment = self.n*(self.α_1 + self.α_2)*( self.u_DM(self.q_BCW(τ)) - self.cost_DM(self.q_BCW(τ)) )
        
        lifetime_utility = discount*(DM_segment + CM_segment )
        
        return lifetime_utility   
    
    def welfare_bcw_func_delta(self, τ, delta):
        """Economy with perfectly competitive banks and competitive pricing in goods market
        for calculating CEV
        """
        discount = 1.0 / ( 1.0 - self.β ) 
        
        labor_CM = self.x_star 
        
        CM_segment = self.U_CM(self.x_star*delta) - self.h(labor_CM)
        
        DM_segment = self.n*(self.α_1+self.α_2)*( self.u_DM(self.q_BCW(τ)*delta) - self.cost_DM(self.q_BCW(τ)) )
        
        lifetime_utility = discount*(DM_segment + CM_segment )
        
        return lifetime_utility     
                                                 
    def SME_stat_func(self, z, i, τ):
        """Statistics calculation with lottery probability weighting"""
        ## calculate aggregate markup
        Y = self.x_star + self.Total_q_func(z, i, τ)
        DM_share = self.Total_q_func(z, i, τ) / Y
        CM_share = self.x_star / Y
        nor_share = DM_share / CM_share
        
        if τ > self.β-1.0:
            ρ_grid = self.support_grid_func(z, i, τ)
            pdf_normalized_grid = self.dF_normalization_func(z, i, τ)
            
            # Get lottery information
            ρ_range = self.support_grid_func(z, i, τ)
            Rex = self.R_ex(ρ_range, i, z)
            R_grid, R_fit, lottery_supports, lottery_payoffs = self.R(ρ_range, Rex)
            
            # Calculate lottery-weighted price statistics
            price_weighted = np.zeros_like(ρ_grid)
            markup_weighted = np.zeros_like(ρ_grid)
            
            for j, ρ in enumerate(ρ_grid):
                π1, π2 = self.get_lottery_probabilities(ρ, lottery_supports, lottery_payoffs, R_fit)
                
                if np.abs(π1) != 1.0:  # Lottery case
                    # Find lottery segment prices
                    ρ1, ρ2 = ρ, ρ  # Default
                    for k in range(lottery_supports.shape[0]):
                        if lottery_supports[k, 0] <= ρ <= lottery_supports[k, 1]:
                            ρ1, ρ2 = lottery_supports[k, :]
                            break
                    
                    # Expected price and markup
                    price_weighted[j] = π1 * ρ1 + π2 * ρ2
                    markup_weighted[j] = π1 * (ρ1 / self.c) + π2 * (ρ2 / self.c)
                else:  # No lottery case
                    price_weighted[j] = ρ
                    markup_weighted[j] = ρ / self.c
            
            # Calculate (lottery-weighted) price dispersion
            price_mean = np.trapz(price_weighted * pdf_normalized_grid, ρ_grid)
            y_1 = pdf_normalized_grid * ((price_weighted - price_mean)**2.0)
            price_sd = np.sqrt(np.trapz(y_1, ρ_grid))
            price_cv = price_sd / price_mean    
            
            # Calculate (lottery-weighted) markup dispersion
            markup_mean = self.markup_func(z, i, τ) 
            y_2 = pdf_normalized_grid * ((markup_weighted - markup_mean)**2.0) * (nor_share**2.0) + 0.0  # CM_dispersion = 0
            markup_sd = np.sqrt(np.trapz(y_2, ρ_grid)) 
            markup_cv = markup_sd / markup_mean
        else:
            price_mean = self.c
            price_sd = 0.0
            price_cv = 0.0
            
            markup_mean = 1.0
            markup_sd = 0.0
            markup_cv = 0.0

        mpy = z / Y
        loan_gdp = self.Total_loan_func(i, z, τ) / Y
        
        stat_result = {
            'price_mean': price_mean,
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
        
        zstar = np.zeros(self.τ_grid.size)
        istar = zstar.copy()
        i_dstar = zstar.copy()
        
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
            
            if istar[idx_τ]>self.i_r: 
                
                i_dstar[idx_τ] = self.i_d(istar[idx_τ])
 
            else:
                i_dstar[idx_τ] = istar[idx_τ]
 
            ξstar[idx_τ] = self.Total_loan_func(istar[idx_τ], zstar[idx_τ], τ)
            
            Dstar[idx_τ] = self.Total_deposit_func(istar[idx_τ], zstar[idx_τ], τ)
            
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
                           'i_dstar': i_dstar, 
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

    def SME_stat_ir(self, z_guess, i_guess, τ, i_r_min=0.0, i_r_max=0.1, i_r_grid_size=50):
        tic = time.time()
    
        i_r_grid = np.linspace(i_r_min, i_r_max, i_r_grid_size)
        
        zstar = np.zeros(i_r_grid.size)
        istar = np.zeros(i_r_grid.size)
        i_dstar = np.zeros(i_r_grid.size)
        i_binding = np.zeros(i_r_grid.size, dtype=bool)  
    
        ξstar = np.zeros(i_r_grid.size)   
        Dstar = np.zeros(i_r_grid.size)    
        reserve = np.zeros(i_r_grid.size)  
        excess_reserve = np.zeros(i_r_grid.size)  
    
        qstar = np.zeros(i_r_grid.size)    
        π_firm_star = np.zeros(i_r_grid.size)  
        w_star = np.zeros(i_r_grid.size)  
    
        original_i_r = self.i_r
    
        for idx, i_r in enumerate(i_r_grid):
            
            self.i_r = i_r
        
            zstar[idx], istar[idx] = self.solve_z_i(z_guess, i_guess, τ)
        
            i_binding[idx] = (i_r < istar[idx])  
            
            
            if self.i_d(istar[idx])>self.i_r: 
                
                i_dstar[idx] = self.i_d(istar[idx])
 
            else:
                i_dstar[idx] = istar[idx]
            
        
            ξstar[idx] = self.Total_loan_func(istar[idx], zstar[idx], τ)
            Dstar[idx] = self.Total_deposit_func(istar[idx], zstar[idx], τ)
            reserve[idx] = self.chi * Dstar[idx]
            excess_reserve[idx] = Dstar[idx] - ξstar[idx] - reserve[idx]
        
            qstar[idx] = self.Total_q_func(zstar[idx], istar[idx], τ)
            π_firm_star[idx] = self.firm_profit_func(zstar[idx], istar[idx], τ)
            w_star[idx] = self.welfare_func(zstar[idx], istar[idx], τ)
            
        # Restore original i_r value
        self.i_r = original_i_r
            
        # Store results
        allocation_grid = {
            'i_r_grid': i_r_grid,
            'i_binding': i_binding,
            'zstar': zstar,
            'istar': istar,
            'i_dstar': i_dstar,
            'ξstar': ξstar,
            'Dstar': Dstar,
            'reserve': reserve,
            'excess_reserve': excess_reserve,
            'qstar': qstar,
            'π_firm_star': π_firm_star,
            'w_star': w_star
        }
    
        toc = time.time() - tic
        print("Elapsed time of solving SME:", toc, "seconds")
    
        return allocation_grid
