# Standard python packages
import numpy as np
from numba import njit

import matplotlib.pyplot as plt

# New packages
import utils
import jacobian as jac
from het_block import het
from simple_block import simple
from solved_block import solved
from two_asset  import *
import nonlinear
import determinacy as det
import estimation as est
import time



# Outputs
# - consumption
# - investment
# - hours
# - wages
# - nominal interest rates
# - price inflation



# Computing the empirical variance and using as weighting matrix for
# matching



# In model with three shocks
# --------------------------
# I can sim the model, including aggregates & micro regression moments
#   for different values of the parameters
# I can produce identification plots showing how aggregate moments
#   change with different parameters
# I can do CMD to estimate parameters


# Smets-Wouters extension
# -----------------------
# Their richer model is Smets-Wouters
# Some question as to how they incorporate these shocks
# Can ask, or just talk that out and do it ourselves


# Which parameters to estimate
# ----------------------------
# - Shock parameters
# - Taylor rule coefficients
# - Phillips curve slope parameters
# - Capital adjustment costs

# Next steps
# ----------
# - Do you want me to test out and generate some results with infeasible
#   variance matrix like before, let that stuff run on the server (or
#   Ulrich's machine)? While I can work up the other application.
# - Or should I just keep on this, do the smets wouters extension, get
#   into SCF, wrap this one up


# Which data to use?
# ------------------
# - Second moments of Smets Wouters aggregate time series
# - Consumption given liquid and illiquid asset values
# - Liquid vs. illiquid wealth shares
# - Shares of wealthy HtM
# - Average MPCs





########################################################################
########################################################################
########################################################################


# Solve Model: Get steady state and Jacobian for some set of parameters, or just Jacobian if ss unchanged
def SolveModel(theta, T, ss, use_saved, runcode):

    if theta == []:
        # Use defaults
        theta_use = paramsDefaults(runcode)
    else:
        theta_use = theta


    if runcode == 0:
        # Set parameters
        phi = theta_use[0]

        # Solve for steady state if none provided, make sure parameters
        # are set to the the values in theta within ss so Jacobian
        # computation uses parameters in theta, not original parameter
        # values still in ss
        if ss == []:
            print('Recomputing steady state')
            ss = hank_ss(phi=phi, noisy=False)
        else:
            ss['phi']    = phi


    if runcode == 1:
        # Set parameters
        phi, kappap = theta_use[0:2]

        # Solve for steady state if none provided, make sure parameters
        # are set to the the values in theta within ss so Jacobian
        # computation uses parameters in theta, not original parameter
        # values still in ss
        if ss == []:
            print('Recomputing steady state')
            ss = hank_ss(kappap=kappap, phi=phi, noisy=False)
        else:
            ss['phi']    = phi
            ss['kappap'] = kappap



    # (Re)Compute Jacobians
    if (runcode == 0) or (runcode == 1):
        exogenous  = ['Z', 'rstar', 'G', 'markup', 'markup_w', 'beta', 'rinv_shock']
        unknowns   = ['Y', 'r', 'w']
    targets    = ['asset_mkt', 'fisher', 'wnkpc']
    block_list = [household_inc, pricing, arbitrage, production,
                  dividend, taylor, fiscal, finance, wage, union, mkt_clearing,
                  microBetaCZ, microBetaCA, microBetaCB]
    G = jac.get_G(block_list, exogenous, unknowns, targets, T=T, ss=ss, save=True, use_saved=use_saved)

    # Add impulse responses of exogenous shocks to themselves
    for o in exogenous:
        G[o] = {i : np.identity(T)*(o==i) for i in exogenous}

    inputs  = exogenous
    outputs_all = G.keys()

    return ss, G, inputs, outputs_all


########################################################################
## Model MA Coefficients ###############################################
########################################################################


# Deviation of exogenous aggregates from their steady state values are
# assumed to follow an MA(infty) process hit by exogenous structural
# shocks. This function returns the MA coefficients characterizing that
# process (and the IRFs, since MA rep gives IRFs)
#
# Note that currently, the shock std dev is built into the MA coeffs,
# rather than left as a separate parameter.
def getShockMACoeffs(runcode, T, theta=[]):

    # Inputs and outputs
    inputs, outputs, theta0 = getruncodeFeatures(runcode)

    # Parameters
    if theta == []:
        # Use defaults
        theta_use = paramsDefaults(runcode)
    else:
        theta_use = theta

    # Set up the MA coefficients in the MA(infty) rep which characterize
    # the response (IRF) of the exogenous aggregates to the exogenous
    # shocks
    rho   = {i : 0.9 for i in inputs}
    sigma = {i : 0.01 for i in inputs}

    if runcode == 1:
        rho['Z'] = theta_use[2]

    mZs = {i : rho[i]**(np.arange(T))*sigma[i] for i in inputs}

    return mZs



# Get MA coefficients (IRFs) for endogenous aggregates
def getEndogMACoeffs(G, mZs, outputs):
    inputs      = mZs.keys()
    mXs_byo_byi = {o : {i : G[o][i] @ mZs[i] for i in inputs} for o in outputs}
    mXs_byi     = {i : np.stack([mXs_byo_byi[o][i] for o in outputs], axis=1) for i in inputs}
    mXs         = np.stack([mXs_byi[i] for i in inputs], axis=2)
    return mXs_byo_byi, mXs_byi, mXs



########################################################################
## Simulating Model Aggregates #########################################
########################################################################


# Simulating model aggregates given IRFs (MA coeffs) by directly drawing
#
# FIX: THIS IS SLOW. Maybe pass in same Sigma, don't recompute all the time
def SimModel_Draw(mYs, Npd, Nsim):

    T, Noutputs, Ninputs = mYs.shape
    dY = np.zeros((Npd, Noutputs, Nsim))

    # Estimate covariances given the IRFs (MA coeffs) and assuming unit
    # shocks. The unit shock assumption means that the shock standard
    # deviation must be incorporated into the MA coeffs (IRF) provided
    Sigma = est.all_covariances(mYs, np.ones(Ninputs))

    # Build the full covariance matrix
    V = est.build_full_covariance_matrix(Sigma, np.zeros(Noutputs), Npd)

    # Draw and return aggregate outputs
    for s in range(Nsim):
        dY[:,:,s] = np.random.multivariate_normal(np.zeros(Npd*Noutputs), V).reshape((Npd, Noutputs))

    return dY



# Given Jacobians, structural shock MA process, simulate exog and endog
# aggregates Simulating Model Aggregates from either MA rep or from
# direct drawing
def SimModel(ss, G, Npd, T, Nsim, outputs, mZs, fromMA=True, seed=None):

    inputs   = list(mZs.keys())
    Ninputs  = len(inputs)
    Noutputs = len(outputs)

    if type(seed) != type(None):
        np.random.seed(seed)

    # Get MA coefficients needed for simulation
    mYs_byo_byi, mYs_byi, mYs = getEndogMACoeffs(G, mZs, outputs)

    # Initialize matrices for exogeous and endogenous inputs'
    # deviation from steady state
    dY = np.zeros((Npd, Noutputs, Nsim))

    # Simulate from MA rep
    if fromMA:

        # Draw shocks that determine exogenous inputs
        eps = np.random.normal(size=(T+Npd,Ninputs,Nsim)) #* np.reshape(sigmas, (1,Ninputs,1)), not needed, sig in dZs

        # Loop over time periods
        for t in range(Npd):
            for n, i in enumerate(inputs):
                for m, o in enumerate(outputs):
                    dY[t,m,:] += np.matmul(eps[t:(t+T),n,:].transpose(), mYs_byo_byi[o][i])

    # Simulate from direct drawing
    else:
        dY = SimModel_Draw(mYs, Npd, Nsim)

    return dY



def SimsArrayToDict(dY_array, outputs_all):
    dY_dict = {i : np.squeeze(dY_array[:,n,:]) for n, i in enumerate(outputs_all)}
    return dY_dict



########################################################################
## Computing Empirical Moments #########################################
########################################################################


# Computing moments from Simulated Data
# Reported moments depend upon runcode

# Autocovariances of a series for lag = 0,...,nlags
def my_autocov(X, nlags):
    N   = len(X)
    ind = nlags+1
    V   = np.cov(np.stack([X[(ind-i-1):(N-i)] for i in range(ind)], axis = 0))
    return V[1,:]


# Compute muhat for a give runcode given provided dataset
def compute_muhat(runcode, dt):
    inputs, outputs, theta0 = getruncodeFeatures(runcode)

    if (runcode == 0):
        Npd, Nvar, Nsim = dt.shape
        Ncov = int(Nvar*(Nvar+1)/2 - Nvar)

        muhats = np.zeros((Nvar*2+Ncov, Nsim))
        for s in range(Nsim):
            # print("Calculating moments for %d / %d..." % (s, Nsim))

            # Var and 1st order autocorrelation of each endog aggregate
            muhats[:(Nvar*2),s] = np.apply_along_axis(lambda X: my_autocov(X,1), 0, dt[:,:,s]).reshape(Nvar*2)
            # print(len(np.apply_along_axis(lambda X: my_autocov(X,2), 0, dt[:,:,s]).reshape(Nvar*2)))

            # Contemporaneous covariance between endog aggregates
            V   = np.cov(dt[:,:,s].transpose())
            ctr = 0
            for m in range(Nvar-1):
                Vadd = V[m,(m+1):]
                # print(len(V[m,(m+1):]))
                muhats[(Nvar*2+ctr):(Nvar*2+ctr+len(Vadd)), s] = Vadd
                ctr += len(Vadd)

    elif runcode == 1:

        # Indices of endgenous aggregates
        ind_Y   = outputs.index('Y')
        ind_pi  = outputs.index('pi')
        ind_r   = outputs.index('r')
        ind_bCZ = outputs.index('BetaCZ')
        ind_bCA = outputs.index('BetaCA')
        ind_bCB = outputs.index('BetaCB')

        # For each sim, extract moments
        Npd, Nvar, Nsim = dt.shape
        muhats = np.zeros((9, Nsim))
        for s in range(Nsim):
            # print("Calculating moments for %d / %d..." % (s, Nsim))

            # 1st order autocorrelation of each endog aggregate
            var_autocorrs = np.apply_along_axis(lambda X: my_autocov(X,1), 0, dt[:,:,s])
            muhats[:3,s]  = var_autocorrs[1, np.array([ind_Y, ind_pi, ind_r])]

            # Contemporaneous covariance between endog aggregates
            V           = np.cov(dt[:,:,s].transpose())
            muhats[3,s] = V[ind_Y, ind_pi]
            muhats[4,s] = V[ind_Y, ind_r]
            muhats[5,s] = V[ind_pi, ind_r]

            # Compute the variance of the regression coefficients
            muhats[6,s] = V[ind_bCZ, ind_bCZ]
            muhats[7,s] = V[ind_bCA, ind_bCA]
            muhats[8,s] = V[ind_bCB, ind_bCB]

    return muhats


# At each time period, compute the micro variance of things and the reg coeffs
# Take the mean of this quantity over time to get the moments



########################################################################
## Computing Analytical Moments ########################################
########################################################################


# Model h() function that spits out model-implied analytical moments given parameters
def h_analytical(G, outputs, mZs, runcode):
    Noutputs = len(outputs)

    # Compute MA coeffs for response of outputs to each structural shock
    mXs_byo_byi, mXs_byi, mXs = getEndogMACoeffs(G, mZs, outputs)
    Ninputs = mXs.shape[2]

    # Compute covariances analytically, no measurement error
    # Shape is T x Noutputs x Noutputs
    # Gives correlation and autocorrelation at all lags 1,...T betwen
    # all combinations of variables
    Sigma  = est.all_covariances(mXs, np.ones(Ninputs))

    # Return model-implied moments. What exactly is returned deps on
    # runcode
    ## FIX THIS
    if runcode == 0:
        Ncov   = int(Noutputs*(Noutputs+1)/2 - Noutputs)
        muhats = np.zeros(Noutputs*2 + Ncov)

        # Var and 1st order autocorrelation of each endog aggregate
        muhats[:(Noutputs*2)] = np.hstack([Sigma[0:2,m,m] for m in range(Noutputs)])

        # Contemporaneous covariance between endog aggregates
        ctr = 0
        for m in range(Noutputs-1):
            Vadd = Sigma[0,m,(m+1):]
            muhats[(Noutputs*2+ctr):(Noutputs*2+ctr+len(Vadd))] = Vadd
            ctr += len(Vadd)


    elif runcode == 1:
        muhats = np.zeros(9)

        # Indices of endgenous aggregates
        ind_Y   = outputs.index('Y')
        ind_pi  = outputs.index('pi')
        ind_r   = outputs.index('r')
        ind_bCZ = outputs.index('BetaCZ')
        ind_bCA = outputs.index('BetaCA')
        ind_bCB = outputs.index('BetaCB')

        # 1st order autocorrelation of each endog aggregate
        muhats[0] = Sigma[1,ind_Y,  ind_Y]
        muhats[1] = Sigma[1,ind_pi, ind_pi]
        muhats[2] = Sigma[1,ind_r,  ind_r]

        # Contemporaneous covariance between endog aggregates
        muhats[3] = Sigma[0, ind_Y,  ind_pi]
        muhats[4] = Sigma[0, ind_Y,  ind_r]
        muhats[5] = Sigma[0, ind_pi, ind_r]

        # Compute the variance of the regression coefficients
        muhats[6] = Sigma[0, ind_bCZ, ind_bCZ]
        muhats[7] = Sigma[0, ind_bCA, ind_bCA]
        muhats[8] = Sigma[0, ind_bCB, ind_bCB]


    return muhats



########################################################################
## h() Function for Moment-Matching ####################################
########################################################################


# Model h() function that spits out moments given parameters, can be analytically computed or empirical
#
# Want initial ss and G because can reuse some stuff there
def h(theta, ss, Npd, T, outputs, runcode, empirical=False, simFromMA=False, Nsim=1000, verbose=False):


    # Resolve model for given parameters
    # Using saved because assuming model already initialized
    # print(G['i']['Z'][:,0:2].transpose())
    tic = time.clock()
    use_saved=True
    ss, G, inputs, outputs_all = SolveModel(theta, T, ss, use_saved, runcode)
    if verbose:
        print("Solving: %f" % (time.clock()-tic))

    # Recompute IRFs given parameter values
    tic = time.clock()
    mZs = getShockMACoeffs(runcode, T, theta)
    if verbose:
        print("IRFs: %f" % (time.clock()-tic))

    # Compute moments: Empirical, else analytical
    Ninputs  = len(inputs)
    Noutputs = len(outputs)
    if empirical:
        tic = time.clock()
        # Simulate data
        dX = SimModel(ss, G, Npd, T, Nsim, outputs, mZs, simFromMA, 314)

        # Compute muhat and average over draws
        toret = np.mean(compute_muhat(runcode, dX), axis=1)
        if verbose:
            print("Moments: %f" % (time.clock()-tic))
        return toret

    # Analytical
    else:
        tic = time.clock()
        toret = h_analytical(G, outputs, mZs, runcode)
        if verbose:
            print("Moments: %f" % (time.clock()-tic))
        return toret



########################################################################
## Features by code ####################################################
########################################################################


def paramsDefaults(runcode):
    if runcode == 0:
        phi0   = 1.5
        theta0 = [phi0]
    elif runcode == 1:
        phi0     = 1.5
        kappap0  = 0.1
        rho_Z0   = 0.5
        theta0   = [phi0, kappap0, rho_Z0]

    return theta0

# For each parameter, establish a range and a default
def paramsCheckID(runcode):
    if runcode == 0:
        phis     = np.linspace(start=1.01, stop=2.0, num=20)
        thetas   = [phis]
    elif runcode == 1:
        phis     = np.linspace(start=1.01, stop=1.9, num=5)
        kappaps  = np.linspace(start=.01, stop=0.2, num=5)
        rho_Zs   = np.linspace(start=0.2, stop=0.7, num=5)

        # Put everything together
        thetas = [phis, kappaps, rho_Zs]

    theta0 = paramsDefaults(runcode)

    return thetas, theta0


def getruncodeFeatures(runcode):
    inputs  = ['Z', 'rstar', 'G', 'markup', 'markup_w', 'beta', 'rinv_shock']
    # outputs = ['Y', 'C', 'K']
    # outputs = ['Y', 'r', 'pi', 'VarA', 'VarB', 'VarC', 'CovAC', 'CovBC', 'CovCZ']
    outputs = ['Y', 'r', 'pi', 'BetaCZ', 'BetaCA', 'BetaCB']
    theta0  = paramsDefaults(runcode)
    return inputs, outputs, theta0


def getStrSave(runcode, T, Npd, Nsim, fromMA):
    strSave = "runcode%d_T%d_Npd%d_Nsim%d_fromMA%d" % (runcode, T, Npd, Nsim, fromMA)
    return strSave


def quadratic_form(h, W):
    return np.vdot(h, np.matmul(W, h))















