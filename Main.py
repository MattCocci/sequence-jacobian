# Standard python packagesD
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
import wcopt
import warnings
import multiprocess as mp
import copy

warnings.simplefilter('error')


# Commit message
# - Add comments
# - Rewrite parameter vector as dictionary with param names
# - Fixed ordering in compute_muhat


# Runcode 0
# - Summary: Single parameter, small no. of shocks, many moments
# - Params:  1, [phi0]
# - Shocks:  3, ['Z', 'rstar', 'G']
# - Moments: All variances, autocovariances, covariances for
#            Y, r, pi, Beta_c_zm, Beta_c_a, Beta_c_b

# Runcode 1
# - Summary: Multiple parameters, small number of shocks for testing
# - Params:  3, [phi0, kappap0, rho_Z0]
# - Shocks:  3, ['Z', 'rstar', 'G']
# - Moments: All variances, autocovariances at first order, covariances
#               'Var(Y)',
#               'Var(pi)',
#               'Var(r)',
#               'AutoCorr(Y)',
#               'AutoCorr(pi)',
#               'AutoCorr(r)',
#               'Cov(Y, pi)',
#               'Cov(Y, r)',
#               'Cov(pi, r)',
#               'Var(beta_C_Z)',
#               'Var(Beta_c_a)',
#               'Var(Beta_c_b)']


# Runcode 2
# - Params: 3, [phi0, kappap0, rho_Z0]
# - Shocks: all, ['Z', 'rstar', 'G', 'markup', 'markup_w', 'beta', 'rinv_shock']
# - Moments:
#               'Var(Y)',
#               'Var(pi)',
#               'Var(r)',
#               'Var(C)',
#               'Var(I)',
#               'Var(i)',
#               'AutoCorr(Y)',
#               'AutoCorr(pi)',
#               'AutoCorr(r)',
#               'AutoCorr(C)',
#               'AutoCorr(I)',
#               'AutoCorr(i)',
#               'Corr(Y,pi)',
#               'Corr(Y,r)',
#               'Corr(Y,i)',
#               'Corr(Y,C)',
#               'Corr(pi,r)',
#               'Corr(pi,I)',
#               'Corr(C,pi)',
#               'Corr(C,I)',
#               'Corr(C,i)',
#               'Corr(C,r)',
#               'Corr(I,i)',
#               'Corr(I,r)',
#               'Var(Beta_c_zm)',
#               'Var(Beta_c_a)',
#               'Var(Beta_c_b  )',
#               'Var(Beta_a_zm)',
#               'Var(Beta_b_zm)',
#               'Var(Beta_a_zmY)',
#               'Var(Beta_b_zmY)',
#               'Var(Corr_A_ZM, )',
#               'Var(Corr_B_ZM )',
#               'Corr(Beta_c_zm,Y)',
#               'Corr(Beta_c_a,Y)',
#               'Corr(Beta_c_b,Y)',
#               'Corr(Beta_a_zm,Y)',
#               'Corr(Beta_b_zm,Y)',
#               'Corr(Beta_a_zmY,Y)',
#               'Corr(Beta_b_zmY,Y)',
#               'Corr(Corr_A_ZM,Y)',
#               'Corr(Corr_B_ZM,Y)'

# Runcode 3
# - Params: 9, [phi0, kappap0, rho_Z0, kappaw0, rho_rstar0, rho_beta0, sigma_Z0, sigma_rstar0, sigma_beta0]
# - Shocks: all, ['Z', 'rstar', 'G', 'markup', 'markup_w', 'beta', 'rinv_shock']
# - Moments:
#               'Var(Y)',
#               'Var(pi)',
#               'Var(r)',
#               'Var(C)',
#               'Var(I)',
#               'Var(i)',
#               'AutoCorr(Y)',
#               'AutoCorr(pi)',
#               'AutoCorr(r)',
#               'AutoCorr(C)',
#               'AutoCorr(I)',
#               'AutoCorr(i)',
#               'Corr(Y,pi)',
#               'Corr(Y,r)',
#               'Corr(Y,i)',
#               'Corr(Y,C)',
#               'Corr(pi,r)',
#               'Corr(pi,I)',
#               'Corr(C,pi)',
#               'Corr(C,I)',
#               'Corr(C,i)',
#               'Corr(C,r)',
#               'Corr(I,i)',
#               'Corr(I,r)',
#               'Var(Beta_c_zm)',
#               'Var(Beta_c_a)',
#               'Var(Beta_c_b  )',
#               'Var(Beta_a_zm)',
#               'Var(Beta_b_zm)',
#               'Var(Beta_a_zmY)',
#               'Var(Beta_b_zmY)',
#               'Var(Corr_A_ZM, )',
#               'Var(Corr_B_ZM )',
#               'Corr(Beta_c_zm,Y)',
#               'Corr(Beta_c_a,Y)',
#               'Corr(Beta_c_b,Y)',
#               'Corr(Beta_a_zm,Y)',
#               'Corr(Beta_b_zm,Y)',
#               'Corr(Beta_a_zmY,Y)',
#               'Corr(Beta_b_zmY,Y)',
#               'Corr(Corr_A_ZM,Y)',
#               'Corr(Corr_B_ZM,Y)'



# Runcode 4: SHOULD BE SELECTED MOMENTS
# - Params: 9, [phi0, kappap0, rho_Z0, kappaw0, rho_rstar0, rho_beta0, sigma_Z0, sigma_rstar0, sigma_beta0]
# - Shocks: all, ['Z', 'rstar', 'G', 'markup', 'markup_w', 'beta', 'rinv_shock']
# - Moments:
#               'Var(Y)',
#               'Var(pi)',
#               'AutoCorr(Y)',
#               'AutoCorr(pi)',
#               'AutoCorr(r)',
#               'AutoCorr(i)',
#               'Corr(Y,pi)',
#               'Corr(Y,r)',
#               'Corr(Y,C)',
#               'Corr(pi,r)',
#               'Corr(pi,I)',
#               'Corr(C,I)',
#               'Corr(C,i)',
#               'Corr(C,r)',
#               'Var(Corr_B_ZM)',
#               'Corr(Beta_a_zm,Y)',
#               'Corr(Beta_b_zm,Y)',
#               'Corr(Beta_a_zmY,Y)',
#               'Corr(Beta_b_zmY,Y)',


# Runcode 5: SHOULD BE SELECTED MOMENTS
# - Params: 3, [phi0, kappap0, rho_Z0]
# - Shocks: all, ['Z', 'rstar', 'G', 'markup', 'markup_w', 'beta', 'rinv_shock']
# - Moments:
#               'Var(Y)',
#               'Var(pi)',
#               'AutoCorr(Y)',
#               'AutoCorr(pi)',
#               'AutoCorr(r)',
#               'AutoCorr(i)',
#               'Corr(Y,pi)',
#               'Corr(Y,r)',
#               'Corr(Y,C)',
#               'Corr(pi,r)',
#               'Corr(pi,I)',
#               'Corr(C,I)',
#               'Corr(C,i)',
#               'Corr(C,r)',
#               'Var(Corr_B_ZM)',
#               'Corr(Beta_a_zm,Y)',
#               'Corr(Beta_b_zm,Y)',
#               'Corr(Beta_a_zmY,Y)',
#               'Corr(Beta_b_zmY,Y)',




# Computing the empirical variance and using as weighting matrix for
# moment matching



# In model with three shocks
# --------------------------
# I can sim the model, including aggregates & micro regression moments
#   for different values of the parameters
# I can produce identification plots showing how aggregate moments
#   change with different parameters
# I can do CMD to estimate parameters


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
def SolveModel(modelInfo, init):

    # RUNCODE
    runcode = modelInfo['runcode']

    ## SET PARAMETER VALUES ##
    # If modelInfo dictionary not already initialized, initialize it
    # Also set parameters to use, theta_use
    if init:
        modelInfo.update(getruncodeFeatures(runcode))
        theta_use = modelInfo['theta0']
    else:
        theta_use = modelInfo['theta']

    ## STEADY STATE ##
    # - Solve for it, if not provided
    # - If ss provided, set parameters to whatever is in theta_use,
    #   so we use those when computing Jacobian, not the original
    #   parameter values that were stored in ss when we first solved
    if 'ss' not in modelInfo:
        print('Computing steady state')
        modelInfo['ss'] = hank_ss(noisy=False, **theta_use)
    else:
        modelInfo['ss'].update(theta_use)

    ## JACOBIANS

    # Compute or recompute jacobians. If init, get all Jacobians
    if init:
        kwargs = {'save' : True, 'use_saved' : False}
    else:
        kwargs = {'outputs' : modelInfo['outputs'], 'save' : True, 'use_saved' : True}
    modelInfo['G'] = jac.get_G(modelInfo['block_list'], modelInfo['exogenous'], modelInfo['unknowns'], modelInfo['targets'], T=modelInfo['T'], ss=modelInfo['ss'], **kwargs)

    # Add impulse responses of exogenous shocks to themselves
    for o in modelInfo['exogenous']:
        modelInfo['G'][o] = {i : np.identity(modelInfo['T'])*(o==i) for i in modelInfo['exogenous']}

    # Get list of all outputs
    if init:
        modelInfo['outputs_all'] = modelInfo['G'].keys()

    return modelInfo


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
def getShockMACoeffs(modelInfo):

    # Parameters
    if 'theta' not in modelInfo:
        theta = modelInfo['theta0']
    else:
        theta = modelInfo['theta']

    ## Set up the MA coefficients ##
    # in the MA(infty) rep which characterize the response (IRF) of the
    # exogenous aggregates to the exogenous shocks

    # Defaults
    rho   = {v : 0.90 for v in modelInfo['exogenous']}
    sigma = {v : 0.01 for v in modelInfo['exogenous']}

    # Alts
    if (modelInfo['runcode'] == 1) or (modelInfo['runcode'] == 2) or (modelInfo['runcode'] == 5):
        rho['Z'] = theta['rho_Z']
    if (modelInfo['runcode'] == 3) or (modelInfo['runcode'] == 4):
        rho['Z']       = theta['rho_Z']
        rho['rstar']   = theta['rho_rstar']
        rho['beta']    = theta['rho_beta']
        sigma['Z']     = theta['sigma_Z']
        sigma['rstar'] = theta['sigma_rstar']
        sigma['beta']  = theta['sigma_beta']

    mZs = {i : rho[i]**(np.arange(modelInfo['T']))*sigma[i] for i in modelInfo['exogenous']}

    return mZs



# Get MA coefficients (IRFs) for endogenous aggregates
def getEndogMACoeffs(modelInfo, mZs):
    G           = modelInfo['G']
    inputs      = modelInfo['exogenous']
    outputs     = modelInfo['outputs']
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

    verbose = False

    T, Noutputs, Ninputs = mYs.shape
    dY = np.zeros((Npd, Noutputs, Nsim))

    ## Covariances given IRFs ##
    # Estimate covariances given the IRFs (MA coeffs) and assuming unit
    # shocks. The unit shock assumption means that the shock standard
    # deviation must be incorporated into the MA coeffs (IRF) provided
    # print("Estimating all covariances")
    # print(mYs.shape)
    # tic = time.perf_counter()
    Sigma = est.all_covariances(mYs, np.ones(Ninputs))
    # print(time.perf_counter()-tic)

    ## Build full covariance matrix ##
    # print("Building full covariance matrix")
    # tic = time.perf_counter()
    V = est.build_full_covariance_matrix(Sigma, np.zeros(Noutputs), Npd)
    # print(time.perf_counter()-tic)

    # For comparing Sigma and V
    # print(Sigma.shape)
    # print(V.shape)

    ## Draw and return aggregate outputs ##
    # print("Drawing")
    # tic = time.perf_counter()
    # for s in range(Nsim):
        # dY[:,:,s] = np.random.multivariate_normal(np.zeros(Npd*Noutputs), V).reshape((Npd, Noutputs))
    draws = np.random.multivariate_normal(np.zeros(Npd*Noutputs), V, size=Nsim)
    dY    = draws.T.reshape((Npd,Noutputs,Nsim), order='C')
    # dY_orig = draws.reshape((Npd,Noutputs)) # This is how the reshape works in original code
    # print(time.perf_counter()-tic)

    # # Use these to translate y_{st} for some s and t into an
    # # index/position in the big vector of draws.
    # pd  = np.repeat(np.arange(Npd).reshape((Npd,1)), Noutputs, axis=1)
    # srs = np.repeat(np.arange(Noutputs).reshape((Noutputs,1)), Npd, axis=1).T
    # pd_flat  = pd.reshape(Npd*Noutputs)
    # srs_flat = srs.reshape(Npd*Noutputs)

    # Check if things are where they're supposed to be in V given
    # definition of Sigma and positioning of elements
    # i = 0
    # t = 1
    # j = 0
    # s = 1
    # ind1 = (pd_flat == t) & (srs_flat == i)
    # ind2 = (pd_flat == s) & (srs_flat == j)
    # print(V[ind1,ind2])
    # print(Sigma[0,i,i])

    # Compare moments of simulated series to analytical
    # print(np.mean(np.vstack([np.diag(np.cov(dY_mine[:,:,s].T)) for s in range(Nsim)]), axis=0))
    # print([np.var(dY_mine[:,s,0]) for s in range(Noutputs)])
    # print(np.diag(Sigma[0,:,:]))

    return dY



# Given Jacobians, structural shock MA process, simulate exog and endog
# aggregates Simulating Model Aggregates from either MA rep or from
# direct drawing
def SimModel(modelInfo, mZs, Npd, Nsim, fromMA=True, seed=None):

    T        = modelInfo['T']
    inputs   = modelInfo['exogenous']
    outputs  = modelInfo['outputs']
    Ninputs  = len(inputs)
    Noutputs = len(outputs)

    if type(seed) != type(None):
        np.random.seed(seed)

    # Get MA coefficients needed for simulation
    # print("Computing MA coeffs")
    # tic = time.perf_counter()
    mYs_byo_byi, mYs_byi, mYs = getEndogMACoeffs(modelInfo, mZs)
    # print("SimModel")
    # print(mYs.shape)
    # print(mYs[4, :, :])
    # print(time.perf_counter()-tic)

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



# Compare simmed series when constructed from MA rep vs. direct drawing
def compareSimulationsMAvDraw(modelInfo, Npd, Nsim):

    mZs = getShockMACoeffs(modelInfo)

    # Simulate from MA: T x Noutputs x Nsim
    dY_MA = SimModel(modelInfo, mZs, Npd, Nsim, True, None)

    # Simulate from direct draw
    dY_draw = SimModel(modelInfo, mZs, Npd, Nsim, False, None)

    # print("From MA")
    # print("From Drawing")
    # print(dY_MA.shape)
    # print(dY_draw.shape)

    # Show simulations of each series, using each method
    plt.plot(dY_MA[:,:,1])
    plt.show()
    plt.plot(dY_draw[:,:,1])
    plt.show()

    # Compute moments given simulation
    inputs  = modelInfo['exogenous']
    outputs = modelInfo['outputs']
    Ninputs = len(inputs)
    print("Moments")
    muhats_MA   = compute_muhat(modelInfo, dY_MA[:, :, :])
    muhats_draw = compute_muhat(modelInfo, dY_draw[:, :, :])
    print(np.vstack([np.mean(muhats_MA, axis=1), np.mean(muhats_draw, axis=1)]).T)



########################################################################
## Computing Empirical Moments #########################################
########################################################################


# Computing moments from Simulated Data
# Reported moments depend upon runcode

# Autocovariances of a series for lag = 1,...,nlags
def my_autocov(X, nlags):
    N   = len(X)
    ind = nlags+1
    V   = np.cov(np.stack([X[(ind-i-1):(N-i)] for i in range(ind)], axis = 0))
    return V[0,1:]

# Autocorr of a series for lag = 1,...,nlags
def my_autocorr(X, nlags):
    return my_autocov(X,nlags) / np.var(X)

# Unconditional variance, based on AR(1) approximation
def my_uncond_var(X):
    rho = my_autocorr(X,1)
    N   = len(X)
    if 1-(rho**2) < 0:
        return np.var(X)
    else:
        return np.var(X[1:] - rho*X[0:(N-1)]) / (1-rho**2)



# Compute muhat from dataset for given runcode, looping over sims
def compute_muhat(modelInfo, dt):
    runcode = modelInfo['runcode']

    Npd, Nvar, Nsim = dt.shape

    if (runcode == 0):
        Ncov   = int(Nvar*(Nvar+1)/2 - Nvar)
        muhats = np.zeros((Nvar*2+Ncov, Nsim))
        for s in range(Nsim):
            # print("Calculating moments for %d / %d..." % (s, Nsim))

            # Var for each endog aggregate
            # muhats[:Nvar,s] = np.apply_along_axis(lambda X: np.var(X), 0, dt[:,:,s]).reshape(Nvar)
            # print(muhats[:Nvar,s])
            muhats[:Nvar,s] = np.apply_along_axis(lambda X: my_uncond_var(X), 0, dt[:,:,s]).reshape(Nvar)
            # print(muhats[:Nvar,s])

            # 1st order autocov of each endog aggregate
            muhats[Nvar:(Nvar*2),s] = np.apply_along_axis(lambda X: my_autocov(X,1), 0, dt[:,:,s]).reshape(Nvar)
            # print(np.apply_along_axis(lambda X: my_autocorr(X,5), 0, dt[:,:,s]))

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
        outputs = modelInfo['outputs']
        ind_Y   = outputs.index('Y')
        ind_pi  = outputs.index('pi')
        ind_r   = outputs.index('r')

        # For each sim, extract moments
        Npd, Nvar, Nsim = dt.shape
        muhats = np.zeros((9, Nsim))
        for s in range(Nsim):
            # print("Calculating moments for %d / %d..." % (s, Nsim))

            # Uncond Variances
            # vars_        = np.squeeze(np.apply_along_axis(lambda X: np.var(X), 0, dt[:,:,s]))
            vars_         = np.squeeze(np.apply_along_axis(lambda X: my_uncond_var(X), 0, dt[:,:,s]))
            muhats[:3,s]  = vars_[np.array([ind_Y, ind_pi, ind_r])]

            # First order autocov
            autocovs_     = np.squeeze(np.apply_along_axis(lambda X: my_autocov(X,1), 0, dt[:,:,s]))
            muhats[3:6,s] = autocovs_[np.array([ind_Y, ind_pi, ind_r])]

            # Contemporaneous covariance between endog aggregates
            ind = 6
            V               = np.cov(dt[:,:,s].transpose())
            muhats[ind,s]   = V[ind_Y, ind_pi]
            muhats[ind+1,s] = V[ind_Y, ind_r]
            muhats[ind+2,s] = V[ind_pi, ind_r]

    elif (runcode == 2) or (runcode == 3):
        # Variance of micro coeffs, non consumption
        # Corr of beta coeff and Y

        outputs = modelInfo['outputs']
        ind_Y   = outputs.index('Y')
        ind_pi  = outputs.index('pi')
        ind_r   = outputs.index('r')
        ind_C   = outputs.index('C')
        ind_I   = outputs.index('I')
        ind_i   = outputs.index('i')
        ind_ZMY = outputs.index('ZMY')
        ind_A   = outputs.index('A')
        ind_B   = outputs.index('B')

        ind_Beta_a_zm  = outputs.index('Beta_a_zm')
        ind_Beta_b_zm  = outputs.index('Beta_b_zm')
        ind_Beta_c_zm  = outputs.index('Beta_c_zm')

        ind_Corr_c_a   = outputs.index('Corr_c_a')
        ind_Corr_c_b   = outputs.index('Corr_c_b')
        ind_Cov_c_a    = outputs.index('Cov_c_a')
        ind_Cov_c_b    = outputs.index('Cov_c_b')

        ind_Corr_a_zm  = outputs.index('Corr_a_zm')
        ind_Corr_b_zm  = outputs.index('Corr_b_zm')
        ind_Cov_a_zm   = outputs.index('Cov_a_zm')
        ind_Cov_b_zm   = outputs.index('Cov_b_zm')

        ind_Beta_a_zmY = outputs.index('Beta_a_zmY')
        ind_Beta_b_zmY = outputs.index('Beta_b_zmY')


        # For each sim, extract moments
        Npd, Nvar, Nsim = dt.shape
        muhats = np.zeros((6*2 + 12 + 11 + 14, Nsim))
        for s in range(Nsim):
            # print("Calculating moments for %d / %d..." % (s, Nsim))

            # Uncond Variances
            # vars_       = np.squeeze(np.apply_along_axis(lambda X: np.var(X), 0, dt[:,:,s]))
            vars_        = np.squeeze(np.apply_along_axis(lambda X: my_uncond_var(X), 0, dt[:,:,s]))
            muhats[:6,s] = vars_[np.array([ind_Y, ind_pi, ind_r, ind_C, ind_I, ind_i])]

            # First order autocorrelation
            autocovs_      = np.squeeze(np.apply_along_axis(lambda X: my_autocov(X,1), 0, dt[:,:,s]))
            muhats[6:12,s] = autocovs_[np.array([ind_Y, ind_pi, ind_r, ind_C, ind_I, ind_i])] / muhats[:6,s]

            # Contemporaneous corr between endog aggregates
            ind = 12
            V                = np.cov(dt[:,:,s].transpose())
            muhats[ind+0,s]  = V[ind_Y, ind_pi]  /  np.sqrt( vars_[ind_Y ] *  vars_[ind_pi] )
            muhats[ind+1,s]  = V[ind_Y, ind_r]   /  np.sqrt( vars_[ind_Y ] *  vars_[ind_r ] )
            muhats[ind+2,s]  = V[ind_Y, ind_i]   /  np.sqrt( vars_[ind_Y ] *  vars_[ind_i ] )
            muhats[ind+3,s]  = V[ind_Y, ind_C]   /  np.sqrt( vars_[ind_Y ] *  vars_[ind_C ] )
            muhats[ind+4,s]  = V[ind_pi, ind_r]  /  np.sqrt( vars_[ind_pi] *  vars_[ind_r ] )
            muhats[ind+5,s]  = V[ind_pi, ind_I]  /  np.sqrt( vars_[ind_pi] *  vars_[ind_I ] )
            muhats[ind+6,s]  = V[ind_C, ind_pi]  /  np.sqrt( vars_[ind_C ] *  vars_[ind_pi] )
            muhats[ind+7,s]  = V[ind_C, ind_I]   /  np.sqrt( vars_[ind_C ] *  vars_[ind_I ] )
            muhats[ind+8,s]  = V[ind_C, ind_i]   /  np.sqrt( vars_[ind_C ] *  vars_[ind_i ] )
            muhats[ind+9,s]  = V[ind_C, ind_r]   /  np.sqrt( vars_[ind_C ] *  vars_[ind_r ] )
            muhats[ind+10,s] = V[ind_I, ind_i]   /  np.sqrt( vars_[ind_I ] *  vars_[ind_i ] )
            muhats[ind+11,s] = V[ind_I, ind_r]   /  np.sqrt( vars_[ind_I ] *  vars_[ind_r ] )

            # Compute the variance of the regression and corr coefficients
            ind = ind + 12
            muhats[ind+0,s]  = vars_[ind_Corr_c_a  ]
            muhats[ind+1,s]  = vars_[ind_Corr_c_b  ]
            muhats[ind+2,s]  = vars_[ind_Corr_a_zm ]
            muhats[ind+3,s]  = vars_[ind_Corr_b_zm ]

            muhats[ind+4,s]  = vars_[ind_Cov_c_a  ]
            muhats[ind+5,s]  = vars_[ind_Cov_c_b  ]
            muhats[ind+6,s]  = vars_[ind_Cov_a_zm ]
            muhats[ind+7,s]  = vars_[ind_Cov_b_zm ]

            muhats[ind+8,s]  = vars_[ind_Beta_a_zm]
            muhats[ind+9,s]  = vars_[ind_Beta_b_zm]
            muhats[ind+10,s] = vars_[ind_Beta_c_zm]


            # Corr of reg/corr coeffs and Y
            ind = ind + 11
            muhats[ind+0,s]  = V[ind_Corr_c_a  , ind_Y] / np.sqrt(vars_[ind_Corr_c_a ] * vars_[ind_Y])  # Corr(Corr_t(c,a), Y)
            muhats[ind+1,s]  = V[ind_Corr_c_b  , ind_Y] / np.sqrt(vars_[ind_Corr_c_b ] * vars_[ind_Y])  # Corr(Corr_t(c,b), Y)
            muhats[ind+2,s]  = V[ind_Corr_a_zm , ind_Y] / np.sqrt(vars_[ind_Corr_a_zm] * vars_[ind_Y])  # Corr(Corr_t(y,a), Y)
            muhats[ind+3,s]  = V[ind_Corr_b_zm , ind_Y] / np.sqrt(vars_[ind_Corr_b_zm] * vars_[ind_Y])  # Corr(Corr_t(y,b), Y)

            muhats[ind+4,s]  = V[ind_Cov_c_a   , ind_Y] / np.sqrt(vars_[ind_Cov_c_a  ] * vars_[ind_Y])  # Corr(Cov_t(c,a), Y)
            muhats[ind+5,s]  = V[ind_Cov_c_b   , ind_Y] / np.sqrt(vars_[ind_Cov_c_b  ] * vars_[ind_Y])  # Corr(Cov_t(c,b), Y)
            muhats[ind+6,s]  = V[ind_Cov_a_zm  , ind_Y] / np.sqrt(vars_[ind_Cov_a_zm ] * vars_[ind_Y])  # Corr(Cov_t(y,a), Y)
            muhats[ind+7,s]  = V[ind_Cov_b_zm  , ind_Y] / np.sqrt(vars_[ind_Cov_b_zm ] * vars_[ind_Y])  # Corr(Cov_t(y,b), Y)

            muhats[ind+8,s]  = V[ind_Beta_a_zm , ind_Y] / np.sqrt(vars_[ind_Beta_a_zm] * vars_[ind_Y])  # Corr(Beta_t(a,y), Y)
            muhats[ind+9,s]  = V[ind_Beta_b_zm , ind_Y] / np.sqrt(vars_[ind_Beta_b_zm] * vars_[ind_Y])  # Corr(Beta_t(b,y), Y)

            muhats[ind+10,s] = V[ind_Beta_a_zm , ind_Y]                                                 # Cov(Beta_t(a,y), Y)
            muhats[ind+11,s] = V[ind_Beta_b_zm , ind_Y]                                                 # Cov(Beta_t(b,y), Y)

            muhats[ind+12,s] = (modelInfo['ss']['Y_Cov_a_zm'] + V[ind_ZMY, ind_A]) / (modelInfo['ss']['Y2_Var_zm']  + vars_[ind_ZMY])
            muhats[ind+13,s] = (modelInfo['ss']['Y_Cov_b_zm'] + V[ind_ZMY, ind_B]) / (modelInfo['ss']['Y2_Var_zm']  + vars_[ind_ZMY])


    elif (runcode == 4) or (runcode == 5):
        # Variance of micro coeffs, non consumption
        # Corr of beta coeff and Y

        outputs = modelInfo['outputs']
        ind_Y   = outputs.index('Y')
        ind_pi  = outputs.index('pi')
        ind_r   = outputs.index('r')
        ind_C   = outputs.index('C')
        ind_I   = outputs.index('I')
        ind_i   = outputs.index('i')
        ind_ZMY = outputs.index('ZMY')
        ind_AY  = outputs.index('AY')
        ind_BY  = outputs.index('BY')
        ind_AZM = outputs.index('AZM')
        ind_BZM = outputs.index('BZM')
        ind_AZMY = outputs.index('AZMY')
        ind_BZMY = outputs.index('BZMY')
        ind_A    = outputs.index('A')
        ind_B    = outputs.index('B')

        ind_Beta_c_zm  = outputs.index('Beta_c_zm')
        ind_Beta_c_a   = outputs.index('Beta_c_a')
        ind_Beta_c_b   = outputs.index('Beta_c_b')
        ind_Beta_a_zm  = outputs.index('Beta_a_zm')
        ind_Beta_b_zm  = outputs.index('Beta_b_zm')
        ind_Beta_a_zmY = outputs.index('Beta_a_zmY')
        ind_Beta_b_zmY = outputs.index('Beta_b_zmY')
        ind_Corr_a_zm  = outputs.index('Corr_a_zm')
        ind_Corr_b_zm  = outputs.index('Corr_b_zm')

        ind_Cov_a_zm   = outputs.index('Cov_a_zm')
        ind_Cov_b_zm   = outputs.index('Cov_b_zm')
        ind_Y_Cov_a_zm = outputs.index('Y_Cov_a_zm')
        ind_Y_Cov_b_zm = outputs.index('Y_Cov_b_zm')

        ZM = modelInfo['ss']['ZM']
        A  = modelInfo['ss']['A']
        Y  = modelInfo['ss']['Y']

        # For each sim, extract moments
        Npd, Nvar, Nsim = dt.shape
        muhats = np.zeros((25, Nsim))
        for s in range(Nsim):
            # print("Calculating moments for %d / %d..." % (s, Nsim))

            # Uncond Variances
            # vars_        = np.squeeze(np.apply_along_axis(lambda X: np.var(X), 0, dt[:,:,s]))
            vars_        = np.squeeze(np.apply_along_axis(lambda X: my_uncond_var(X), 0, dt[:,:,s]))
            muhats[:2,s] = vars_[np.array([ind_Y, ind_pi])]

            # First order autocorr
            autocovs_      = np.squeeze(np.apply_along_axis(lambda X: my_autocov(X,1), 0, dt[:,:,s]))
            muhats[2:6,s]  = autocovs_[np.array([ind_Y, ind_pi, ind_i, ind_I])] / vars_[np.array([ind_Y, ind_pi, ind_i, ind_I])]

            # Contemporaneous corr between endog aggregates
            ind = 6
            V                = np.cov(dt[:,:,s].transpose())
            muhats[ind+0,s]  = V[ind_Y,  ind_pi] /  np.sqrt( vars_[ind_Y ] *  vars_[ind_pi] )
            muhats[ind+1,s]  = V[ind_Y,  ind_i]  /  np.sqrt( vars_[ind_Y ] *  vars_[ind_i ] )
            muhats[ind+2,s]  = V[ind_Y,  ind_C]  /  np.sqrt( vars_[ind_Y ] *  vars_[ind_C ] )
            muhats[ind+3,s]  = V[ind_pi, ind_i]  /  np.sqrt( vars_[ind_pi] *  vars_[ind_i ] )
            muhats[ind+4,s]  = V[ind_pi, ind_I]  /  np.sqrt( vars_[ind_pi] *  vars_[ind_I ] )
            muhats[ind+5,s]  = V[ind_C,  ind_I]  /  np.sqrt( vars_[ind_C ] *  vars_[ind_I ] )
            muhats[ind+6,s]  = V[ind_C,  ind_i]  /  np.sqrt( vars_[ind_C ] *  vars_[ind_i ] )
            muhats[ind+7,s]  = V[ind_C,  ind_pi] /  np.sqrt( vars_[ind_C ] *  vars_[ind_pi] )
            muhats[ind+8,s]  = V[ind_I,  ind_i]  /  np.sqrt( vars_[ind_I ] *  vars_[ind_i ] )

            # Compute the variance of the regression and corr coefficients
            ind = ind + 9
            muhats[ind+0,s] = vars_[ind_Beta_b_zm]

            # Corr of reg/corr coeffs and Y
            ind = ind + 1
            muhats[ind+0,s] = V[ind_Beta_a_zmY, ind_Y] / np.sqrt(vars_[ind_Beta_a_zmY] * vars_[ind_Y])
            muhats[ind+1,s] = V[ind_Beta_b_zmY, ind_Y] / np.sqrt(vars_[ind_Beta_b_zmY] * vars_[ind_Y])

            # Regression coefficients
            ind = ind + 2
            muhats[ind+0,s] = (modelInfo['ss']['Y_Cov_a_zm'] +  V[ind_ZMY, ind_A]) / \
                              (modelInfo['ss']['Y2_Var_zm']  + V[ind_ZMY, ind_ZMY])

    return muhats



########################################################################
## Computing Analytical Moments ########################################
########################################################################


# Model h() function that spits out model-implied analytical moments given parameters
def h_analytical(modelInfo, mZs):
    runcode  = modelInfo['runcode']
    inputs   = modelInfo['exogenous']
    outputs  = modelInfo['outputs']
    Noutputs = len(outputs)
    Ninputs  = len(inputs)

    # Compute MA coeffs for response of outputs to each structural shock
    mXs_byo_byi, mXs_byi, mXs = getEndogMACoeffs(modelInfo, mZs)
    print("h_analytical")
    # print(mXs.shape)
    # print(mXs[4, :, :])

    # Compute covariances analytically, no measurement error
    # Shape is T x Noutputs x Noutputs
    # Gives correlation and autocorrelation at all lags 1,...T betwen
    # all combinations of variables
    Sigma  = est.all_covariances(mXs, np.ones(Ninputs))

    # Return model-implied moments. Which ones depends upon runcode
    ## FIX THIS
    if (runcode == 0):
        Ncov   = int(Noutputs*(Noutputs+1)/2 - Noutputs)
        muhats = np.zeros(Noutputs*2 + Ncov)

        # Uncond var and 1st order autocorrelation of each endog aggregate
        muhats[:(Noutputs)]             = np.hstack([Sigma[0,m,m] for m in range(Noutputs)])
        muhats[(Noutputs):(Noutputs*2)] = np.hstack([Sigma[1,m,m] for m in range(Noutputs)])

        # Contemporaneous covariance between endog aggregates
        ctr = 0
        for m in range(Noutputs-1):
            # print("SHAPE")
            # print(Sigma.shape)
            Vadd = Sigma[0,m,(m+1):]
            muhats[(Noutputs*2+ctr):(Noutputs*2+ctr+len(Vadd))] = Vadd
            ctr += len(Vadd)

    elif (runcode == 1):
        muhats = np.zeros(9)

        # Indices of endgenous aggregates
        ind_Y   = outputs.index('Y')
        ind_pi  = outputs.index('pi')
        ind_r   = outputs.index('r')

        # Variance of each endog aggregate
        muhats[0] = Sigma[0,ind_Y,  ind_Y]
        muhats[1] = Sigma[0,ind_pi, ind_pi]
        muhats[2] = Sigma[0,ind_r,  ind_r]

        # 1st order autocorrelation of each endog aggregate
        ind = 3
        muhats[ind]   = Sigma[1,ind_Y,  ind_Y]
        muhats[ind+1] = Sigma[1,ind_pi, ind_pi]
        muhats[ind+2] = Sigma[1,ind_r,  ind_r]

        # Contemporaneous covariance between endog aggregates
        ind = 6
        muhats[ind]   = Sigma[0, ind_Y,  ind_pi]
        muhats[ind+1] = Sigma[0, ind_Y,  ind_r]
        muhats[ind+2] = Sigma[0, ind_pi, ind_r]


    elif (runcode == 2) or (runcode == 3):
        muhats = np.zeros(12 + 12 + 11 + 14)

        # Indices of exogenous aggregates
        ind_Y   = outputs.index('Y')
        ind_pi  = outputs.index('pi')
        ind_r   = outputs.index('r')
        ind_C   = outputs.index('C')
        ind_I   = outputs.index('I')
        ind_i   = outputs.index('i')
        ind_ZMY = outputs.index('ZMY')
        ind_A   = outputs.index('A')
        ind_B   = outputs.index('B')

        ind_Beta_a_zm  = outputs.index('Beta_a_zm')
        ind_Beta_b_zm  = outputs.index('Beta_b_zm')
        ind_Beta_c_zm  = outputs.index('Beta_c_zm')

        ind_Corr_c_a   = outputs.index('Corr_c_a')
        ind_Corr_c_b   = outputs.index('Corr_c_b')
        ind_Cov_c_a    = outputs.index('Cov_c_a')
        ind_Cov_c_b    = outputs.index('Cov_c_b')

        ind_Corr_a_zm  = outputs.index('Corr_a_zm')
        ind_Corr_b_zm  = outputs.index('Corr_b_zm')
        ind_Cov_a_zm   = outputs.index('Cov_a_zm')
        ind_Cov_b_zm   = outputs.index('Cov_b_zm')

        ind_Beta_a_zmY = outputs.index('Beta_a_zmY')
        ind_Beta_b_zmY = outputs.index('Beta_b_zmY')


        # Variance of endogenous aggregates
        muhats[0] = Sigma[0, ind_Y, ind_Y]
        muhats[1] = Sigma[0, ind_pi, ind_pi]
        muhats[2] = Sigma[0, ind_r, ind_r]
        muhats[3] = Sigma[0, ind_C, ind_C]
        muhats[4] = Sigma[0, ind_I, ind_I]
        muhats[5] = Sigma[0, ind_i, ind_i]

        # 1st order autocorrelation of each endog aggregate
        ind = 6
        muhats[ind+0] = Sigma[1, ind_Y,  ind_Y]   / Sigma[0, ind_Y,  ind_Y]
        muhats[ind+1] = Sigma[1, ind_pi, ind_pi]  / Sigma[0, ind_pi, ind_pi]
        muhats[ind+2] = Sigma[1, ind_r,  ind_r]   / Sigma[0, ind_r,  ind_r]
        muhats[ind+3] = Sigma[1, ind_C,  ind_C]   / Sigma[0, ind_C,  ind_C]
        muhats[ind+4] = Sigma[1, ind_I,  ind_I]   / Sigma[0, ind_I,  ind_I]
        muhats[ind+5] = Sigma[1, ind_i,  ind_i]   / Sigma[0, ind_i,  ind_i]

        # Contemporaneous covariance between endog aggs
        ind = ind+6
        muhats[ind+0]  = Sigma[0, ind_Y,  ind_pi]  /  np.sqrt(Sigma[0, ind_Y , ind_Y ] * Sigma[0, ind_pi, ind_pi])
        muhats[ind+1]  = Sigma[0, ind_Y,  ind_r]   /  np.sqrt(Sigma[0, ind_Y , ind_Y ] * Sigma[0, ind_r , ind_r ])
        muhats[ind+2]  = Sigma[0, ind_Y,  ind_i]   /  np.sqrt(Sigma[0, ind_Y , ind_Y ] * Sigma[0, ind_i , ind_i ])
        muhats[ind+3]  = Sigma[0, ind_Y,  ind_C]   /  np.sqrt(Sigma[0, ind_Y , ind_Y ] * Sigma[0, ind_C , ind_C ])
        muhats[ind+4]  = Sigma[0, ind_pi, ind_r]   /  np.sqrt(Sigma[0, ind_pi, ind_pi] * Sigma[0, ind_r , ind_r ])
        muhats[ind+5]  = Sigma[0, ind_pi, ind_I]   /  np.sqrt(Sigma[0, ind_pi, ind_pi] * Sigma[0, ind_I , ind_I ])
        muhats[ind+6]  = Sigma[0, ind_C,  ind_pi]  /  np.sqrt(Sigma[0, ind_C , ind_C ] * Sigma[0, ind_pi, ind_pi])
        muhats[ind+7]  = Sigma[0, ind_C,  ind_I]   /  np.sqrt(Sigma[0, ind_C , ind_C ] * Sigma[0, ind_I , ind_I ])
        muhats[ind+8]  = Sigma[0, ind_C,  ind_i]   /  np.sqrt(Sigma[0, ind_C , ind_C ] * Sigma[0, ind_i , ind_i ])
        muhats[ind+9]  = Sigma[0, ind_C,  ind_r]   /  np.sqrt(Sigma[0, ind_C , ind_C ] * Sigma[0, ind_r , ind_r ])
        muhats[ind+10] = Sigma[0, ind_I,  ind_i]   /  np.sqrt(Sigma[0, ind_I , ind_I ] * Sigma[0, ind_i , ind_i ])
        muhats[ind+11] = Sigma[0, ind_I,  ind_r]   /  np.sqrt(Sigma[0, ind_I , ind_I ] * Sigma[0, ind_r , ind_r ])


        # Variance of regression and corr coeffs
        ind= ind + 12

        muhats[ind+0]  = Sigma[0, ind_Corr_c_a  , ind_Corr_c_a ]    # Var(Corr_t(c,a))
        muhats[ind+1]  = Sigma[0, ind_Corr_c_b  , ind_Corr_c_b ]    # Var(Corr_t(c,b))
        muhats[ind+2]  = Sigma[0, ind_Corr_a_zm , ind_Corr_a_zm ]   # Var(Corr_t(a,y))
        muhats[ind+3]  = Sigma[0, ind_Corr_b_zm , ind_Corr_b_zm ]   # Var(Corr_t(b,y))

        muhats[ind+4]  = Sigma[0, ind_Cov_c_a  , ind_Cov_c_a ]      # Var(Cov_t(c,a))
        muhats[ind+5]  = Sigma[0, ind_Cov_c_b  , ind_Cov_c_b ]      # Var(Cov_t(c,b))
        muhats[ind+6]  = Sigma[0, ind_Cov_a_zm , ind_Cov_a_zm ]     # Var(Cov_t(y,a))
        muhats[ind+7]  = Sigma[0, ind_Cov_b_zm , ind_Cov_b_zm ]     # Var(Cov_t(y,b))

        muhats[ind+8]  = Sigma[0, ind_Beta_a_zm , ind_Beta_a_zm ]   # Var(Beta_t(a,y))
        muhats[ind+9]  = Sigma[0, ind_Beta_b_zm , ind_Beta_b_zm ]   # Var(Beta_t(b,y))
        muhats[ind+10] = Sigma[0, ind_Beta_c_zm , ind_Beta_c_zm ]   # Var(Beta_t(c,y))


        # Corr/Cov of reg (or corr coeffs) with Y
        ind= ind + 11
        muhats[ind+0]  = Sigma[0, ind_Corr_c_a , ind_Y]  / np.sqrt(Sigma[0, ind_Corr_c_a ,  ind_Corr_c_a ] * Sigma[0,ind_Y,ind_Y])   # Corr(Corr_t(c,a), Y)
        muhats[ind+1]  = Sigma[0, ind_Corr_c_b , ind_Y]  / np.sqrt(Sigma[0, ind_Corr_c_b ,  ind_Corr_c_b ] * Sigma[0,ind_Y,ind_Y])   # Corr(Corr_t(c,b), Y)
        muhats[ind+2]  = Sigma[0, ind_Corr_a_zm , ind_Y] / np.sqrt(Sigma[0, ind_Corr_a_zm , ind_Corr_a_zm ] * Sigma[0,ind_Y,ind_Y])  # Corr(Corr_t(y,a), Y)
        muhats[ind+3]  = Sigma[0, ind_Corr_b_zm , ind_Y] / np.sqrt(Sigma[0, ind_Corr_b_zm , ind_Corr_b_zm ] * Sigma[0,ind_Y,ind_Y])  # Corr(Corr_t(y,b), Y)

        muhats[ind+4]  = Sigma[0, ind_Cov_c_a , ind_Y]  / np.sqrt(Sigma[0, ind_Cov_c_a ,  ind_Cov_c_a ] * Sigma[0,ind_Y,ind_Y])      # Corr(Cov_t(c,a), Y)
        muhats[ind+5]  = Sigma[0, ind_Cov_c_b , ind_Y]  / np.sqrt(Sigma[0, ind_Cov_c_b ,  ind_Cov_c_b ] * Sigma[0,ind_Y,ind_Y])      # Corr(Cov_t(c,b), Y)
        muhats[ind+6]  = Sigma[0, ind_Cov_a_zm , ind_Y] / np.sqrt(Sigma[0, ind_Cov_a_zm , ind_Cov_a_zm ] * Sigma[0,ind_Y,ind_Y])     # Corr(Cov_t(y,a), Y)
        muhats[ind+7]  = Sigma[0, ind_Cov_b_zm , ind_Y] / np.sqrt(Sigma[0, ind_Cov_b_zm , ind_Cov_b_zm ] * Sigma[0,ind_Y,ind_Y])     # Corr(Cov_t(y,b), Y)

        muhats[ind+8]  = Sigma[0, ind_Beta_a_zm , ind_Y] / np.sqrt(Sigma[0, ind_Beta_a_zm , ind_Beta_a_zm ] * Sigma[0,ind_Y,ind_Y])  # Corr(Beta_t(a,y), Y)
        muhats[ind+9]  = Sigma[0, ind_Beta_b_zm , ind_Y] / np.sqrt(Sigma[0, ind_Beta_b_zm , ind_Beta_b_zm ] * Sigma[0,ind_Y,ind_Y])  # Corr(Beta_t(b,y), Y)

        muhats[ind+10] = Sigma[0, ind_Beta_a_zm , ind_Y]                                                                             # Cov(Beta_t(a,y), Y)
        muhats[ind+11] = Sigma[0, ind_Beta_b_zm , ind_Y]                                                                             # Cov(Beta_t(b,y), Y)

        muhats[ind+12] = (modelInfo['ss']['Y_Cov_a_zm'] + Sigma[0, ind_ZMY, ind_A]) / (modelInfo['ss']['Y2_Var_zm']  + Sigma[0, ind_ZMY,  ind_ZMY])
        muhats[ind+13] = (modelInfo['ss']['Y_Cov_b_zm'] + Sigma[0, ind_ZMY, ind_B]) / (modelInfo['ss']['Y2_Var_zm']  + Sigma[0, ind_ZMY,  ind_ZMY])



    elif (runcode == 4) or (runcode == 5):
        muhats = np.zeros(25)

        # Indices of exogenous aggregates
        ind_Y   = outputs.index('Y')
        ind_pi  = outputs.index('pi')
        ind_r   = outputs.index('r')
        ind_C   = outputs.index('C')
        ind_I   = outputs.index('I')
        ind_i   = outputs.index('i')
        ind_ZM  = outputs.index('ZM')
        ind_ZMY = outputs.index('ZMY')
        ind_AZM  = outputs.index('AZM')
        ind_BZM  = outputs.index('BZM')
        ind_AY  = outputs.index('AY')
        ind_BY  = outputs.index('BY')
        ind_A   = outputs.index('A')
        ind_B   = outputs.index('B')

        ind_Beta_c_zm  = outputs.index('Beta_c_zm')
        ind_Beta_c_a   = outputs.index('Beta_c_a')
        ind_Beta_c_b   = outputs.index('Beta_c_b')
        ind_Beta_a_zm  = outputs.index('Beta_a_zm')
        ind_Beta_b_zm  = outputs.index('Beta_b_zm')
        ind_Beta_a_zmY = outputs.index('Beta_a_zmY')
        ind_Beta_b_zmY = outputs.index('Beta_b_zmY')

        ind_Cov_a_zm   = outputs.index('Cov_a_zm')
        ind_Cov_b_zm   = outputs.index('Cov_b_zm')
        ind_Corr_a_zm  = outputs.index('Corr_a_zm')
        ind_Corr_b_zm  = outputs.index('Corr_b_zm')

        ind_Y_Cov_a_zm = outputs.index('Y_Cov_a_zm')
        ind_Y_Cov_b_zm = outputs.index('Y_Cov_b_zm')

        ZM  = modelInfo['ss']['ZM']
        A   = modelInfo['ss']['A']
        Y   = modelInfo['ss']['Y']
        AY  = modelInfo['ss']['AY']
        ZMY = modelInfo['ss']['ZMY']

        # Variance of endogenous aggregates
        muhats[0] = Sigma[0, ind_Y, ind_Y]
        muhats[1] = Sigma[0, ind_pi, ind_pi]

        # 1st order autocorrelation of each endog aggregate
        ind = 2
        muhats[ind+0] = Sigma[1, ind_Y, ind_Y]   / Sigma[0, ind_Y, ind_Y]
        muhats[ind+1] = Sigma[1, ind_pi, ind_pi] / Sigma[0, ind_pi, ind_pi]
        muhats[ind+2] = Sigma[1, ind_i, ind_i]   / Sigma[0, ind_i, ind_i]
        muhats[ind+3] = Sigma[1, ind_I, ind_I]   / Sigma[0, ind_I, ind_I]

        # Contemporaneous covariance between endog aggs
        ind = ind+4
        muhats[ind+0]  = Sigma[0, ind_Y, ind_pi]   /  np.sqrt(Sigma[0, ind_Y , ind_Y ] * Sigma[0, ind_pi, ind_pi])
        muhats[ind+1]  = Sigma[0, ind_Y, ind_i]    /  np.sqrt(Sigma[0, ind_Y , ind_Y ] * Sigma[0, ind_i , ind_i ])
        muhats[ind+2]  = Sigma[0, ind_Y, ind_C]    /  np.sqrt(Sigma[0, ind_Y , ind_Y ] * Sigma[0, ind_C , ind_C ])
        muhats[ind+3]  = Sigma[0, ind_pi, ind_i]   /  np.sqrt(Sigma[0, ind_pi, ind_pi] * Sigma[0, ind_i , ind_i ])
        muhats[ind+4]  = Sigma[0, ind_pi, ind_I]   /  np.sqrt(Sigma[0, ind_pi, ind_pi] * Sigma[0, ind_I , ind_I ])
        muhats[ind+5]  = Sigma[0, ind_C, ind_I]    /  np.sqrt(Sigma[0, ind_C , ind_C ] * Sigma[0, ind_I , ind_I ])
        muhats[ind+6]  = Sigma[0, ind_C, ind_i]    /  np.sqrt(Sigma[0, ind_C , ind_C ] * Sigma[0, ind_i , ind_i ])
        muhats[ind+7]  = Sigma[0, ind_C, ind_pi]    /  np.sqrt(Sigma[0, ind_C , ind_C ] * Sigma[0, ind_pi , ind_pi ])
        muhats[ind+8]  = Sigma[0, ind_I, ind_i]    /  np.sqrt(Sigma[0, ind_I , ind_I ] * Sigma[0, ind_i , ind_i ])

        # Variance of regression and corr coeffs
        ind= ind + 9
        muhats[ind+0] = Sigma[0, ind_Beta_b_zm , ind_Beta_b_zm ]

        # Corr of reg/corr coeffs and Y
        ind = ind + 1
        muhats[ind+0] = Sigma[0, ind_Beta_a_zmY, ind_Y] / np.sqrt(Sigma[0, ind_Beta_a_zmY, ind_Beta_a_zmY] * Sigma[0,ind_Y,ind_Y])
        muhats[ind+1] = Sigma[0, ind_Beta_b_zmY, ind_Y] / np.sqrt(Sigma[0, ind_Beta_b_zmY, ind_Beta_b_zmY] * Sigma[0,ind_Y,ind_Y])

        # Regression coefficients
        ind = ind+2
        muhats[ind+0] = (modelInfo['ss']['Y_Cov_a_zm'] + Sigma[0, ind_ZMY,  ind_A]) / \
                        (modelInfo['ss']['Y2_Var_zm']  + Sigma[0, ind_ZMY,  ind_ZMY])

    return muhats


def GetIdentificationLabels(runcode, outputs):

    if (runcode == 1):
        labs = ['Var(Y)',
                'Var(pi)',
                'Var(r)',
                'AutoCorr(Y)',
                'AutoCorr(pi)',
                'AutoCorr(r)',
                'Cov(Y, pi)',
                'Cov(Y, r)',
                'Cov(pi, r)',
                'Var(beta_C_Z)',
                'Var(Beta_c_a)',
                'Var(Beta_c_b)']

    elif (runcode == 2) or (runcode == 3):
        labs = [
                'Var(Y)',
                'Var(pi)',
                'Var(r)',
                'Var(C)',
                'Var(I)',
                'Var(i)',
                'AutoCorr(Y)',
                'AutoCorr(pi)',
                'AutoCorr(r)',
                'AutoCorr(C)',
                'AutoCorr(I)',
                'AutoCorr(i)',
                'Corr(Y,pi)',
                'Corr(Y,r)',
                'Corr(Y,i)',
                'Corr(Y,C)',
                'Corr(pi,r)',
                'Corr(pi,I)',
                'Corr(C,pi)',
                'Corr(C,I)',
                'Corr(C,i)',
                'Corr(C,r)',
                'Corr(I,i)',
                'Corr(I,r)',
                'Var(Corr_t(c,a))',
                'Var(Corr_t(c,b))',
                'Var(Corr_t(y,a))',
                'Var(Corr_t(y,b))',
                'Var(Cov_t(c,a))',
                'Var(Cov_t(c,b))',
                'Var(Cov_t(y,a))',
                'Var(Cov_t(y,b))',
                'Var(Beta_t(a,y))',
                'Var(Beta_t(b,y))',
                'Var(Beta_t(c,y))',
                'Corr(Corr_t(c,a),Y)',
                'Corr(Corr_t(c,b),Y)',
                'Corr(Corr_t(y,a),Y)',
                'Corr(Corr_t(y,b),Y)',
                'Corr(Cov_t(c,a),Y)',
                'Corr(Cov_t(c,b),Y)',
                'Corr(Cov_t(y,a),Y)',
                'Corr(Cov_t(y,b),Y)',
                'Corr(Beta_t(a,y),Y)',
                'Corr(Beta_t(b,y),Y)',
                'Cov(Beta_t(a,y),Y)',
                'Cov(Beta_t(b,y),Y)',
                'plim beta(a,yY)',
                'plim beta(b,yY)'
                ]
    elif (runcode == 4) or (runcode == 5):
        labs = [
                'Var(Y)',
                'Var(pi)',
                'AutoCorr(Y)',
                'AutoCorr(pi)',
                'AutoCorr(i)',
                'AutoCorr(I)',
                'Corr(Y,pi)',
                'Corr(Y,i)',
                'Corr(Y,C)',
                'Corr(pi,i)',
                'Corr(pi,I)',
                'Corr(C,I)',
                'Corr(C,i)',
                'Corr(C,pi)',
                'Corr(I,i)',
                'Var(Beta_b_zm)',
                'Corr(Beta_a_zmY,Y)',
                'Corr(Beta_b_zmY,Y)',
                ]
    else:
        labs = []
        for m in outputs:
            labs.append("Var(" + m + ")")
            labs.append("AutoCov(" + m + ")")

        Noutputs = len(outputs)
        for m in range(Noutputs):
            for n in range(Noutputs):
                if n > m:
                    labs.append("Cov(" + outputs[m] + "," + outputs[n] + ")")

    return labs


########################################################################
## h() Function for Moment-Matching ####################################
########################################################################

def thetaDictToVector(theta_dict, param_names):
    return np.array([theta_dict[param] for param in param_names])

def thetaVectorToDict(theta_vector, param_names):
    return {param : theta_vector[i] for i, param in enumerate(param_names)}



# h() fcn that, given params, returns moments computed analytically or
# empirically
#
# Takes in modelInfo bc assume model already initialized, so we can use
# saved and save time by not resolving for the steady state. This means
# we're only allowed to change params that leave steady state unchanged.
#
# theta is a parameter vector, same size as modelInfo['param_names']
#
def h(modelInfo, theta_vector, Npd, empirical=False, simFromMA=False, Nsim=1000, verbose=False, muhatsRet=False):

    ## RESOLVE MODEL for given parameters ##
    # Don't initialize when calling SolveModel bc assuming model already initialized
    tic                = time.perf_counter()
    modelInfo['theta'] = thetaVectorToDict(theta_vector, modelInfo['param_names'])
    modelInfo          = SolveModel(modelInfo, False) # Shouldn't be initializing at this step
    if verbose: print("Solving: %f" % (time.perf_counter()-tic))


    # RECOMPUTE IRFs given parameter values, checking for issues
    tic = time.perf_counter()
    mZs = getShockMACoeffs(modelInfo)
    if np.max([mZs[i].max() for i in modelInfo['exogenous']]) > 100:
        print('Failing')
        print(theta_vector)
        print(mZs)
        raise Exception('Large mZ component')
    if verbose: print("IRFs: %f" % (time.perf_counter()-tic))


    # COMPUTE MOMENTS: Empirical, else analytical
    inputs, outputs   = modelInfo['exogenous'], modelInfo['outputs']
    Ninputs, Noutputs = len(inputs), len(outputs)
    if empirical:
        tic = time.perf_counter()
        # Simulate data
        dX = SimModel(modelInfo, mZs, Npd, Nsim, simFromMA, 314)

        # Compute muhat and average over draws
        muhats = compute_muhat(modelInfo, dX)
        toret = np.mean(muhats, axis=1)
        if verbose:
            print("Moments: %f" % (time.perf_counter()-tic))

    # Analytical
    else:
        tic   = time.perf_counter()
        toret = h_analytical(modelInfo, mZs)
        if verbose:
            print("Moments: %f" % (time.perf_counter()-tic))

    if muhatsRet:
        return toret, muhats
    else:
        return toret



# Compare h when generated by simulation vs. analytically
# Note that if Npd small, many unconditional empirical moments are
# likely to differ substantially from the analytical moments given that
# the series are very persistent, and unconditional moments may take a
# long time to converge.
def comparehEmpiricalAnalytical(modelInfo, Npd, Nsim):

    # Set theta0
    theta0 = np.array([modelInfo['theta0'][param] for param in modelInfo['param_names']])

    ## Test h fcn and make sure they all compute the same thing
    # print("")
    print('Testing h, empirical, sim from MA')
    h_empirical_MA, muhats_MA = h(modelInfo, theta0, Npd, empirical=True, simFromMA=True, Nsim=Nsim, muhatsRet=True)

    print('Testing h, empirical, direct draw')
    h_empirical_draw, muhats_draw = h(modelInfo, theta0, Npd, empirical=True, simFromMA=False, Nsim=Nsim, muhatsRet=True)

    print('Testing analytical h')
    h_analytical = h(modelInfo, theta0, Npd, empirical=False)

    print(np.vstack([h_empirical_MA, h_empirical_draw, h_analytical]).T)


    ## Plot muhat histograms against the analytical answer to compare
    Nh = len(h_analytical)
    # print(muhats_MA.shape)
    for m in range(Nh):
        plt.figure()
        plt.subplot(2,1,1)
        plt.hist(muhats_MA[m,:])
        plt.axvline(h_analytical[m], color='k', linestyle='dashed')
        plt.title(('Moment %d' % m))

        plt.subplot(2,1,2)
        plt.hist(muhats_draw[m,:])
        plt.axvline(h_analytical[m], color='k', linestyle='dashed')

        plt.show()


########################################################################
## Moment Matching #####################################################
########################################################################


def MomentMatch(runcode, exogenous, unknowns, targets, outputs, theta0, block_list, T, ss, G, Npd, Nsim, hempirical, hfromMA, muhat, Vhat, Nobs, lamb, l, u, fullyFeasible):

    cv        = 1.96
    modelInfo = {\
                'runcode' : runcode,
                'exogenous' : exogenous,
                'unknowns' : unknowns,
                'targets' : targets,
                'outputs' : outputs,
                'theta0' : theta0,
                'block_list' : block_list,
                'T' : T,
                'ss' : ss,
                'G' : G,
                'param_names' : param_names
                }
    h_        = lambda theta: h(modelInfo, theta, Npd, hempirical, hfromMA, 200, False)
    h_theta0  = h_(modelInfo['theta0'])
    h__       = lambda theta: h_(theta).reshape((len(h_theta0),1))
    Gfcn_     = lambda theta: wcopt.ComputeG(h__,theta)
    res       = wcopt.TestSingle(muhat, Vhat, Nobs, lamb, modelInfo['theta0'], modelInfo['theta0'], cv, l, u, fullyFeasible, Gfcn_, h__, True, True)
    return res



def MomentMatchRuns(runcode, Npd, Nsim, fromMA, hempirical, hfromMA, nparallel=0, seed=314):

    # INITIALIZE
    modelInfo = SolveModel({'runcode' : runcode}, True)
    K = len(modelInfo['param_names'])
    I = np.eye(K)


    # Load simulated moments for matching
    strSave  = getStrSave(runcode, modelInfo['T'], Npd, Nsim, fromMA)
    savepath = "./Results/muhats_" + strSave + ".npy"
    muhats   = np.load(savepath)


    # Setup up parallel pool
    if nparallel != 0:
        pool = mp.Pool(nparallel)

    # Compute Vhat to use for each, so we won't do fully feasible
    Nobs = 1
    Vhat = Nobs*np.cov(muhats)

    # Set bounds and run
    if (runcode == 4):
      l = np.array([1.10, 0.001, 0.10, 0.001, 0.10, 0.10, 0.001, 0.001, 0.001]).reshape((9,1))
      u = np.array([3,       1,  0.99,    1,  0.99, 0.99,   0.5,   0.5, 0.5]).reshape((9,1))
    else:
      l = np.array([1.05, 0.001, 0.10]).reshape((3,1))
      u = np.array([3,        1, 0.99]).reshape((3,1))
    res = [[] for k in range(K)]
    for k, param in enumerate(theta0):
        print("\tParam %d / %d" % (k+1, K))

        if nparallel == 0:
            for s in range(Nsim):
                print("Matching Simulation %d / %d" % (s+1, Nsim))

                res[k].append(MomentMatch(modelInfo['runcode'], modelInfo['exogenous'], modelInfo['unknowns'],
                                          modelInfo['targets'], modelInfo['outputs'], modelInfo['theta0'],
                                          modelInfo['block_list'], modelInfo['T'], modelInfo['ss'], modelInfo['G'],
                                          Npd, Nsim, hempirical, hfromMA, muhats[:,s], Vhat, Nobs, I[:,k:(k+1)], l, u, False))
        else:
            pool = mp.Pool(nparallel)
            res[k] = pool.starmap(MomentMatch,
                                  [(modelInfo['runcode'], modelInfo['exogenous'], modelInfo['unknowns'],
                                    modelInfo['targets'], modelInfo['outputs'], modelInfo['theta0'],
                                    modelInfo['block_list'], modelInfo['T'], modelInfo['ss'], modelInfo['G'],
                                    Npd, Nsim, hempirical, hfromMA, muhats[:,s], Vhat, Nobs, I[:,k:(k+1)], l, u, False) for s in range(Nsim)]
                                  )
            pool.close()

    np.save("res_" + strSave + ".npy", res)
    return res





########################################################################
## Features by code ####################################################
########################################################################

# Set parameter defaults and whether the parameter is specifically an
# exogenous shock parameter, rather than a structural economic param
def paramsDefaults(runcode):
    if runcode == 0:
        theta0       = {'phi' : 1.5}
        shock_params = {'phi' : False}
        param_names  = ['phi']
        param_bounds = {'phi'    : np.array([1.050, 3.00])}

    elif (runcode == 1) or (runcode == 2) or (runcode == 5):
        theta0       = {'phi'    : 1.5,
                        'kappap' : 0.1,
                        'rho_Z'  : 0.85
                       }
        shock_params = {'phi'    : False,
                        'kappap' : False,
                        'rho_Z'  : False
                       }
        param_names  = ['phi', 'kappap', 'rho_Z']
        param_bounds = {'phi'    : np.array([1.050, 3.00]),
                        'kappap' : np.array([0.001, 1.00]),
                        'rho_Z'  : np.array([0.100, 0.99])}

    elif (runcode == 3) or (runcode == 4):
        theta0       = {'phi'         : 1.50,
                        'kappap'      : 0.10,
                        'rho_Z'       : 0.85,
                        'kappaw'      : 0.10,
                        'rho_rstar'   : 0.85,
                        'rho_beta'    : 0.85,
                        'sigma_Z'     : 0.01,
                        'sigma_rstar' : 0.01,
                        'sigma_beta'  : 0.01
                       }
        shock_params = {'phi'         : False,
                        'kappap'      : False,
                        'rho_Z'       : False,
                        'kappaw'      : False,
                        'rho_rstar'   : False,
                        'rho_beta'    : False,
                        'sigma_Z'     : False,
                        'sigma_rstar' : False,
                        'sigma_beta'  : False
                       }
        param_names  = ['phi', 'kappap', 'rho_Z', 'kappaw',
                        'rho_rstar', 'rho_beta',
                        'sigma_Z', 'sigma_rstar', 'sigma_beta']
        param_bounds = {'phi'         : np.array([1.100, 3.000]),
                        'kappap'      : np.array([0.001, 1.000]),
                        'rho_Z'       : np.array([0.010, 0.990]),
                        'kappaw'      : np.array([0.001, 1.000]),
                        'rho_rstar'   : np.array([0.010, 0.990]),
                        'rho_beta'    : np.array([0.010, 0.990]),
                        'sigma_Z'     : np.array([0.001, 0.500]),
                        'sigma_rstar' : np.array([0.001, 0.500]),
                        'sigma_beta'  : np.array([0.001, 0.500])}
    return theta0, shock_params, param_names, param_bounds


# For each parameter, establish a range and a default
def paramsCheckID(runcode):
    if runcode == 0:
        thetas = {'phi' : np.linspace(start=1.01, stop=2.0, num=20)}

    elif runcode == 1:
        thetas = {'phi'    : np.linspace(start=1.01, stop=1.9, num=5),
                  'kappap' : np.linspace(start=.01, stop=0.2, num=5),
                  'rho_Z'  : np.linspace(start=0.05, stop=0.95, num=20)}

    elif (runcode == 2) or (runcode == 5):
        thetas = {'phi'    : np.linspace(start=1.10, stop=3, num=20),
                  'kappap' : np.linspace(start=.001, stop=1, num=20),
                  'rho_Z'  : np.linspace(start=0.10, stop=0.99, num=20)}

    elif (runcode == 3) or (runcode == 4):
        thetas = {'phi'         : np.linspace(start=1.10, stop=3, num=20),
                  'kappap'      : np.linspace(start=.001, stop=1, num=20),
                  'rho_Z'       : np.linspace(start=0.10, stop=0.99, num=20),
                  'kappaw'      : np.linspace(start=.001, stop=1, num=20),
                  'rho_rstar'   : np.linspace(start=0.1, stop=0.99, num=20),
                  'rho_beta'    : np.linspace(start=0.1, stop=0.99, num=20),
                  'sigma_Z'     : np.linspace(start=0.001, stop=0.5, num=20),
                  'sigma_rstar' : np.linspace(start=0.001, stop=0.5, num=20),
                  'sigma_beta'  : np.linspace(start=0.001, stop=0.5, num=20)}

        # eiss     = np.linspace(start=0.1, stop=2, num=20)
        # deltas   = np.linspace(start=0.05, stop=0.1, num=20)
        # alphas   = np.linspace(start=0.2, stop=0.45, num=20)
        # frischs  = np.linspace(start=0.5, stop=2, num=20)

    theta0, shock_params, param_names, param_bounds = paramsDefaults(runcode)

    return thetas, theta0



# Set features of the model that are needed to solve the model and
# simulate from the model
#
# exogenous: Exogenous shocks in the model
# outputs:   Outputs to compute and store jacobians for (subset of outputs_all)
# unknowns:  Unknowns endogenous variables in the model
# targets:   Model equations to solve
def getruncodeFeatures(runcode):

    # Set shocks and outputs to carry around
    if (runcode == 2) or (runcode == 3) or (runcode == 4) or (runcode == 5):
        exogenous = ['Z', 'rstar', 'G', 'markup', 'markup_w', 'beta', 'rinv_shock']
        outputs   = ['Y', 'pi', 'r', 'C', 'I', 'i', 'A', 'B', 'ZMY',
                     'AY', 'BY', 'AZM', 'BZM', 'AZMY', 'BZMY', 'ZM',
                     'Var_a', 'Var_b', 'Var_c', 'Var_zm',
                     'Cov_c_a', 'Cov_c_b', 'Cov_a_zm', 'Cov_b_zm', 'Cov_c_zm', 'Cov_a_zmY', 'Cov_b_zmY',
                     'Corr_c_a', 'Corr_c_b', 'Corr_a_zm', 'Corr_b_zm',
                     'Y_Cov_a_zm', 'Y_Cov_b_zm', 'Y_Cov_c_zm', 'Y2_Var_zm',
                     'Beta_a_zm', 'Beta_b_zm', 'Beta_c_zm', 'Beta_c_a', 'Beta_c_b',
                     'Beta_a_zmY', 'Beta_b_zmY'
                     ]
    else:
        exogenous = ['Z', 'rstar', 'G']
        outputs = ['Y', 'r', 'pi', 'Beta_c_zm', 'Beta_c_a', 'Beta_c_b']
        # outputs = ['Y', 'C', 'K']

    # Unkowns and equations that we solve for those unkowns
    unknowns  = ['Y', 'r', 'w']
    targets = ['asset_mkt', 'fisher', 'wnkpc']

    # Get parameter defaults
    theta0, shock_params, param_names, param_bounds = paramsDefaults(runcode)

    # Blocks
    block_list = [household_inc, pricing, arbitrage, production,
                  dividend, taylor, fiscal, finance, wage, union, mkt_clearing,
                  microMoments]

    # Pack into dictionary
    dt = {'exogenous'    : exogenous,
          'unknowns'     : unknowns,
          'targets'      : targets,
          'outputs'      : outputs,
          'param_names'  : param_names,
          'param_bounds' : param_bounds,
          'theta0'       : theta0,
          'shock_params' : shock_params,
          'block_list'   : block_list,
          'T'            : 300
          }

    return dt



def getStrSave(runcode, T, Npd, Nsim, fromMA):
    strSave = "runcode%d_T%d_Npd%d_Nsim%d_fromMA%d" % (runcode, T, Npd, Nsim, fromMA)
    return strSave


def quadratic_form(h, W):
    return np.vdot(h, np.matmul(W, h))





########################################################################
## Running #############################################################
########################################################################


def muhatsSim(runcode, Npd, Nsim, fromMA, save=True, verbose=False, seed=314):

    # Initailize model
    modelInfo = SolveModel({'runcode' : runcode}, True)

    # Get MA coefficients
    mZs = getShockMACoeffs(modelInfo)

    # Simulate aggregates
    tic = time.perf_counter()
    dY      = SimModel(modelInfo, mZs, Npd, Nsim, fromMA, seed)
    dY_dict = SimsArrayToDict(dY, modelInfo['outputs'])
    if verbose:
        print(time.perf_counter()-tic)

    # Compute moments
    muhats = compute_muhat(modelInfo, dY)

    # Save moments
    strSave  = getStrSave(runcode, modelInfo['T'], Npd, Nsim, fromMA)
    savepath = "./Results/muhats_" + strSave + ".npy"
    np.save(savepath, muhats)

    return muhats, dY, dY_dict


########################################################################
## Identification Plots ################################################
########################################################################


# Identification plots
def identificationPlots(runcode, Npd, Nsim, hempirical, hsimFromMA, fromMA, NsimPlot, saving=True):

    ## INITIALIZE ##
    print("Initializing Model...")
    modelInfo   = SolveModel({'runcode' : runcode}, True)
    param_names = modelInfo['param_names']

    ## Get parameter values to loop over for ID plots ##
    thetas_ID, theta0_dict = paramsCheckID(runcode)
    theta0_vector          = thetaDictToVector(theta0_dict, param_names)
    Nparams                = len(theta0_vector)

    ## Evaluate moments at initial parameters ##
    print("Getting moments at initial params")
    Nmoments = len(h(modelInfo, theta0_vector, Npd, hempirical, hsimFromMA, Nsim, False))

    ## Load simulated moments for matching ##
    strSave  = getStrSave(runcode, modelInfo['T'], Npd, Nsim, fromMA)
    savepath = "./Results/muhats_" + strSave + ".npy"
    muhats   = np.load(savepath)
    Nobs     = 1
    V        = Nobs*np.cov(muhats) # Compute Vhat to use for each, so we won't do fully feasible

    ## Loop over parameter values and generate identification plots for moments
    print("ID Plots, Individual Moments")
    plt.close('all')
    labs = GetIdentificationLabels(runcode, list(modelInfo['outputs_all']))
    for p, param in enumerate(param_names):
        print("Param %d / %d..." % (p+1, Nparams))

        # Set everything to defaults to start
        theta_vector = copy.deepcopy(theta0_vector)
        # print(theta0)

        # Create array to hold values of moments across different parameter vals
        Nid = len(thetas_ID[param])
        h_ = np.zeros((Nmoments, Nid))

        # Loop over possible values for this particular parameter
        for i in range(Nid):
            # Set that parameter
            theta_vector[p] = thetas_ID[param][i]
            # print(theta)

            # Compute h at this param and plot all moments
            h_[:,i] = h(modelInfo, theta_vector,  Npd, hempirical, hsimFromMA, Nsim, False)
            for m in range(Nmoments):
                if not saving:
                    plt.subplot(np.round(np.ceil(Nmoments/4)), 4, m+1)
                plt.plot(thetas_ID[param], h_[m,:])
                plt.title(labs[m])
                if saving:
                    ## FIX BY SAVING NAME
                    plt.savefig("Plots/ID_Moments_runcode%d_Param%d_%s.pdf" % (modelInfo['runcode'], p+1, labs[m]))
                    plt.close('all')
                else:
                    plt.show()


    ## Loop over parameter values and generate identification plots for obj fcn
    print("ID Plots, Objfcn")
    plt.close("all")
    I = np.eye(Nparams)
    for s in range(NsimPlot):
        print("Simulation %d / %d" % (s+1, NsimPlot))

        # Form objective function
        h_     = lambda theta: h(modelInfo, theta, Npd, hempirical, hsimFromMA, 200, False).reshape((muhats.shape[0],1))
        Gfcn_  = lambda theta: wcopt.ComputeG(h_,theta)
        objfcn = lambda theta, W: wcopt.quadForm(W, muhats[:,s]-h_(theta)).sum()

        # Loop over parameters that we're calibrating
        for p, param in enumerate(param_names):
            print("\tParam %d / %d" % (p+1, Nparams))

            theta_vector = copy.deepcopy(theta0_vector)

            # Loop over parameter values
            Nid = len(thetas_ID[param])
            f   = np.zeros((2,Nid))
            for i in range(Nid):
                # Set parameter
                theta_vector[p] = thetas_ID[param][i]

                # Naive
                # print("")
                # print(muhats[:,s])
                # print(h_(theta))
                # print(V)
                # print("")
                f[0,i] = objfcn(theta_vector, np.linalg.inv(np.diag(np.diag(V))))

                # Opt
                f[1,i] = objfcn(theta_vector, np.linalg.inv(V))

            # Plot
            plt.plot(thetas_ID[param], f[0,:])
            plt.title('Naive')
            plt.savefig("Plots/ID_Objfcn_runcode%d_Param%d_Sim%d_naive.pdf" % (modelInfo['runcode'], p+1, s))
            plt.close("all")

            plt.plot(thetas_ID[param], f[1,:])
            plt.title('Opt')
            plt.savefig("Plots/ID_Objfcn_runcode%d_Param%d_Sim%d_opt.pdf" % (modelInfo['runcode'], p+1, s))
            plt.close("all")





