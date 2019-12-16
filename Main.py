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
import wcopt
import warnings
import copy
import multiprocessing as mp



warnings.simplefilter('error')


# Runcode
# 0: Single parameter, 3 shocks
# 1: Multiple params, 3 shocks
# 2: Multiple params, all shocks


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
# def SolveModel(runcode, theta, T, ss, exogenous, outputs, unknowns, targets, block_list, use_saved):
def SolveModel(modelInfo, init):

    # RUNCODE
    runcode = modelInfo['runcode']

    ## SET PARAMETER VALUES
    # If modelInfo dictionary not already initialized, initalize it
    # Also set parameters theta
    if init:
        modelInfo.update(getruncodeFeatures(runcode))
        theta_use = modelInfo['theta0']
    else:
        theta_use = modelInfo['theta']

    ## STEADY STATE
    # - Solve for it, if non provided
    # - If ss provided, set parameters to whatever is in theta_use,
    #   so we use those when computing Jacobian, not the original
    #   parameter values that were stored in ss when we first solved
    if runcode == 0:
        phi = theta_use[0]
        if 'ss' not in modelInfo:
            print('Computing steady state')
            modelInfo['ss'] = hank_ss(phi=phi, noisy=False)
        else:
            modelInfo['ss']['phi'] = phi

    elif (runcode == 1) or (runcode == 2):
        phi, kappap = theta_use[0:2]
        if 'ss' not in modelInfo:
            print('Computing steady state')
            modelInfo['ss'] = hank_ss(kappap=kappap, phi=phi, noisy=False)
        else:
            modelInfo['ss']['phi']    = phi
            modelInfo['ss']['kappap'] = kappap


    ## JACOBIANS

    # Compute or recompute
    if init: # Get all jacobians
        modelInfo['G'] = jac.get_G(modelInfo['block_list'], modelInfo['exogenous'], modelInfo['unknowns'], modelInfo['targets'], T=modelInfo['T'], ss=modelInfo['ss'], save=True, use_saved=False)
    else:
        modelInfo['G'] = jac.get_G(modelInfo['block_list'], modelInfo['exogenous'], modelInfo['unknowns'], modelInfo['targets'], T=modelInfo['T'], ss=modelInfo['ss'], outputs=modelInfo['outputs'], save=True, use_saved=True)

    # Add impulse responses of exogenous shocks to themselves
    for o in modelInfo['exogenous']:
        modelInfo['G'][o] = {i : np.identity(modelInfo['T'])*(o==i) for i in modelInfo['exogenous']}

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

    # Set up the MA coefficients in the MA(infty) rep which characterize
    # the response (IRF) of the exogenous aggregates to the exogenous
    # shocks
    rho   = {i : 0.9 for i in modelInfo['exogenous']}
    sigma = {i : 0.01 for i in modelInfo['exogenous']}

    if (modelInfo['runcode'] == 1) or (modelInfo['runcode'] == 2):
        rho['Z'] = theta[2]

    mZs = {i : rho[i]**(np.arange(modelInfo['T']))*sigma[i] for i in modelInfo['exogenous']}

    return mZs



# Get MA coefficients (IRFs) for endogenous aggregates
def getEndogMACoeffs(G, mZs, inputs, outputs):
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
    # print("Estimating all covariances")
    # print(mYs.shape)
    # tic = time.perf_counter()
    Sigma = est.all_covariances(mYs, np.ones(Ninputs))
    # print(time.perf_counter()-tic)

    # Build the full covariance matrix
    # print("Building full covariance matrix")
    # tic = time.perf_counter()
    V = est.build_full_covariance_matrix(Sigma, np.zeros(Noutputs), Npd)
    # print(time.perf_counter()-tic)

    # Draw and return aggregate outputs
    # print("Drawing")
    # tic = time.perf_counter()
    # for s in range(Nsim):
        # dY[:,:,s] = np.random.multivariate_normal(np.zeros(Npd*Noutputs), V).reshape((Npd, Noutputs))
    dY = np.random.multivariate_normal(np.zeros(Npd*Noutputs), V, size=Nsim).T.reshape((Npd,Noutputs,Nsim), order='C')
    # print(time.perf_counter()-tic)

    return dY



# Given Jacobians, structural shock MA process, simulate exog and endog
# aggregates Simulating Model Aggregates from either MA rep or from
# direct drawing
def SimModel(modelInfo, mZs, Npd, Nsim, fromMA=True, seed=None):

    G        = modelInfo['G']
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
    mYs_byo_byi, mYs_byi, mYs = getEndogMACoeffs(G, mZs, inputs, outputs)
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
    return V[0,:]


# Compute muhat for a give runcode given provided dataset
def compute_muhat(modelInfo, dt):
    runcode = modelInfo['runcode']

    if (runcode == 0) or (runcode == 2):
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
        outputs = modelInfo['outputs']
        ind_Y   = outputs.index('Y')
        ind_pi  = outputs.index('pi')
        ind_r   = outputs.index('r')
        ind_bCZ = outputs.index('BetaCZ')
        ind_bCA = outputs.index('BetaCA')
        ind_bCB = outputs.index('BetaCB')

        # For each sim, extract moments
        Npd, Nvar, Nsim = dt.shape
        muhats = np.zeros((12, Nsim))
        for s in range(Nsim):
            # print("Calculating moments for %d / %d..." % (s, Nsim))

            # Variance and autocorrelation at various lags
            # Row 0 is variance
            # Row 1 is 1st lag autocorr, etc.
            var_autocovs  = np.apply_along_axis(lambda X: my_autocov(X,1), 0, dt[:,:,s])
            muhats[:3,s]  = var_autocovs[0, np.array([ind_Y, ind_pi, ind_r])]
            muhats[3:6,s] = var_autocovs[1, np.array([ind_Y, ind_pi, ind_r])]

            # Contemporaneous covariance between endog aggregates
            ind = 6
            V               = np.cov(dt[:,:,s].transpose())
            muhats[ind,s]   = V[ind_Y, ind_pi]
            muhats[ind+1,s] = V[ind_Y, ind_r]
            muhats[ind+2,s] = V[ind_pi, ind_r]

            # Compute the variance of the regression coefficients
            ind = 9
            muhats[ind,s]   = V[ind_bCZ, ind_bCZ]
            muhats[ind+1,s] = V[ind_bCA, ind_bCA]
            muhats[ind+2,s] = V[ind_bCB, ind_bCB]

    return muhats


# At each time period, compute the micro variance of things and the reg coeffs
# Take the mean of this quantity over time to get the moments


########################################################################
## Computing Analytical Moments ########################################
########################################################################


# Model h() function that spits out model-implied analytical moments given parameters
def h_analytical(modelInfo, mZs):
    runcode  = modelInfo['runcode']
    inputs   = modelInfo['exogenous']
    outputs  = modelInfo['outputs']
    G        = modelInfo['G']
    Noutputs = len(outputs)
    Ninputs  = len(inputs)

    # Compute MA coeffs for response of outputs to each structural shock
    mXs_byo_byi, mXs_byi, mXs = getEndogMACoeffs(G, mZs, inputs, outputs)

    # Compute covariances analytically, no measurement error
    # Shape is T x Noutputs x Noutputs
    # Gives correlation and autocorrelation at all lags 1,...T betwen
    # all combinations of variables
    Sigma  = est.all_covariances(mXs, np.ones(Ninputs))

    # Return model-implied moments. Which ones depends upon runcode
    ## FIX THIS
    if (runcode == 0) or (runcode == 2):
        Ncov   = int(Noutputs*(Noutputs+1)/2 - Noutputs)
        muhats = np.zeros(Noutputs*2 + Ncov)

        # Var and 1st order autocorrelation of each endog aggregate
        muhats[:(Noutputs*2)] = np.hstack([Sigma[0:2,m,m] for m in range(Noutputs)])

        # Contemporaneous covariance between endog aggregates
        ctr = 0
        for m in range(Noutputs-1):
            # print("SHAPE")
            # print(Sigma.shape)
            Vadd = Sigma[0,m,(m+1):]
            muhats[(Noutputs*2+ctr):(Noutputs*2+ctr+len(Vadd))] = Vadd
            ctr += len(Vadd)


    elif runcode == 1:
        muhats = np.zeros(12)

        # Indices of endgenous aggregates
        ind_Y   = outputs.index('Y')
        ind_pi  = outputs.index('pi')
        ind_r   = outputs.index('r')
        ind_bCZ = outputs.index('BetaCZ')
        ind_bCA = outputs.index('BetaCA')
        ind_bCB = outputs.index('BetaCB')

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

        # Compute the variance of the regression coefficients
        ind = 9
        muhats[ind]   = Sigma[0, ind_bCZ, ind_bCZ]
        muhats[ind+1] = Sigma[0, ind_bCA, ind_bCA]
        muhats[ind+2] = Sigma[0, ind_bCB, ind_bCB]


    return muhats


def GetIdentificationLabels(runcode, outputs):
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


# Model h() function that spits out moments given parameters, can be analytically computed or empirical
#
# Want initial ss and G because can reuse some stuff there
def h(modelInfo, theta, Npd, empirical=False, simFromMA=False, Nsim=1000, verbose=False):

    # Resolve model for given parameters
    # Using saved because assuming model already initialized
    tic                = time.perf_counter()
    modelInfo['theta'] = theta
    modelInfo          = SolveModel(modelInfo, False) # Shouldn't be initializing at this step
    if verbose:
        print("Solving: %f" % (time.perf_counter()-tic))

    # RECOMPUTE IRFs given parameter values
    tic = time.perf_counter()
    mZs = getShockMACoeffs(modelInfo)
    if np.max([mZs[i].max() for i in modelInfo['exogenous']]) > 100:
        print('Failing')
        print(theta)
        print(mZs)
        raise Exception('Large mZ component')
    if verbose:
        print("IRFs: %f" % (time.perf_counter()-tic))

    # COMPUTE MOMENTS: Empirical, else analytical
    inputs   = modelInfo['exogenous']
    outputs  = modelInfo['outputs']
    Ninputs  = len(inputs)
    Noutputs = len(outputs)
    if empirical:
        tic = time.perf_counter()
        # Simulate data
        dX = SimModel(modelInfo, mZs, Npd, Nsim, simFromMA, 314)

        # Compute muhat and average over draws
        toret = np.mean(compute_muhat(modelInfo, dX), axis=1)
        if verbose:
            print("Moments: %f" % (time.perf_counter()-tic))

    # Analytical
    else:
        tic = time.perf_counter()
        toret = h_analytical(modelInfo, mZs)
        if verbose:
            print("Moments: %f" % (time.perf_counter()-tic))

    return toret



########################################################################
## Moment Matching #####################################################
########################################################################


def MomentMatch(runcode, theta0, T, ss, Npd, inputs, outputs, unknowns, targets, block_list, hempirical, hfromMA, Nsim, muhat, Vhat, Nobs, lamb, l, u, fullyFeasible):
    cv       = 1.96
    h_       = lambda theta: h(runcode, theta, T, ss, Npd, inputs, outputs, unknowns, targets, block_list, hempirical, hfromMA, 200, False)
    h_theta0 = h_(theta0)
    h__      = lambda theta: h_(theta).reshape((len(h_theta0),1))
    Gfcn_    = lambda theta: wcopt.ComputeG(h__,theta)
    res      = wcopt.TestSingle(muhat, Vhat, Nobs, lamb, theta0, theta0, cv, l, u, fullyFeasible, Gfcn_, h__, True, True)
    return res



def MomentMatchRuns(runcode, T, Npd, Nsim, fromMA, hempirical, hfromMA, nparallel=0, seed=314):
    # Load simulated moments for matching
    strSave  = getStrSave(runcode, T, Npd, Nsim, fromMA)
    savepath = "./Results/muhats_" + strSave + ".npy"
    muhats   = np.load(savepath)

    # Some info to start
    inputs, unknowns, targets, outputs, theta0, shock_param, block_list = getruncodeFeatures(runcode)
    K = len(theta0)
    I = np.eye(K)

    # Solve for steady state
    ss, G, inputs, outputs_all, block_list = SolveModel(runcode, theta0, T, [], inputs, outputs, unknowns, targets, block_list, False)

    # Setup up parallel pool
    if nparallel != 0:
        pool = mp.Pool(nparallel)

    # Compute Vhat to use for each, so we won't do fully feasible
    Nobs = 1
    Vhat = Nobs*np.cov(muhats)
    l = np.array([1.001, 0.01, 0.01]).reshape((3,1))
    u = np.array([5, 5, 0.9999]).reshape((3,1))
    res = [[] for k in range(K)]
    for k in range(K):
        print("\tParam %d / %d" % (k+1, K))

        if nparallel == 0:
            for s in range(Nsim):
                print("Matching Simulation %d / %d" % (s+1, Nsim))

                res[k].append(MomentMatch(runcode, theta0, T, ss, Npd, inputs, outputs, unknowns, targets,
                                          block_list, hempirical, hfromMA, Nsim, muhats[:,s], Vhat, Nobs, I[:,k:(k+1)], l, u, False))
        else:
            pool = mp.Pool(2)
            res[k] = pool.starmap(MomentMatch, [(runcode, theta0, T, ss, Npd, inputs, outputs, unknowns, targets,
                                                 block_list, hempirical, hfromMA, Nsim, muhats[:,s], Vhat, Nobs, I[:,k:(k+1)], l, u, False) for s in range(Nsim)])
            pool.close()

    np.save("res_" + strSave + ".npy", res)
    return res





########################################################################
## Features by code ####################################################
########################################################################


def paramsDefaults(runcode):
    if runcode == 0:
        phi0        = 1.5
        theta0      = [phi0]
        shock_param = [False]
    elif (runcode == 1) or (runcode == 2):
        phi0        = 1.5
        kappap0     = 0.1
        rho_Z0      = 0.85
        theta0      = [phi0, kappap0, rho_Z0]
        shock_param = [False, False, False]

    return np.array(theta0), np.array(shock_param)


# For each parameter, establish a range and a default
def paramsCheckID(runcode):
    if runcode == 0:
        phis     = np.linspace(start=1.01, stop=2.0, num=20)
        thetas   = [phis]
    elif runcode == 1:
        phis     = np.linspace(start=1.01, stop=1.9, num=5)
        kappaps  = np.linspace(start=.01, stop=0.2, num=5)
        rho_Zs   = np.linspace(start=0.05, stop=0.95, num=20)

        # Put everything together
        thetas = [phis, kappaps, rho_Zs]

    elif runcode == 2:
        phis     = np.linspace(start=1.01, stop=1.9, num=5)
        kappaps  = np.linspace(start=.01, stop=0.2, num=5)
        rho_Zs   = np.linspace(start=0.05, stop=0.95, num=20)

        # Put everything together
        thetas = [phis, kappaps, rho_Zs]

    theta0, shock_param = paramsDefaults(runcode)

    return np.array(thetas), theta0



def getruncodeFeatures(runcode):
    if runcode == 2:
        exogenous = ['Z', 'rstar', 'G', 'markup', 'markup_w', 'beta', 'rinv_shock']
    else:
        exogenous = ['Z', 'rstar', 'G']

    unknowns  = ['Y', 'r', 'w']
    targets = ['asset_mkt', 'fisher', 'wnkpc']

    outputs = ['Y', 'r', 'pi', 'BetaCZ', 'BetaCA', 'BetaCB']
    # outputs = ['Y', 'C', 'K']

    theta0, shock_param  = paramsDefaults(runcode)

    block_list = [household_inc, pricing, arbitrage, production,
                  dividend, taylor, fiscal, finance, wage, union, mkt_clearing,
                  microBetaCZ, microBetaCA, microBetaCB]

    dt = {'exogenous'   : exogenous,
          'unknowns'    : unknowns,
          'targets'     : targets,
          'outputs'     : outputs,
          'theta0'      : theta0,
          'shock_param' : shock_param,
          'block_list'  : block_list,
          'T'           : 300}

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

    return muhats


########################################################################
## Identification Plots ################################################
########################################################################

# Identification plots
def identificationPlots(runcode, Npd):

    # INITIALIZE
    print("Initializing Model...")
    modelInfo = SolveModel({'runcode' : runcode}, True)

    # Get parameters to loop over
    thetas, theta0 = paramsCheckID(runcode)
    Nparams = len(thetas)

    # Evaluate at initial parameters
    print("Getting moments at initial params")
    hempirical = False
    hsimFromMA = False
    Nsim       = 1000
    Nmoments   = len(h(modelInfo, theta0, Npd, hempirical, hsimFromMA, Nsim, False))

    # Loop over parameters
    labs = GetIdentificationLabels(runcode, list(modelInfo['outputs_all']))
    for p in range(Nparams):
        print("Param %d / %d..." % (p+1, Nparams))

        # Set everything to defaults to start
        theta = copy.deepcopy(theta0)
        # print(theta0)

        # Create array to hold values of moments across different parameter vals
        h_ = np.zeros((Nmoments, len(thetas[p])))

        # Loop over values in identification range
        for i in range(len(thetas[p])):
            # Set that parameter
            theta[p] = thetas[p][i]
            print(theta)

            # Compute h
            h_[:,i] = h(modelInfo, theta,  Npd, hempirical, hsimFromMA, Nsim, False)

        # Plot identification plots by moment
        for m in range(Nmoments):
            # plt.subplot(np.round(np.ceil(Nmoments/4)), 4, m+1)
            plt.plot(h_[m,:])
            plt.title(labs[m])
            plt.show()

        # print(h_)


















































