import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.misc import derivative
from scipy.optimize import minimize, linprog



def qreg(y, x, tau):
#   bhat are the estimates
#   y is a vector of outcomes
#   x is a matrix with columns of explanatory variables
#   tau is a scalar for choosing the conditional quantile to be
#   estimated

    n, m = x.shape

    # vectors and matrices for linprog
    f   = np.vstack( [tau*np.ones((n,1)), (1-tau)*np.ones((n,1)), np.zeros((m,1))] )
    Aeq = np.hstack( [np.eye(n), -np.eye(n), x] )
    beq = y
    lb  = np.vstack( [np.zeros((n,1)), np.zeros((n,1)), -np.inf*np.ones((m,1))] )
    ub  = np.inf * np.ones((m+2*n,1))

    # Solve the linear programme
    res = linprog(f, A_eq=Aeq, b_eq=beq, bounds=list(zip(lb, ub)))

    # Pick out betas from (u,v,beta)-vector.
    # return res.x[end-m+1:end]
    return res.x[-m:]


# Sorted eigenvalues of symmetric/hermitian matrix
def eig_sorted(A):
    D, Q   = np.linalg.eigh(A)
    idx    = D.argsort()[::-1]
    return D[idx], Q[:,idx]


# Compute worst-case optimal weighting matrix
def ComputeWorstCaseOptimal_Single(V, G, lamb_, zero_thresh):

    # Process inputs
    p, k      = G.shape
    sqrtVdiag = np.diag(np.sqrt(np.diag(V)))

    # Ensure lambda the correct shape
    lamb = np.reshape(lamb_, (k,1))

    # Differs from sign in that x=0 => sign(x) = 1
    mysign = lambda x: 1*(x>=0) - 1*(x<0)

    # Eigenvalues, with sorting
    D, Q   = eig_sorted( np.eye(p) - (G @ np.linalg.solve(G.T @ G, G.T)) )
    Gperp  = Q[:,np.abs(D) > zero_thresh]

    # Set up median regression
    Y =  sqrtVdiag @ ( G @ np.linalg.solve(G.T @ G, lamb))
    X = -sqrtVdiag @ Gperp

    #######################################################################
    ## Run median regression
    # mod = sm.regression.quantile_regression.QuantReg(Y, X)
    # res = mod.fit(q=0.5)
    # z   = np.reshape(res.params, (p-k,1))

    ## Run median reg
    z = qreg(Y, X, 0.5)
    z = np.reshape(z, (p-k,1))
    # if not np.allclose(z, z2):
        # raise Exception('Median regs don''t match')


    #######################################################################

    # Construct optimal weight matrix using median regression output,
    # possibly adding in to ensure psd
    W = G @ G.T + (Gperp @ z @ lamb.T @ G.T  +  G @ lamb @ z.T @ Gperp.T) \
                / (lamb.T @ np.linalg.solve(G.T @ G, lamb)).sum()
    #print(W)
    delta = 0.000001;
    while np.min(np.linalg.eigvalsh(W)[0]) < 0:
        #print(np.linalg.eig(W)[0])
        delta = 10*delta
        W =   G @ G.T  + delta*(Gperp @ Gperp.T) \
                       + (Gperp @ z @ lamb.T @ G.T + G @ lamb @ z.T @ Gperp.T) \
                       / (lamb.T @ np.linalg.solve(G.T @ G, lamb)).sum()
    #print(W)

    # Construct
    x    = W @ G @ np.linalg.solve(G.T @ W @ G, lamb)
    s    = mysign(x) * (sqrtVdiag @ np.ones((p,1)))
    Vout = s @ s.T

    # Compute standard errors
    stderr = np.abs(Y - X @ z).sum()

    return W, stderr, Vout, x, z




# Inputs
# ------
# muhat   Vector of estimated reduced-form moments
# V       (Estimate of) cov matrix for reduced-form moments
#
#         This is divided through by sample size to ease calculations
#
# lambda  lambda'theta is what we construct a CI for
# h       Function of theta that gives model-implied moments
# theta0  True values & starting parameter value for optimization
#
def TestSingle(muhat, V, Nobs, lamb, theta0, theta_true, cv, l, u, fullyFeasible, GFcn, h, start_theta0):

    # Define some zero threshold, below which, I'll call something "zero" up
    # to numerical precision. Used to identify "zero" eigenvalues, which
    # numerically aren't identically zero.
    zero_thresh = 1e-8

    ######################################################
    ## General info and setup
    ######################################################

    # Number of parameters and moments
    k = len(theta0)
    p = len(muhat)

    # For the purposes of calculations, make sure all n-dim arrays are
    # (n x 1) arrays
    theta0 = np.reshape(theta0, (k,1))
    muhat  = np.reshape(muhat, (p,1))
    if len(lamb) < len(theta_true):
        lambda_ = np.hstack([lamb,0])
    else:
        lambda_ = lamb
    lambda_ = np.reshape(lambda_, (k,1))

    # Structure to hold results
    res = {}

    # Whether the full V matrix is available, or only the diagonal
    fullV = not np.any(np.isnan(V))

    # Set up estimation function
    estimateFcn = lambda theta0, W: computeCMDEst(muhat, h, theta0, W)


    ######################################################
    ## Compute G at theta0---which are both the true values and the starting
    ## values for optimization
    ######################################################

    G = {}
    G['true'] = GFcn(theta_true)
    if np.linalg.matrix_rank(G['true']) < k:
        print('Warning: G of deficient rank at true parameter')


    ######################################################
    ## Specify V & weighting mats assumed under different approaches, est
    ######################################################

    # Naive
    # - Assumes (incorrectly) that all moments uncorrelated
    # - Then uses "efficient" weighting matrix under that assumption
    res['naive']          = {}
    res['naive']['name']  = 'Naive'
    res['naive']['V']     = np.diag(np.diag(V))
    res['naive']['W']     = np.linalg.inv(res['naive']['V'])
    res['naive']['theta'] = estimateFcn(theta0, res['naive']['W'])
    res['naive']['h']     = h(res['naive']['theta'])
    if res['naive']['h'].shape != (p,1):
        raise Exception('h must have shape (p,1). Check not (p,) or (1,p)')
    if start_theta0:
        theta0_use = theta0
    else:
        theta0_use = res['naive']['theta']

    # G to use in constructing asysmptotic variance eestimators
    if fullyFeasible:
        G['naive'] = GFcn(res['naive']['theta'])
        Guse       = G['naive']
    else:
        Guse = G['true']

    # Construct function for getting onstep estimator
    getOnestep = lambda W: getOnestep_(res['naive']['h'], W, Guse, res['naive']['theta'], muhat)


    # Worst-case optimal
    # - Assumes only diagonal
    # - Then estimates using worst-case optimal weighting matrix
    # - Starts estimation from true value
    res['wcopt']                        = {}
    res['wcopt']['name']                = 'Worst-Case Optimal'
    res['wcopt']['W'], \
        res['wcopt']['stderr_check'], \
        res['wcopt']['V'], \
        res['wcopt']['x_check'], \
        res['wcopt']['z_check']         = ComputeWorstCaseOptimal_Single(V, Guse, lambda_, zero_thresh)
    res['wcopt']['stderr_check']        = res['wcopt']['stderr_check'] / np.sqrt(Nobs)
    res['wcopt']['theta']               = estimateFcn(theta0_use, res['wcopt']['W'])

    # Ensure psd
    #[V_, D_] = eig(res.wcopt.W);
    #res.wcopt.W = V_*max(D,0)*V_';


    # Worst-case optimal, one-step
    # - Asymptotically the same as worst-case optimal
    # - BUT achieves this asymptotic distribution by doing one-step update
    #   of naive estimate (which might provide better point estimates for
    #   computational reasons)
    # - Hence V and W are the same, but estimate constructed differently
    res['wcopt_onestep'] = {}
    res['wcopt_onestep']['name']        = 'Worst-Case Optimal, One-Step'
    res['wcopt_onestep']['V']           = res['wcopt']['V']
    res['wcopt_onestep']['W']           = res['wcopt']['W']
    res['wcopt_onestep']['theta']       = getOnestep(res['wcopt_onestep']['W'])
    res['wcopt_onestep']['lcomb_check'] = np.vdot(lambda_, res['naive']['theta']) - np.vdot(res['wcopt']['x_check'], (res['naive']['h']-muhat))


    # Optimal and Full-information optimal, one-step
    # - Requires the full V matrix is given/available
    # - Then W should be the optimal weighting matrix
    if fullV:
        res['opt']          = {}
        res['opt']['name']  = 'Full-Info Optimal'
        res['opt']['V']     = V
        res['opt']['W']     = np.linalg.inv(V)
        res['opt']['theta'] = estimateFcn(theta0_use, res['opt']['W'])

        res['opt_onestep']          = {}
        res['opt_onestep']['name']  = 'Full-Info Optimal, One-Step'
        res['opt_onestep']['V']     = res['opt']['V']
        res['opt_onestep']['W']     = np.linalg.inv(res['opt']['V'])
        res['opt_onestep']['theta'] = getOnestep(res['opt_onestep']['W'])
    else:
        res['opt']         = []
        res['opt_onestep'] = []



    ######################################################
    ## Given V, W, theta estimates, now compute everything else
    ######################################################

    approaches  = ['naive', 'wcopt', 'wcopt_onestep']
    if fullV:
        approaches = approaches + ['opt', 'opt_onestep']
    Napproaches = len(approaches)
    lcomb_true  = np.vdot(lambda_, theta_true)
    for n, approach in enumerate(approaches):
        # print(approach)

        if fullyFeasible:
            res[approach]['G'] = GFcn(h, res[approach]['theta'])
        else:
            res[approach]['G'] = G['true']

        res[approach]['lcomb']      = np.vdot(lambda_, res[approach]['theta'])
        res[approach]['stderr']     = np.sqrt(ComputeVariance(res[approach]['G'], res[approach]['W'], res[approach]['V'], lambda_.T)/Nobs).sum()
        res[approach]['muhat']      = muhat
        res[approach]['h']          = h(res[approach]['theta'])
        res[approach]['x']          = (lambda_.T @ np.linalg.solve(res[approach]['G'].T @ res[approach]['W'] @ res[approach]['G'], res[approach]['G'].T @ res[approach]['W'])).T
        res[approach]['lcomb_true'] = lcomb_true
        res[approach]['ci']         = res[approach]['lcomb'] +  np.array([-cv, cv])*res[approach]['stderr']
        res[approach]['coverage']   = (lcomb_true > res[approach]['ci'][0]) and (lcomb_true < res[approach]['ci'][1])
        res[approach]['dist']       = 100*np.abs(res[approach]['lcomb'] - lcomb_true) / lcomb_true

    return res




# Construct function for getting onstep estimator
def getOnestep_(h0, W, G, theta0, muhat):
    subtr = np.linalg.solve(G.T @ W @ G,  G.T @ W @ (h0-muhat))
    return np.reshape(theta0[0:len(subtr)] - subtr, (theta0.shape[0],1))


# Compute variance
def ComputeVariance(G,W,V,M):
    return quadForm(V, (M @ np.linalg.inv(G.T @ W @ G) @ G.T @ W).T)


# Finite difference
def FiniteDiff(h, theta, h_theta, v_perturb):

    # Loop over dimensions
    delta            = 0.01    # Size of param perturbation. Will shrink until computed deriv converges
    thresh           = 1e-6    # Threshold for convergence
    converged        = False   # Whether computed derivative has converged
    iters            = 0       # Flag the first run trhough
    maxabspctchg_old = np.inf
    while not converged:

        # Perturb the parameter vector in direction of v_perturb
        theta_fwd = theta + delta*v_perturb
        theta_bwd = theta - delta*v_perturb

        # Compute h under each perturbed theta value
        h_fwd = h(theta_fwd)
        h_bwd = h(theta_bwd)

        # Compute estimate of derivativ
        d_fwd = (h_fwd - h_theta)/delta
        d_bwd = (h_theta - h_bwd)/delta
        d_ctr = (h_fwd - h_bwd)/(2*delta)
        d_new = d_ctr

        # Check for convergence in derivative if not iter=0
        if iters > 0:
            inds         = np.nonzero(d_old)
            abspctchg    = np.abs((d_new[inds] - d_old[inds]) / d_old[inds])
            maxabspctchg = abspctchg.max()
            wrongdir     = (maxabspctchg > maxabspctchg_old)
            converged    = (maxabspctchg < thresh) or wrongdir

            if not converged:
                if np.isnan(maxabspctchg):
                    raise Exception('Error: max abs pct chg = NaN')
                # print('\nNorm = %f\n' % (maxabspctchg))
                # print('\nShrinking to delta = %f\n' % (0.1*delta))
            maxabspctchg_old = maxabspctchg

        # Update delta, G_old, iter in case needed for next iteration
        delta = 0.1*delta
        d_old = d_new
        iters = iters+1

    if wrongdir:
        d = d_old
    else:
        d = d_new

    return d


def ComputeG(h, theta):

    # Preliminaries
    h_theta   = h(theta)
    p         = len(h_theta)
    K         = len(theta)

    # Loop over parameters, and compute G column-by-column
    G = np.zeros((p,K))
    I = np.eye(K)
    for k in range(K):
        G[:,k] = np.squeeze(FiniteDiff(h, theta, h_theta, I[:,k:(k+1)]))

    return G


def quadForm(A, x):
    return x.T @ A @ x


def computeCMDEst(muhat, h, theta0, W):
    objfcn = lambda theta: quadForm(W, muhat-h(theta)).sum()
    res    = minimize(objfcn, theta0, options={'disp' : False}, tol=1e-6)
    return np.reshape(res.x, (theta0.shape[0],1))



















