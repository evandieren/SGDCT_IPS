import numpy as np
import scipy.special as sp
import scipy.stats as st

## VON MISES ##

def f_vm(X,theta,other=None):
    # This is the f for the SDE with the exact von Mises kernel, with \phi = -\nabla W.

    # forced kappa = 0.5
    kappa = 0.5
    def von_mises(X,kappa) :
        return np.exp(kappa*np.cos(X))/(2*np.pi*sp.iv(0, kappa))

    N = X.shape[0]
    P_1 = np.zeros((N,1))
    for j in range(N):
        r = X[j,0]*np.ones((N,1)) - X # X[j,0] is X^j, the jth particle / just take the delta
        P_1[j,0] = (1/N)*np.sum((kappa*np.sin(r))*von_mises(r,kappa))
    return P_1

def von_mises_estimate(x,w) :
    #reconstruct the estimated functions from the Fourier series terms
    J = len(w)
    y = 1/(2*np.pi)
    for j in range (1,J+1) :
        y+=w[j]*np.cos(j*x) # periodic so no need for boundary conditions
    return y

def von_mises(X,kappa) :
    #g^*, true Von Mises distribution
    bessel = sp.iv(0, kappa)
    return np.exp(kappa*np.cos(X))/(2*math.pi*bessel) # no need for boundary cond

## MULTI COS POTENTIAL ##

def multi_cos(x,theta):
    "x is scalar"
    J = len(theta)
    return np.sum(theta*np.cos(x*np.arange(1,J+1)))


def f_multi_cos_potential(X,theta,other=None):
    """
    real function for W = sum_{j=1}^J gamma_j cos(jx), then phi = - nabla W

    X is np array of size (N,1)
    theta is of size (J,1)

    f for multi_cos potential is
    f(x,theta) = 1/N sum_n sum_j j theta_j sin(jx) 
    """
    J = theta.shape[0]
    N = X.shape[0]

    P = np.zeros((J,N),dtype=np.float64)
    for n_ in range(N):
        r = X[n_,0]*np.ones((N,1)) - X
        for j in range(1,J+1) :
            P[j-1,n_] = (1/N)*np.sum(j*np.sin(j*r))
    f_ = np.zeros((N,1))
    for n in range(N):
        f_[n] = np.sum(theta*(P[:,n].reshape(-1,1)))

    return f_

# Onsager kernel

def f_onsager(X,theta,other=None):
    # This is the f for the SDE with the exact Onsager kernel, with \phi = -\nabla W.

    def onsager_deriv(X):
        return np.sign(np.sin(X))*np.cos(X)

    N = X.shape[0]
    P = np.zeros((N,1))
    for j in range(N):
        r = X[j,0]*np.ones((N,1)) - X # X[j,0] is X^j, the jth particle / just take the delta
        P[j,0] = (-1/N)*np.sum(onsager_deriv(r))
    return P

# Opinion dynamics
def f_opinion(X,theta,other=None):
    # This is the f for the SDE with the exact Opinion Dynamics kernel, with \phi = -\nabla W.

    th = [1,3]

    phi = lambda x : -th[0]*x*(np.linalg.norm(x,axis=1)<= th[1]).reshape(-1,1)

    N = X.shape[0]
    P = np.zeros((N,1))
    for j in range(N):
        r = X[j,0]*np.ones((N,1)) - X # X[j,0] is X^j, the jth particle / just take the delta
        P[j,0] = (1/N)*np.sum(phi(r))
    return P

def f_opinion_exp(X,theta,other=None):
    # This is the f for the SDE with the Opinion Dynamics kernel with exp, with \phi = -\nabla W.

    th = [2,0.5]

    phi = lambda x: -th[0]*x*np.exp(-0.01/(1-(x-(th[1]-1))**2))

    N = X.shape[0]
    P = np.zeros((N,1))
    for j in range(N):
        r = X[j,0]*np.ones((N,1)) - X # X[j,0] is X^j, the jth particle / just take the delta
        P[j,0] = (1/N)*np.sum(np.nan_to_num(phi(r)))
    return P

# Curie-Weiss
def f_curie_weiss(X,theta,other=None):
    P = 0.5*(X-np.mean(X))
    return -X-P

# Multi-Hermite
def f_multi_hermite_potential(X,theta,Hermite):
    """
    real function for W = sum_{j=1}^J gamma_j H_j(x), then phi = - nabla W

    X is np array of size (N,1)
    theta is of size (J,1)
    """

    J = theta.shape[0]
    N = X.shape[0]

    P = np.zeros((J,N),dtype=np.float64)
    for n_ in range(N) :
        r = X[n_,0]*np.ones((N,1)) - X
        for j in range(1,J+1) :
            P[j-1,n_] = (-1/N)*np.sum(2*j*Hermite[j-1](r))
    f_ = -X

    for n in range(N):
        f_[n] += np.sum(theta*(P[:,n].reshape(-1,1)))

    return f_

def multi_hermite(x,theta,hermite):
    "x is scalar"
    J = len(theta)
    f = 0
    for j in range(J):
        f += theta[j]*hermite[j](x)
    return f


# Model dictionnary creation

def dict_(model,N,J,w_true=None):
    dict_out = {}
    if model == "von_mises":
        dict_out['f'] = f_vm
        dict_out["drift_flag"] = False
        dict_out['n1'] = N # this can be changed if we need less observations than original model
        dict_out['J'] = J
        dict_out['N'] = N
        dict_out['g'] = lambda x: np.eye(N)
        dict_out['C'] = 10
        dict_out['C_0'] = 100
        dict_out['x0_dist'] = st.uniform(loc=0,scale=2*np.pi)
        dict_out['theta0_proc'] = lambda dict_: st.uniform(loc=0.2,scale=0.4).rvs(dict_['J']).reshape(dict_['J'],1)
        dict_out["target_func"] = lambda x,theta : von_mises_estimate(x,theta)

        kappa = 0.5
        w_true = np.zeros(J)
        for i in range (1,J+1) :
            w_true[i-1] = sp.iv(i, kappa)/(np.pi* sp.iv(0, kappa))
        dict_out['theta_star_dist'] = w_true

    elif model == "potential_multicos":
        dict_out['f'] = f_multi_cos_potential
        dict_out["drift_flag"] = False
        dict_out['n1'] = N
        dict_out['J'] = J
        dict_out['N'] = N
        dict_out['g'] = lambda x: np.eye(N)
        dict_out['C'] = 10
        dict_out['C_0'] = 100
        dict_out['x0_dist'] = st.uniform(loc=0,scale=2*np.pi)
        dict_out['theta0_proc'] = lambda dict_: st.uniform(loc=0,scale=10).rvs(dict_['J']).reshape(dict_['J'],1)
        dict_out["target_func"] = lambda x,theta : multi_cos(x,theta)
        if w_true is not None:
            dict_out['theta_star_dist'] = w_true
        else:
            dict_out['theta_star_dist'] = 1/np.arange(1,J+1)

    elif model == "onsager":
        dict_out['f'] = f_onsager
        dict_out["drift_flag"] = False
        dict_out['n1'] = N
        dict_out['J'] = J
        dict_out['N'] = N
        dict_out['g'] = lambda x: np.eye(N)
        dict_out['C'] = 10
        dict_out['C_0'] = 100
        dict_out['x0_dist'] = st.uniform(loc=0,scale=2*np.pi)
        dict_out['theta0_proc'] = lambda dict_: st.uniform(loc=0,scale=10).rvs(dict_['J']).reshape(dict_['J'],1)
        dict_out["target_func"] = lambda x,theta : f_onsager(x,theta)
        
        if w_true is not None:
            dict_out['theta_star_dist'] = w_true
        else:
            dict_out['theta_star_dist'] = np.zeros(J)
            dict_out['theta_star_dist'][1::2] = -4/(np.pi*(4*np.arange(1,J//2+1)**2-1)) # as we only look for even freqs
    
    elif model == "opinion":
        dict_out['f'] = f_opinion
        dict_out["drift_flag"] = False
        dict_out['n1'] = N
        dict_out['J'] = J
        dict_out['N'] = N
        dict_out['g'] = lambda x: np.eye(N)
        dict_out['C'] = 10
        dict_out['C_0'] = 100
        dict_out['x0_dist'] = st.uniform(loc=0,scale=2*np.pi)
        dict_out['theta0_proc'] = lambda dict_: st.uniform(loc=0,scale=10).rvs(dict_['J']).reshape(dict_['J'],1)

        theta = [1,3]

        if w_true is not None:
            dict_out["theta_star_dist"] = w_true
        else:
            dict_out["theta_star_dist"] = theta[0]/(np.pi*np.arange(1,J+1)**3)*((np.arange(1,J+1)**2*theta[1]**2-2)*\
                                            np.sin(theta[1]*np.arange(1,J+1))\
                                                +2*theta[1]*np.arange(1,J+1)*np.cos(theta[1]*np.arange(1,J+1)))
    elif model == "opinion_exp":
        dict_out['f'] = f_opinion_exp
        dict_out['n1'] = N
        dict_out['J'] = J
        dict_out['N'] = N
        dict_out['g'] = lambda x: np.eye(N)
        dict_out['C'] = 10
        dict_out['C_0'] = 100
        dict_out['x0_dist'] = st.uniform(loc=0,scale=2*np.pi)
        dict_out['theta0_proc'] = lambda dict_: st.uniform(loc=0,scale=10).rvs(dict_['J']).reshape(dict_['J'],1)

        theta = [1,3]

        if w_true is not None:
            dict_out["theta_star_dist"] = w_true
        else:
            dict_out["theta_star_dist"] = theta[0]/(np.pi*np.arange(1,J+1)**3)*((np.arange(1,J+1)**2*theta[1]**2-2)*\
                                            np.sin(theta[1]*np.arange(1,J+1))\
                                                +2*theta[1]*np.arange(1,J+1)*np.cos(theta[1]*np.arange(1,J+1)))
            # this is false but does not really matter atm, we need to recompute the true theta star

    elif model == "curie_weiss":
        dict_out['f'] = f_curie_weiss
        dict_out["drift_flag"] = True
        dict_out["drift"] = lambda x: -x
        dict_out['n1'] = N
        dict_out['J'] = J
        dict_out['hermite'] = [sp.hermite(j, monic=False) for j in range(J)]
        dict_out['N'] = N
        dict_out['g'] = lambda x: np.eye(N)
        dict_out['C'] = 10
        dict_out['C_0'] = 100
        dict_out['x0_dist'] = st.uniform(loc=0,scale=2)
        dict_out['theta0_proc'] = lambda dict_: st.uniform(loc=0.2,scale=0.4).rvs(dict_['J']).reshape(dict_['J'],1)
        kappa = 0.5
        if w_true is not None:
            dict_out["theta_star_dist"] = w_true
        else:
            a = np.polynomial.hermite.poly2herm([0,0,1/4])[1:J+1]
            dict_out['theta_star_dist'] = np.concatenate([a,np.zeros(max(J-len(a),0))])

    elif model == "potential_multihermite":
        dict_out['f'] = f_multi_hermite_potential
        dict_out["drift_flag"] = w_true
        dict_out["drift"] = lambda x : -x
        dict_out['n1'] = N
        dict_out['J'] = J
        dict_out['hermite'] = [sp.hermite(j, monic=False) for j in range(J)]
        dict_out['N'] = N
        dict_out['g'] = lambda x: np.eye(N)
        dict_out['C'] = 10
        dict_out['C_0'] = 100
        dict_out['x0_dist'] = st.uniform(loc=0,scale=5)
        dict_out['theta0_proc'] = lambda dict_: st.uniform(loc=0,scale=10).rvs(dict_['J']).reshape(dict_['J'],1)
        dict_out['theta0_dist'] = st.uniform(loc=0,scale=10)
        if w_true is not None:
            dict_out['theta_star_dist'] = w_true
        else:
            dict_out['theta_star_dist'] = 1/np.arange(1,J+1)

        dict_out["target_func"] = lambda x,theta : multi_hermite(x,theta,[sp.hermite(j+1, monic=False) for j in range(J)])
    else:
        print("model not supported")

    return dict_out
