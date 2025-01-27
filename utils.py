import numpy as np
import scipy.stats as st
import scipy.special as sp
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import math
import time
import models


## Data generation utils

def gen_data(n_steps, dt, T, dict_, directory, model, periodic=True,other=None):
    '''
    Computes the data for our algorithm, such that
    dX_t = f*dt + g@dW
    where X_t is the (N,1) for the N particles in the system.
    '''

    X = np.zeros((n_steps,int(T/dt)+1,dict_['N'],1))
    dX = np.zeros((n_steps,int(T/dt),dict_['N'],1))
    dWs = st.norm(scale=np.sqrt(dt)).rvs(dict_['N']*int(T/dt)*n_steps).reshape(n_steps,int(T/dt),dict_['N'],1)
    
    theta_star = np.zeros((n_steps,dict_['J'],1))
    theta_0 = np.zeros((n_steps,dict_['J'],1))
    for k in range(n_steps):
        print("n_step=",k)

        if isinstance(dict_['theta_star_dist'],np.ndarray):
            theta_star[k] = dict_['theta_star_dist'].reshape(-1,1)
        else:
            theta_star[k] = dict_['theta_star_dist'].rvs(dict_['J']).reshape(-1,1) # \R^J

        X[k,0] = dict_['x0_dist'].rvs(dict_['N']).reshape(dict_['N'],1)

        # looping over time
        for i in tqdm(range(int(T/dt))):
            X[k,i+1,:] = EM(dict_['f'],dict_['g'],dWs[k,i],dt,theta_star[k],X[k,i,:],other)
            dX[k,i] = X[k,i+1,:]-X[k,i,:]
            if periodic:
                X[k,i+1,:] = X[k,i+1,:]%(2*np.pi)
        theta_0[k] = dict_["theta0_proc"](dict_)
        
    path = Path(f"./{directory}/{model}")

    path.mkdir(parents=True, exist_ok=True)

    np.save(path/f"X_{int(T)}_{dt}_{dict_['N']}_{theta_star[0,0]}.npy",X)
    np.save(path/f"dX_{int(T)}_{dt}_{dict_['N']}_{theta_star[0,0]}.npy",dX)
    np.save(path/f"dW_{int(T)}_{dt}_{dict_['N']}_{theta_star[0,0]}.npy",dWs)
    np.save(path/f"theta_star_{int(T)}_{dt}_{dict_['N']}_{theta_star[0,0]}.npy",theta_star)
    np.save(path/f"theta_0_{int(T)}_{dt}_{dict_['J']}.npy",theta_0)
    print(f"saved everything in {directory}/{model}")

def EM(f,g,dW,dt,theta_star,x_old,other=None):
    '''for X'''
    return x_old + f(x_old,theta_star,other)*dt + g(x_old)@dW

## Error analysis functions

def error_analysis(n_steps,T,dt,theta,theta_star,plot_dir,dict_,model,MLE_flag=False,saving=True,meta=False,sub=False):
    """
    Function to plot the error between theoretical parameters and their respective estimates.

	Args:
		n_steps: int
		T: float
		dt: float
		theta: np.array of dimension (n_steps, T/dt, J, 1)
		theta_star: np.array of dimension (J,1)
    """

    path = Path(f"./{plot_dir}/{model}")

    path.mkdir(parents=True, exist_ok=True)

    theta_hist_mean = np.mean(theta, axis=0) # mean over n_steps
    error_theta_mean = np.mean((theta[:,:,:,0]-theta_star)**2,axis=0) # mean over n_steps

    x = np.arange(0,T+dt,dt)
    if MLE_flag:
        y =10/(0.01*x)
        label_y =  r"$O(t)^{-1}$"
    else:
        y =10/(1+ 0.01*x)
        label_y =  r"$O(1+t)^{-1}$"

    N = dict_["N"]
    
    fig, axs = plt.subplots(figsize=(13, 7))
    colors = ['b', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'k', 'yellow']
    for j in range(dict_['J']):
        plt.plot(x[1:], error_theta_mean[1:,j],color = colors[j%(len(colors))], label = "$w_{%r}$" %(j+1))
    plt.plot(x, y, color = 'k', linestyle = '--',label = label_y)
    plt.title(f"Evolution of the error for the {model.replace('_',' ')} IPS, T={T}, N_steps={n_steps}, dt={dt}, N={N}"+sub*f", # obs={dict_['n1']}"+" (meta=True)"*meta,size=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    axs.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=15)
    plt.ylabel(r'$||w_i-w^{true}_i||$',size = 15)
    plt.xlabel('t',size = 15)
    axs.set_xscale('log')
    axs.set_yscale('log')
    plt.tight_layout()

    if saving:
        plt.savefig(path/f"error_line_{dict_['J']}_{T}_{n_steps}_{N}_{sub*dict_['n1']}_meta_{meta}_MLE_{MLE_flag}.png", bbox_inches='tight')
    else:
        plt.show()

    #plot the trajectories and true values
    fig, axs = plt.subplots(figsize=(13, 7))

    for j in range(dict_['J']):
        plt.plot(x[1:], theta_hist_mean[1:,j],color = colors[j%(len(colors))], label = "$w_{%r}$" %(j+1))
        plt.axhline(y = theta_star[j], color = colors[j%(len(colors))], linestyle = '--')
    axs.set_xscale('log')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    axs.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=15)
    plt.xlabel('t', size = 15)
    plt.ylabel('$w_i$', size = 15)
    plt.tight_layout()
    plt.title(f"Trajectories of $w_i$ for the {model.replace('_',' ')} IPS, T={T}, N_steps={n_steps}, dt={dt}, N={N}"+sub*f", # obs={dict_['n1']}"+" (meta=True)"*meta,size=15)
    if saving:
        plt.savefig(path/f"convergence_weights_{dict_['J']}_{T}_{n_steps}_{N}_{sub*dict_['n1']}_meta_{meta}_MLE_{MLE_flag}.png", bbox_inches='tight')
    else:
        plt.show()

## SGDCT utils

def single_loop(X,dX,T,dt,dict_,sgd,theta_0,meta):
    """
    Single loop over all timesteps (this will be used to run several times for all trajectories)
    We already have the data X
    """
    # allocation for timeseries
    theta = np.zeros((int(T/dt)+1,dict_['J'],1))
    theta[0,:] = theta_0
    # looping over time
    for i in tqdm(range(1,int(T/dt)+1)):
        l = dict_['C']/(dict_['C_0']+i*dt) # might be overwritten if needed
        theta[i,:] = sgd(dict_,l,dt,theta[i-1,:],dX[i-1,:],X[i-1,:],meta) 
    return theta

def sgd_torus(dict_,l,dt,theta_old,dX,x_old,meta=False):
    # lets put theta_old in \R^J for a minute
    J = theta_old.shape[0] #number of particles and number of term in Fourier series
    N = x_old.shape[0]
    l = min(0.1,l)
    
    #compute the gradients : 
    P = np.zeros((J,N),dtype=np.float64) # this is the grad_theta f(x,theta) from the paper SGDCT
    for n_ in range(N):
        r = x_old[n_,0]*np.ones((N,1)) - x_old
        for j in range(1,J+1) :
            P[j-1,n_] = (1/N)*np.sum(j*np.sin(j*r))
    f_ = np.zeros((N,1))
    for n in range(N):
        #print((theta_old*(P[:,n].reshape(-1,1))).shape)
        f_[n] = np.sum(theta_old*(P[:,n].reshape(-1,1)))
    
    return theta_old + l*P@(dX-f_*dt)

def sgd_hermite(dict_,l,dt,theta_old,dX,x_old,meta=False):
    # lets put theta_old in \R^J for a minute
    J = theta_old.shape[0]
    N = x_old.shape[0]
    l = min(0.1,l)
    
    #compute the gradients : 
    P = np.zeros((J,N),dtype=np.float64) # this is the grad_theta f(x,theta) from the paper SGDCT
    for n_ in range(N):
        r = x_old[n_,0]*np.ones((N,1)) - x_old
        for j in range(1,J+1) :
            P[j-1,n_] = (-1/N)*np.sum(2*j*dict_["hermite"][j-1](r))
    f_ = -x_old
    for n in range(N):
        #print((theta_old*(P[:,n].reshape(-1,1))).shape)
        f_[n] += np.sum(theta_old*(P[:,n].reshape(-1,1)))
    return theta_old + l*P@(dX-f_*dt)

# MLE methods

def grad_f(X,J,domain,basis=None):
    """
    This will return a N x J matrix, where each column is \nabla f_j(X_t)
    for j = 1,...,J
    """
    N = X.shape[0]
    P = np.zeros((N,J),dtype=np.float64)
    for n_ in range(N) :
        r = X[n_,0]*np.ones((N,1)) - X
        for j in range(1,J+1):
            if domain == "torus":
                P[n_,j-1] = (-j/N)*np.sum(np.sin(j*r))
            elif domain == "real_line":
                P[n_,j-1] = (2*j/N)*np.sum(basis[j-1](r))
    return P

def construct_F(F,grad,dt,T,sigma):
    # F is the former matrix F at time t-1.
    # Assuming sigma is constant and scalar
    F *= (T-dt)
    F += (grad.T)@(grad)*sigma**2*dt
    F /= T
    return F

def construct_h(h,grad,dX,dt,T):
    # h is the former vector at time t-1.
    h *= (T-dt)
    h -= 2*(grad.T)@dX
    h /= T
    return h

def construct_c(c,grad,sigma,v,dt,T):
    # h is the former vector at time t-1.
    c *= (T-dt)
    c += 2*(grad.T)@v*dt
    c /= T
    return c

def MLE(X,dX,T,dt,dict_,theta_0,domain,basis=None,drift=False):

    if domain not in ["torus","real_line"]:
        print("domain not supported, choose torus or real_line")
        return None
    
    # allocation for timeseries
    theta = np.zeros((int(T/dt)+1,dict_['J'],1))
    theta[0,:] = theta_0
    F = np.zeros((dict_['J'],dict_['J']))
    h = np.zeros((dict_['J'],1))
    c = np.zeros((dict_['J'],1))
    # looping over time 
    for i in tqdm(range(1,int(T/dt)+1)): # This is O(K)
        grad = grad_f(X[i-1],dict_['J'],domain,basis=basis) # this is a N x J matrix (done in O(N^2J))
        T = i*dt
        sigma = 1
        F = construct_F(F,grad,dt,T,sigma) # (done in O(NJ^2))
        h = construct_h(h,grad,dX[i-1],dt,T) # (done in O(NJ))
        if drift:
            v = dict_["drift"](X[i-1])
            c = construct_c(c,grad,sigma,v,dt,T) # (done in O(NJ))
        theta[i,:] = np.dot(np.linalg.inv(F),h+c) # done in O(J^3)
    return 0.5*sigma**2*theta


def main(n_steps,T,dt,dict_,directory,model,self_proc=None,MLE_flag=False,domain=None,meta=False,Xs=None,dXs=None,theta_0=None):
    print("running main utils")
    print("w_true:",dict_['theta_star_dist'].astype(float))
    main_theta = np.zeros((n_steps,int(T/dt)+1,dict_['J'],1))
    
    # Loading data
    path = Path(f"./{directory}/{model}")
    if Xs is None:
        Xs = np.load(path/f"X_{int(T)}_{dt}_{dict_['N']}_{dict_['theta_star_dist'].astype(float)}.npy")
    if dXs is None:
        dXs = np.load(path/f"dX_{int(T)}_{dt}_{dict_['N']}_{dict_['theta_star_dist'].astype(float)}.npy")
    if theta_0 is None:
        theta_0 = np.load(path/f"theta_0_{int(T)}_{dt}_{dict_['N']}_{dict_['J']}.npy")
    print("theta shape",theta_0.shape)

    # Sanity check for number of trajectories
    assert Xs.shape[0] >= n_steps #so that we have enough trajectories for the run
    assert dXs.shape[0] >= n_steps
    assert theta_0.shape[0]>= n_steps
    
    for n in range(n_steps):
        print("n_step",n)

        if MLE_flag:
            main_theta[n,:] = MLE(Xs[n],dXs[n],T,dt,dict_,theta_0[n,0].reshape(-1,1),domain,basis=dict_.get("hermite"),drift=dict_["drift_flag"])
        else:
            main_theta[n,:] = single_loop(Xs[n],dXs[n],T,dt,dict_,self_proc,theta_0[n,0].reshape(-1,1),meta)

        print(main_theta[n,-1])
    return main_theta

























