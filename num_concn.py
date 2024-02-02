from packages import *
from numerical_param import *
import calculate
import selfe_1plate

# function to calculate concn profiles for mean-field PB
def nconc_pb(psi, valency, n_bulk):
    return n_bulk * np.exp(-np.array(valency) * psi[:,np.newaxis] )

# function to calculate num and coeffs for mgrf_1plate use
def nconc_mgrf(psi,uself,eta_profile,uself_bulk, n_bulk, valency, vol_ions,eta_bulk, equal_vols):
    nodes=len(psi)
    n_profile = np.zeros((nodes,len(valency)))
    coeffs = np.zeros((nodes,len(valency)))
    if equal_vols:
        A = n_bulk* np.exp(-np.array(valency) * psi[:,np.newaxis] - (uself - uself_bulk) + vol_ions * eta_bulk)
        coeffs = valency * n_bulk* np.exp(-(uself - uself_bulk) + vol_ions * eta_bulk)
        denom = 1 + np.sum(A * vol_ions, axis=1)
        n_profile= np.true_divide(A,denom[:,np.newaxis])
        coeffs = np.true_divide(coeffs,denom[:,np.newaxis])
    else:
        n_profile = n_bulk * np.exp(-np.array(valency)*psi[:,np.newaxis] - (uself - uself_bulk) - vol_ions * (eta_profile[:,np.newaxis] - eta_bulk))
        coeffs = valency* n_bulk * np.exp(-(uself - uself_bulk) - vol_ions* (eta_profile[:,np.newaxis] - eta_bulk))
    return n_profile,coeffs

# function to calculate concentration profile for given psi profile, n_initial is the initial guess
def nconc_complete(psi, n_initial,uself_bulk,n_bulk, valency, rad_ions, vol_ions, vol_sol, domain, epsilon):

    eta_bulk = calculate.eta_loc(n_bulk, vol_ions, vol_sol)
    eta_profile = calculate.eta_profile(n_initial,vol_ions,vol_sol)
    nodes = len(psi)

    # profile variables
    n_profile = np.copy(n_initial)
    n_guess = np.copy(n_initial)

    # Checking if all molecules have same excluded volume
    vol_diff = np.abs(vol_ions - vol_sol)
    equal_vols = np.all(vol_diff < vol_sol * 1e-5)

    # initializing the self energy 
    uself_profile = selfe_1plate.uself_complete(n_guess, n_bulk,rad_ions, valency,domain, epsilon)

    # Iteration
    convergence = np.inf
    p = 0
    while (convergence > tolerance_num) and (p < iter_max):
        p = p + 1
        if equal_vols:
            A = n_bulk* np.exp(-np.array(valency) * psi[:, np.newaxis] - (uself_profile - uself_bulk) + vol_ions * eta_bulk)
            denom = 1 + np.sum(A * vol_ions, axis=1)
            n_profile = np.true_divide(A, denom[:,np.newaxis])
        else:
            n_profile = n_bulk * np.exp(-np.array(valency)*psi[:,np.newaxis] - (uself_profile - uself_bulk) - vol_ions * (eta_profile[:,np.newaxis] - eta_bulk))
        convergence = np.true_divide(np.linalg.norm(n_profile - n_guess),np.linalg.norm(n_guess))
        n_guess = (num_ratio) * n_profile + (1-num_ratio) * n_guess
        uself_profile = selfe_1plate.uself_complete(n_guess,n_bulk, rad_ions, valency, domain,epsilon)
        eta_profile = calculate.eta_profile(n_guess,vol_ions,vol_sol)
        if p%10==0:
            print('num='+str(convergence))
        if p >= iter_max:
            print("too many iterations for convergence")

    return n_profile, uself_profile