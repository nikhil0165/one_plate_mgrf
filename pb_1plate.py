import numpy as np
from packages import *
from numerical_param import*
import num_concn
import calculate

def pb_1plate(psi_guess,n_bulk,valency, sigma, domain, epsilon_s):  # psi_guess from Debye Hueckel acts as a initial guess

    grid_points = len(psi_guess)
    bounds = (0,domain)
    Lz = bounds[1]
    coeffs = [n_bulk[i] * valency[i]/epsilon_s for i in range(len(valency))]
    slope = -sigma/epsilon_s

    # Bases
    coords = d3.CartesianCoordinates('z')
    dist = d3.Distributor(coords,dtype=np.float64) # No mesh for serial / automatic parallelization
    zbasis = d3.Chebyshev(coords['z'], size= grid_points,bounds =bounds, dealias=dealias)

    # Fields
    z = dist.local_grids(zbasis)
    psi = dist.Field(name='psi', bases=zbasis)
    tau_1 = dist.Field(name='tau_1')# the basis here is the edge
    tau_2 = dist.Field(name='tau_2')# the basis here is the edge

    # Substitutions
    dz = lambda A: d3.Differentiate(A, coords['z'])
    lift_basis = zbasis.derivative_basis(2)
    lift = lambda A,n : d3.Lift(A, lift_basis, n)

    # lambda function for RHS, dedalus understands lambda functions can differentiate it for newton iteration
    boltz = lambda psi: sum(coeffs[i] * np.exp(-valency[i] * psi) for i in range(len(valency)))

    # PDE setup
    problem = d3.NLBVP([psi,tau_1, tau_2], namespace=locals())
    problem.add_equation("-lap(psi) + lift(tau_1,-1) + lift(tau_2,-2) = boltz(psi)")

    # Boundary conditions
    problem.add_equation("dz(psi)(z=0) = slope")
    problem.add_equation("dz(psi)(z=Lz) = 0")

    # Initial guess
    psi['g'] = psi_guess

    # Solver
    solver = problem.build_solver(ncc_cutoff=ncc_cutoff_pb)
    pert_norm = np.inf

    psi.change_scales(dealias)
    while pert_norm > tolerance_pb:
        solver.newton_iteration()
        pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
        #print('convergence for mean-field PB = ' + str(pert_norm))

    psi.change_scales(1)
    
    n_profile = num_concn.nconc_pb(psi['g'],valency,n_bulk)
    
    z = np.squeeze(z)

    q_profile = calculate.charge_density(n_profile, valency)
    res= calculate.res_1plate(psi['g'],q_profile,bounds,sigma,epsilon_s)
    print("Gauss's law residual for mean-field PB is = " + str(res))

    return psi['g'], n_profile,z,psi(z=0).evaluate()['g']

