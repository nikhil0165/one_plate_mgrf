from packages import *
import num_concn
import calculate
import selfe_1plate
import selfe_bulk
from numerical_param import*

def mgrf_1plate(psi_guess,nconc_guess,n_bulk,valency,rad_ions,vol_ions,vol_sol,sigma, domain,epsilon_s,epsilon_p):  # psi_guess from mean-field PB acts as a initial guess

    grid_points = len(psi_guess)
    bounds = (0,domain)
    Lz = bounds[1]
    slope = -sigma/epsilon_s

    psi_g = np.copy(psi_guess)
    uself_guess = selfe_1plate.uself_complete(nconc_guess,n_bulk,rad_ions,valency,domain,epsilon_s,epsilon_p)
    eta_guess=calculate.eta_profile(nconc_guess,vol_ions,vol_sol)

    n_profile = None
    print('selfe_done before the loop')

    # Bulk properties
    n_bulk_numerical = np.multiply(np.ones((grid_points,len(valency))),n_bulk)
    uself_bulk = selfe_bulk.uselfb_numerical(n_bulk_numerical, n_bulk, rad_ions, valency, domain,epsilon_s)[-1]
    eta_bulk = calculate.eta_loc(n_bulk, vol_ions, vol_sol)

    # Checking if all molecules have same excluded volume
    vol_diff = np.abs(vol_ions - vol_sol)
    equal_vols = np.all(vol_diff < vol_sol * 1e-5)
    print(f'equal_vols = {equal_vols}')

    # Solving the matrix
    convergence_tot = np.inf
    p=1
    while(convergence_tot  > tolerance):

        # Bases
        coords = d3.CartesianCoordinates('z')
        dist = d3.Distributor(coords, dtype=np.float64)  # No mesh for serial / automatic parallelization
        zbasis = d3.Chebyshev(coords['z'], size=grid_points, bounds=bounds, dealias=dealias)

        # Fields
        z = dist.local_grids(zbasis)
        psi = dist.Field(name='psi', bases=zbasis)
        tau_1 = dist.Field(name='tau_1')  # the basis here is the edge
        tau_2 = dist.Field(name='tau_2')  # the basis here is the edge

        # Substitutions
        dz = lambda A: d3.Differentiate(A, coords['z'])
        lift_basis = zbasis.derivative_basis(2)
        lift = lambda A, n: d3.Lift(A, lift_basis, n)
        c0 = dist.Field(bases = zbasis)
        c1 = dist.Field(bases = zbasis)

        n_profile_useless, coeffs = num_concn.nconc_mgrf(psi_g,uself_guess,eta_guess,uself_bulk,n_bulk,valency,vol_ions,eta_bulk,equal_vols)
        coeffs = coeffs/epsilon_s

        # lambda function for RHS, dedalus understands lambda functions can differentiate it for newton iteration
        c0['g'] = np.squeeze(coeffs[:, 0])
        c1['g'] = np.squeeze(coeffs[:, 1])
        boltz0 = lambda psi: np.exp(-valency[0] * psi)
        boltz1 = lambda psi: np.exp(-valency[1] * psi)

        if len(valency) == 4:
            c2 = dist.Field(bases = zbasis)
            c3 = dist.Field(bases = zbasis)
            boltz2 = lambda psi: np.exp(-valency[2] * psi)
            boltz3 = lambda psi: np.exp(-valency[3] * psi)

        # PDE setup
        problem = d3.NLBVP([psi, tau_1, tau_2], namespace=locals())
        if len(valency)==2:
            problem.add_equation("-lap(psi) + lift(tau_1,-1) + lift(tau_2,-2) = c0*boltz0(psi) + c1*boltz1(psi)")
        if len(valency)==4:
            c2['g'] = np.squeeze(coeffs[:,2])
            c3['g'] = np.squeeze(coeffs[:,3])
            problem.add_equation("-lap(psi) + lift(tau_1,-1) + lift(tau_2,-2) = c0*boltz0(psi) + c1*boltz1(psi) + c2*boltz2(psi) + c3*boltz3(psi)")

        # Boundary conditions
        problem.add_equation("dz(psi)(z=0) = slope")
        problem.add_equation("dz(psi)(z=Lz) = 0")

        # Initial Guess
        psi['g'] = psi_g

        # Solver
        solver = problem.build_solver(ncc_cutoff=ncc_cutoff_mgrf)
        pert_norm = np.inf
        psi.change_scales(dealias)
        s = 0
        while pert_norm > tolerance_pb:
            solver.newton_iteration()
            pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
            s = s + 1

        psi.change_scales(1)
        psi_g = psi['g']
        #
#        print('PB done')
        if np.any(np.isnan(psi_g)):
            print('nan in psi')

        n_profile,coeff_useless = num_concn.nconc_mgrf(psi_g,uself_guess,eta_guess,uself_bulk,n_bulk,valency,vol_ions,eta_bulk,equal_vols)

        convergence_tot = np.true_divide(np.linalg.norm(n_profile - nconc_guess),np.linalg.norm(nconc_guess))

        nconc_guess = num_ratio*n_profile + (1-num_ratio)*nconc_guess

        uself_guess = selfe_1plate.uself_complete(nconc_guess, n_bulk,rad_ions, valency, domain,epsilon_s,epsilon_p)
        eta_guess = calculate.eta_profile(nconc_guess,vol_ions,vol_sol)

        # deleting dedalus fields as precaution
        del psi,coords,dist,zbasis,z,tau_1,tau_2,dz,lift_basis,lift,problem,solver,pert_norm,c0,c1,boltz0,boltz1

        p = p+1
        if p%10==0:
            print('converg at iter = ' + str(p) + ' is ' + str(convergence_tot))


    #n_profile,uself_profile = num_concn.nconc_complete(psi_g,n_profile,uself_bulk,n_bulk,valency,rad_ions,vol_ions,vol_sol,domain,epsilon_s,epsilon_p)

    uself_profile = selfe_1plate.uself_complete(n_profile,n_bulk,rad_ions,valency,domain,epsilon_s,epsilon_p)
    eta_profile = calculate.eta_profile(n_profile,vol_ions,vol_sol)
    q_profile = calculate.charge_density(n_profile, valency)

    res= calculate.res_1plate(psi_g,q_profile,bounds,sigma,epsilon_s)
    print("Gauss's law residual for MGRF is = " + str(res))

    psi_g,n_profile,uself_profile,Z = calculate.profile_extender(psi_g,n_profile,uself_profile,bounds,np.max(rad_ions),N_exc)

    surface_psi = psi_g[0]

    return psi_g, n_profile,uself_profile,q_profile,Z, surface_psi, res

 
