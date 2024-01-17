# one_plate_mgrf

This is a python package for solving the modified Gaussian renormalized fluctuation theory to get the electrical double layer structure next to a single uniformly charged plate. The code is based on the equations derived in the work of Agrawal and Wang, [Phys. Rev. Lett. 2022, 129, 228001](https://doi.org/10.1103/PhysRevLett.129.228001) and [J. Chem. Theory Comput. 2022, 18, 6271–6280](https://doi.org/10.1021/acs.jctc.2c00607) and is written on top of open-source spectral methods based differential equation solver [Dedalus](https://github.com/DedalusProject/dedalus), developed by [Burns et al., Phys. Rev. Res. 2020, 2 (2), 023068](https://doi.org/10.1103/PhysRevResearch.2.023068). The iteration scheme for solving the non-linear equations in this code are partially adopted from the work of Xu and Maggs[J. Comp. Phys. 275 (2014): 310-322.](https://doi.org/10.1016/j.jcp.2014.07.004), the complete scheme and the method to solve the correlation functions will be soon published as a research article. The code solves for gaussian correlation functions for symmteric double layers in highly parallel manner. Although,the equations derived in the work of Agrawal and Wang account for spatially varying dielectric permittivities, the code is in its current version is for systems with uniform dielectric permittivity. 

This code can be used to reproduce the data presented in:

1. Nikhil R. Agrawal, Chao Duan, and Rui Wang [J. Phys. Chem. B 2024, 128, 1, 303–311](https://doi.org/10.1021/acs.jpcb.3c04739)
2. Nikhil R. Agrawal and Rui Wang [AIChE Journal 2023, 69 (12), e18269](https://doi.org/10.1002/aic.18269)

In the text that follows, the contents of various python files are described.

## numerical_param.py

This is an input file to specify numerical parameters like number of grid points, tolerance criteria, mixing ratios for non-linear solvers, ncc cutoffs for dedalus etc. Note that equations being solved here are highly non-linear and hence some amount of tuning of these numerical parameters is required for efficient calculations.

## physical_param.py 

This is an input file to specify physical environment variables like salt concentrations, ion valencies, born radii of ion, excluded volumes of ions and solvent, domain size, dielectric permittivity, temperature etc. In the second part of this file are derived non dimensional variables from these input parameters, all the calculations are done in these non-dimensional variables. 

## dh_1plate.py

solves the linearized mean-field Poisson-Boltzmann or debye-hueckel theory for electrical double layers. The solution of this is usually used as initial guess to solve for full mean-field PB in pb_1plate.py.

## pb_1plate.py

Solves the full mean-field Poisson-Boltzmann theory for electrical double layers next to a single charged plate. This is a non-linear boundary value problem whose initial guess can come from dh_1plate.py or another solution for mean-field PB. The solution of this can be used as an initial guess to solve for the modified Gaussian renormalized fluctuation theory in mgrf_1plate.py. 

## mgrf_1plate.py

calculates the solution to modified Gaussian renormalized fluctuation theory for 1 plate system. This is also a non-linear boundary value problem whose initial guess can come from pb_1plate.py or solution for mgrf with another set of parameters. This function requires various properties like screening lengths, concentration profiles, self-energies, etc. Functions for these properties are described below.

## num_concn.py
Has threet functions. nconc_mgrf calculates the coefficient in front of the exp(-z\psi) in the mgrf_1plate.py. It also outputs the coefficients which are needed to calculate the Jacobian for the newton-raphson iterative scheme. nconc_complete is the function to calculate concentration profile for given psi profile with n_initial as the initial guess. nconc_pb calculates numer density profiles for mean-field PB. 

## selfe_1plate.py

This file includes functions to calculate the self-energy profiles for a double layer for a 1 plate system based on the equations given in supplemental material of Agrawal and Wang, Phys. Rev. Lett. 2022, 129, 228001. The functions in this file use another file called greens_function_1plate.py which evaluates the fourier transform of the green's functions in the interface.

## selfe_bulk.py

This file includes functions to calculate the self-energy for the bulk solution based on the equations given in supplemental material of Agrawal and Wang, Phys. Rev. Lett. 2022, 129, 228001. Note that although there is an analytical solution for self-energy in the bulk we calculate it numerically so as to cancel out the numerical errors between self energy in interface and in the bulk. The functions in this file use another file called greens_function_bulk.py which evaluates the fourier transform of the green's functions in the bulk.

## greens_function_1plate.py and greens_function_bulk.py

File to calculate fourier transforms of G and Go in the interface and bulk respectively.

## calculate.py

This file contains functions to evaluate properties like screening length, ionic strength, incompressibility fields and charge density profiles. There is also a function called interpolator to interpolate electrostatic potential and ion density profiles in order to increase or decrease grid points. A function to calculate the residual of gauss law is also given.

## energy_1plate.py

Functions to calculate grand free energy of the interface and bulk for both mean-field PB as well as modified Gaussian renormalized fluctuation theory.

## simulator_pb.py

This code saves the solution of the modified Gaussian renormalized fluctuation theory in a .h5 file for the input parameters given in the two *_param.py files. This file uses the solution of pb_1plate.py as the initial guess to solve for mgrf_1plate.py. The input variables for pb_1plate.py and mgrf_2plate can be set separately using the file physical_param.py.

## simulator.py

This code saves the solution of the modified Gaussian renormalized fluctuation theory in a .h5 file for the input parameters given in the two *_param.py files. This file uses a saved solution of mgrf_1plate.py as the initial guess. The physical variables for this saved solution and the final paramters for which we want the double layer structuure can be set separately using the file physical_param.py. The variables deciding which saved solution to choose as initial guess end with "_in_d", for ex: sigma_in_d.

## packages.py

This python file contains the import statements for all the python libraries that are needed for this package. We suggest that you create a separate conda environment where all these libraries are installed.

## Running the code

The code can be run using any of the following commands based on your need: 

python simulator.py physical_param.py

python simulator_pb.py physical_param.py

Note that numerical_param.py has been directly imported in simulator files as the numerical parameters are seldom changed. However one can easily parse them through the command line by changing the first section of simulator files.

## Contact:
This code was developed by Nikhil Agrawal in the lab of Prof. Rui Wang, Pitzer Center for Theoretical Chemistry, University of California, Berkeley, USA. If you need any help feel free to write to Nikhil at nikhilagrawal0165@gmail.com.  

