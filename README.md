## one_plate_mgrf

Python code for solving the modified Gaussian renormalized fluctuation theory to get the electrical double layer structure next to a single uniformly charged plate. This code is based on the equations derived in the work of Agrawal and Wang, Phys. Rev. Lett. 2022, 129, 228001. and J. Chem. Theory Comput. 2022, 18, 6271–6280 and is written on top of open-source spectral methods based differential equation solver \textit{Dedalus}, developed by Burns et al., Phys. Rev. Res. 2020, 2 (2), 023068. The iteration scheme for solving the non-linear equations in this code are soon to be published as a research article. This code solves for a symmteric double layer next to a planar charged surface in contact with an electrolyte solution. Although,the equations derived in the work of Agrawal and Wang account for spatially varying dielectric permittivities, the code is in its current version is for systems with uniform dielectric permittivity. In the text that follows, we have described contents in different files in the package.

# global_constants.py
This file contains three types of variables/parameters which are rarely changed during the course of a research project. The first category is of physical constants which includes variables like temperature, boltzmann constant etc. This is also where you can specify dielectric constants on two side of the charged surface. The next category is of characteristic non-dimensional variables I have used to rescale the various physical quantities in the code to make them more numerically manageable/tractable. The last category is of the numerical parameters like number of grid points or various mixing ratios for iterative procedure. You will have to play around a little bit with these parameters. Explaination of some of the parameters associated with evaluation of greens function are given in https://doi.org/10.1016/j.jcp.2014.07.004.

variables.py
This file contains parameters which are often changed like bulk concentration, valency or size of the ions. This is also where you specify the surface charge. You can also specify the surface charge of the initial guess solution you want to use in (sigma_in_d). In the second part of this file various variable are placed in an array so they can be easily used. Finally the arrays are non dimensionalized using characteristic variables defined in global_constants.py.

calculate.py
This file contains very tiny functions to evaluate things like screening length. This will be a good place to write the formula for 'w' potential for exlculded volume effects.

selfe.py
perhaps the most important file in the package. This file calculate the self energy for the boltzmann factor. The details of self energy calculation can be found in the paper arXiv:2206.02030. Note that although there is an analytical solution for self-energy in the bulk we calculate it numerically so as to cancel out the numerical errors between self energy in interface and in the bulk. There are also functions in this file which calculate grand potential of the double layer This self energy file uses a bunch of other files like greens_function_int.py and greens_function_bulk.py which are described below.

greens_function_int.py and greens_function_bulk.py
files to calculate free space greens function and full greens function in the interface and in the bulk. Although there is an analytical solution for free space greens function we calculate it numerically so as to cancel out the numerical errors.

pb_xumaggs.py
calculates the solution to modified poisson boltzmann equation iteratively using the two step iterative procedure given in https://doi.org/10.1016/j.jcp.2014.07.004. The output of this file/function is the solution you are looking for.

A small but important point to note is that inside this file there is a variable named nodes_exc, please make sure that your choice of variables 'factor' and 'nodes/grid_size' is such that nodes_exc turns out to be an exact integer.

domain variable decides the length of the domain the code will solve for. domain*diameter is the length of system.

num_concn.py
Has two important functions. nconc_xumaggs calculates the coefficient in front of the exp(-z\psi) in the modified poisson boltzmann. It also outputs the coefficients which are needed to calculate the Jacobian for the newton-raphson iterative scheme. nconc_complete is used at the end of pb_xumaggs file to calculate the concentration profile associated with converged potential profile we get out of modified poisson boltzmann.

dh_linear.py
solves the linearized mean-field Poisson-Boltzmann (DH theory for common folk) to be used as initial guess to solve for full mean-field PB in pb_nonlinear.py

pb_nonlinear.py
solves the full mean-field Poisson-Boltzmann to be used as initial guess to solve for pb_xumaggs.py.

simulator_pb.py
This code runs pb_xumagss for the variables given in variables.py to solve for the gaussian renormalized fluctuation theory using pb_xumaggs.py. This file uses the solution of pb_nonlinear.py as the initial guess to solve for pb_xumaggs.py. The input variables for pb_nonlinear.py are same as input variables for pb_xumaggs but one can play with them using variables.py.

simulator.py
This code also runs pb_xumaggs for the variables given in variables.py to solve for the gaussian renormalized fluctuation theory using pb_xumaggs.py. This file uses the solution of a saved solution of pb_xumaggs as the initial guess to solve for pb_xumaggs.py. The variables deciding which saved solution to choose as initial guess are written in variables.py (ex: sigma_in_d).

greens_test.py and selfe_test.py
simple files which you can use to check the validity of code. they compare analytical solutions to numerical solutions.

How to include new excluded volume approach
You will have to replace the functions eta_local and eta_complete with your functions for 'w'. These two functions are called inside:

nconc_complete in num_conc.py
pb_xumaggs.py
simulator.py
Make sure you replace eta everywhere.

About
Gaussian renormalized fluctuation theory for double layers next to a planar charged surface

