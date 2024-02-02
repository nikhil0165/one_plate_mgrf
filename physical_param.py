from packages import *

## Global Input Variables, All quantities are in SI unit _d means dimensional

cb1_d = 0.5* pow(10, 3)# prinamry salt bulk concentration
cb2_d = 0*pow(10,3)# secondary salt bulk concentration
valency1 = [3,-1] # valency of primary salt
valency2 = [1,-1] # valency of secondary salt
born_radius = 2.0* pow(10, -10)
rad_sol_d = born_radius
domain_d= 100.0 #  this times debye huckel length
domain_in_d = 100.0 # domain length of the initial guess also in debye huckel length units 
sigma_d = -0.15 # surface charge density
sigma_in_d = -0.15# initial point for starting calculation in case of high surface charge densities

vol_sol_d = 4/3*pi*pow(rad_sol_d,3)


if cb2_d == 0:
    valency = np.array(valency1)
    rad_ions_d = np.array([born_radius, born_radius])
    vol_ions_d = np.array([vol_sol_d,vol_sol_d])
else:
    valency = np.hstack((valency1,valency2))
    rad_ions_d = np.array([born_radius, born_radius,born_radius,born_radius])# rad of ions
    vol_ions_d = np.array([vol_sol_d,vol_sol_d,vol_sol_d,vol_sol_d])


print(f'cb1_d = {cb1_d/pow(10,3)}')
print(f'cb2_d = {cb2_d/pow(10,3)}')
print(f'sigma_in_d = {sigma_in_d}')
print(f'sigma_d = {sigma_d}')
print(f'domain_in_d = {domain_in_d}')
print(f'domain_d = {domain_d}')
print(f'valency = {valency}')


## Physical constants

Temp = 298# Temperature in Kelvin
ec = 1.602 * pow(10, -19)  # electronic charge
k_b = 1.38064852 * pow(10, -23)  # boltzmann's constant
N_A = 6.02214*pow(10,23) # Avogadro number
epsilon_o_d = 8.854187 * pow(10, -12)  # permittivity in vaccuum
beta = 1 / (k_b * Temp)  # beta
epsilonr_s_d = 80  # relative permittivity/dielectric constant
epsilon_s_d = epsilon_o_d * epsilonr_s_d  # permittivity of the medium


## Characteristic variables - dont play with this

l_b = pow(ec,2)*(beta)*(1 / (4 * pi * epsilon_s_d)) #Bjerrum Length
l_c = l_b # characteristic length scale in the system
q_c = ec # characteristic charge
psi_c = (1/(beta*q_c))# characteristic electrostatic potential
epsilon_c = beta*q_c*q_c/l_c # characteristic dielectric permimitivity
sigma_c = (ec/pow(l_c,2)) # characteristic surface charge density
vol_c = pow(l_c,3) # characteristic volume
nconc_c = 1/vol_c # characteristic number density
conc_c = 1/vol_c # characteristic concentration


## Derived variables

# bulk concentration of primary salt ions
cbulk_plus_d, cbulk_neg_d = (cb1_d, cb1_d) if abs(valency[0]) == abs(valency[1]) else (cb1_d * abs(valency[1]), cb1_d * abs(valency[0]))

nbulk_plus_d, nbulk_neg_d = cbulk_plus_d*N_A, cbulk_neg_d*N_A # number of concentration of ions

n_bulk_d =  [nbulk_plus_d,nbulk_neg_d]

# bulk concentration of secondary salt ions
if cb2_d != 0:
    cb2_plus_d, cb2_neg_d = (cb2_d, cb2_d) if abs(valency[2]) == abs(valency[3]) else (cb2_d * abs(valency[3]), cb2_d * abs(valency[2]))
    nb2_plus_d, nb2_neg_d = cb2_plus_d * N_A, cb2_neg_d * N_A  # number of concentration of ions
    n_bulk_d = [nbulk_plus_d,nbulk_neg_d,nb2_plus_d,nb2_neg_d]

I = sum([(valency[i] ** 2) * n_bulk_d[i] / len(valency) for i in range(len(valency))])
lambda_d_d = np.sqrt(epsilon_s_d / (beta * pow(ec, 2) * I)) # Debye Screening length


print('domain in Ang = ' +str(domain_d*lambda_d_d/pow(10,-10)))

## Scaling the variables with characteristic variables

epsilon_s = epsilon_s_d / epsilon_c
n_bulk = np.true_divide(n_bulk_d,nconc_c)
sigma_i = sigma_in_d/sigma_c
sigma = sigma_d/sigma_c
rad_ions = np.true_divide(rad_ions_d,l_c)
rad_sol  = rad_sol_d/l_c
cbulk_plus = cbulk_plus_d/conc_c
cbulk_neg = cbulk_plus_d/conc_c
vol_sol = vol_sol_d/vol_c
vol_ions = np.true_divide(vol_ions_d,vol_c)
domain  = domain_d*lambda_d_d/l_c
lambda_d = lambda_d_d/l_c 




