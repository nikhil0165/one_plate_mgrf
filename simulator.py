from packages import *
import mgrf_1plate
from numerical_param import *
import energy_1plate
import calculate
from physical_param import *

# Argument parser to accept the input files
parser = argparse.ArgumentParser(description='Code to calculate EDL structure using MGRF Theory with mean-field PB as an initial guess')
parser.add_argument('input_files',nargs='+',help='Paths to the input files for physical parameters')
args = parser.parse_args()

folder_path = os.path.dirname(args.input_files[0])
sys.path.insert(0,folder_path)

# Load the physical input configuration from the first file in the list
module_name = os.path.splitext(os.path.basename(args.input_files[0]))[0]
input_physical = importlib.import_module(module_name)
variables = {name: value for name,value in input_physical.__dict__.items() if not name.startswith('__')}
(locals().update(variables))

print(f'ncc_cutoff_mgrf = {ncc_cutoff_mgrf}')
print(f'ncc_cutoff_greens= {ncc_cutoff_greens}')
print(f'num_ratio = {num_ratio}')
print(f'tolerance = {tolerance}')
print(f'N_grid = {N_grid}')

if cb2_d != 0:
    file_dir = os.getcwd() + '/results-mixture' + str(abs(valency[0]))+ '_' + str(abs(valency[1])) + '_' + str(abs(valency[2]))+ '_' + str(abs(valency[3]))
    file_name =  str(round(cb1_d,9)) + '_' + str(round(cb2_d,5)) + '_' + str(round(float(domain_in_d),2))+ '_' + str(round(rad_ions_d[0],2)) + '_' + str(round(rad_ions_d[1],2)) + '_' + str(round(rad_ions_d[2],2)) + '_' + str(round(rad_ions_d[3],2)) + '_' + str(round(sigma_in_d,5))  + '_' + str(round(epsilonr_s_d,5)) + '_' + str(round(epsilonr_p_d,5)) + '_' + str(int(N_grid))
else:
    file_dir = os.getcwd() + '/results' + str(abs(valency[0])) + '_' + str(abs(valency[1]))
    file_name = str(round(cb1_d,9)) + '_' + str(round(cb2_d,5)) + '_' + str(round(float(domain_in_d),2)) + '_' + str(round(rad_ions_d[0],2)) + '_' + str(round(rad_ions_d[1],2)) + '_' + str(round(sigma_in_d,5))  + '_' + str(round(epsilonr_s_d,5)) + '_' + str(round(epsilonr_p_d,5)) + '_' + str(int(N_grid))

with h5py.File(file_dir + '/mgrf_' + file_name + '.h5','r') as file:
    # Retrieve psi and nconc
    psi_profile = np.array(file['psi'])
    n_profile = np.array(file['nconc'])
    N_exc = file.attrs.get('N_exc')
    grandfe = file.attrs.get('grandfe')

start = timeit.default_timer()

psi_profile, n_profile, N_exc = calculate.interpolator(psi_profile[N_exc:],n_profile[N_exc:],(0,domain),N_grid)
print(len(psi_profile))
print(len(n_profile))

psi_profile,n_profile,uself_profile,q_profile,z,surface_psi,res= mgrf_1plate.mgrf_1plate(psi_profile[N_exc:],n_profile[N_exc:],n_bulk,valency,rad_ions,vol_ions,vol_sol,sigma,domain,epsilon_s, epsilon_p)
print('MGRF_done')
print(f'surface_psi = {surface_psi}')

N_exc = np.nonzero(n_profile[:,0])[0][0]

time =timeit.default_timer() - start
print(f'time = {time}')

grandfe = 0#energy_1plate.grandfe_mgrf_1plate(psi_profile,n_profile,uself_profile,n_bulk,valency,rad_ions,vol_ions,vol_sol,sigma,domain,epsilon_s, epsilon_p)
print(f'grandfe = {grandfe}')


if cb2_d != 0:
    file_name =  str(round(cb1_d,9)) + '_' + str(round(cb2_d,5)) + '_' + str(round(float(domain_d),2)) + '_' + str(round(rad_ions_d[0],2)) + '_' + str(round(rad_ions_d[1],2)) + '_' + str(round(rad_ions_d[2],2)) + '_' + str(round(rad_ions_d[3],2)) + '_' + str(round(sigma_d,5))+ '_' + str(round(epsilonr_s_d,5)) + '_' + str(round(epsilonr_p_d,5)) + '_' + str(int(len(psi_profile) - N_exc))
else:
    file_name = str(round(cb1_d,9)) + '_' + str(round(cb2_d,5))  + '_' + str(round(float(domain_d),2)) + '_' + str(round(rad_ions_d[0],2)) + '_' + str(round(rad_ions_d[1],2)) + '_' + str(round(sigma_d,5)) + '_' + str(round(epsilonr_s_d,5)) + '_' + str(round(epsilonr_p_d,5)) + '_' + str(int(len(psi_profile) - N_exc))


# Writing everything in SI units
with h5py.File(file_dir + '/mgrf_' + file_name + '.h5','w') as file:

    # Storing scalar variables as attributes of the root group
    file.attrs['ec_charge'] = ec
    file.attrs['char_length'] = l_b
    file.attrs['beta'] = beta
    file.attrs['epsilon_s'] = epsilonr_s_d
    file.attrs['epsilon_p'] = epsilonr_p_d
    file.attrs['cb1'] = cb1_d
    file.attrs['cb2'] = cb2_d
    file.attrs['surface_charge'] = sigma_d
    file.attrs['domain'] = domain_d

    # Storing numerical parameters as attributes of the root group
    file.attrs['s_conv'] = s_conv
    file.attrs['N_grid'] = len(psi_profile) - N_exc
    file.attrs['N_exc'] = N_exc
    file.attrs['quads'] = quads
    file.attrs['grandfe_quads'] = grandfe_quads
    file.attrs['dealias'] = dealias
    file.attrs['ncc_cutoff_pb'] = ncc_cutoff_pb
    file.attrs['ncc_cutoff_mgrf'] = ncc_cutoff_mgrf
    file.attrs['num_ratio'] = num_ratio
    file.attrs['selfe_ratio'] = selfe_ratio
    file.attrs['eta_ratio'] = eta_ratio
    file.attrs['tolerance'] = tolerance
    file.attrs['tolerance_pb'] = tolerance_pb
    file.attrs['tolerance_num'] = tolerance_num
    file.attrs['tolerance_greens'] = tolerance_greens


    # Storing parameter arrays
    file.create_dataset('valency',data = valency)
    file.create_dataset('radii',data = rad_ions_d)
    file.create_dataset('volumes',data = np.concatenate((vol_ions_d,[vol_sol_d])))

    # Store all spatial profiles  (SI units)
    file.create_dataset('z_d',data = z*l_c)
    file.create_dataset('psi_d',data = psi_profile*psi_c)
    file.create_dataset('nconc_d',data = n_profile*nconc_c/N_A)
    file.create_dataset('uself_d',data = uself_profile*(1/beta))
    file.create_dataset('charge_d',data = q_profile*(nconc_c*ec))
    file.create_dataset('n_bulk_d',data = n_bulk_d)

    # Store all spatial profiles (non-dimensional)
    file.create_dataset('z',data = z)
    file.create_dataset('psi',data = psi_profile)
    file.create_dataset('nconc',data = n_profile)
    file.create_dataset('uself',data = uself_profile)
    file.create_dataset('charge',data = q_profile)
    file.create_dataset('n_bulk',data =n_bulk)

    # Store scalar variables
    file.attrs['grandfe'] = grandfe # nondimensional
    file.attrs['grandfe_d'] = grandfe*(1/beta) # SI units
    file.attrs['residual'] = res
    file.attrs['surface_psi'] = surface_psi
    file.attrs['time'] = time


