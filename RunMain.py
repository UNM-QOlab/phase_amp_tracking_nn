"""
This module runs allows for running a scan of the channel
noise parameters for different measurements

M. DiMario (2020)

"""
import numpy as np
import matplotlib.pyplot as plt
from RunScan import RunScan

"""Experimental parameters"""
(mpn_per_meas, num_avgs, num_states) = (5.0, 10, 4)
(DE, VIS, DC, PNR, L) = (1.0, 0.997, 0, 10, 10)
dmat_dim = num_states*(PNR+1)
in_dim = dmat_dim + 1
dt = 1/(100  *10**6)
params = {'mpn' : mpn_per_meas,
          'num_avgs' : num_avgs,
          'num_states' : num_states,
          'DE' : DE,
          'VIS' : VIS,
          'DC' : DC,
          'PNR' : PNR,
          'L' : L,
          'dt' : dt}

"""Simulation parameters"""
num_samples = 5 #Number of random walks
time_bins = 20
grid_len = 10 #Needs to be >=200 for good accuracy it seems
num_cores = 3
num_chunks = 10 #Number of chunks to divide between parallel workers
sim_params = {'num_samples' : num_samples,
              'time_bins' : time_bins,
              'grid_len' : grid_len,
              'num_cores' : num_cores,
              'num_chunks' : num_chunks}

"""Parameters for random noise
For this , we are scanning over the phase noise bandwidth"""
phase_bw = 5 * 10**2
ou_theta = 25 * 10**2
mpn_str = 0.1
ou_init = 'dist' #Initial OU process distribution, either mpn or dist
noise_params = {'phase_bw' : phase_bw,
                'ou_theta' : ou_theta,
                'mpn_str' : mpn_str,
                'ou_init' : ou_init}

"""This sets the scanning vector and scan type for the experiments"""
scan_vec = np.array([0.5, 2, 5, 10, 20, 50])*1000
scan_type = 'phase_bw'

"""Select which experiments to run"""
which_exps = {'ref' : True,
              'bayes' : False,
              'nn' : True}

"""Instantiate the RunScan, set parameters, scan vector and type
and run the parameter scan. RunScan handles setting up the experiments and
actually scanning through the noise values"""
run_scan = RunScan()
run_scan.set_params(params,
                    sim_params,
                    noise_params)
run_scan.set_scan(scan_vec, scan_type)
scan_return_dict = run_scan.run_scan(which_exps)

"""Collect the results"""
mpe_raw = scan_return_dict['mpe_raw']
mpe_nc = scan_return_dict['mpe_nc']
mpe_np = scan_return_dict['mpe_np']
mpe_het = scan_return_dict['mpe_het']
mpe_nn = scan_return_dict['mpe_nn']
mpe_bayes = scan_return_dict['mpe_bayes']
mmse_ph_bayes = scan_return_dict['mmse_ph_bayes']
mmse_mpn_bayes = scan_return_dict['mmse_mpn_bayes']
mmse_ph_nn = scan_return_dict['mmse_ph_nn']
mmse_mpn_nn = scan_return_dict['mmse_mpn_nn']

"""Plot the results"""
fig1 = plt.figure(figsize=[8, 6], dpi=300)
ax1 = fig1.add_subplot(1, 1, 1)
ax1.loglog(scan_vec, np.mean(mpe_bayes, axis=1), 'r')
ax1.loglog(scan_vec, np.mean(mpe_nn, axis=1), 'b')
ax1.loglog(scan_vec, np.mean(mpe_np, axis=1), '--')
ax1.loglog(scan_vec, np.mean(mpe_nc, axis=1), '--')
ax1.loglog(scan_vec, np.mean(mpe_raw, axis=1), '--')
ax1.loglog(scan_vec, np.mean(mpe_het, axis=1), '--')
ax1.legend(('Bayes', 'NN',
            'No ph err', 'No corr', 'Perfect corr', 'Het (no ph err)'))
ax1.set_xlabel(scan_type)
ax1.set_ylabel('Error')
ax1.axhline(y=run_scan.pe_ng, color='k', linestyle='--')
ax1.axhline(y=run_scan.pe_het, color='k', linestyle='--')
ax1.set_title('Error probability')
