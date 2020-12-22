"""
This module implements a scan of the channel
noise parameters for different measurements

M. DiMario (2020)

"""
import time
import numpy as np
from scipy.special import erf
from AllFunctions import calc_mse
import RunExperiments

class RunScan():
    """This class is for running the scan of the noise parameter"""

    def __init__(self):
        self.name = 'run'

    def set_params(self, params, sim_params, noise_params):
        """Set the paramsters for the simulation and noise, calculate error
        probabilities and required filter parameters"""

        self.params = params
        self.mpn = params['mpn']
        self.num_avgs = params['num_avgs']
        self.num_states = params['num_states']

        self.sim_params = sim_params
        self.time_bins = sim_params['time_bins']

        self.noise_params = noise_params
        self.phase_bw = noise_params['phase_bw']
        self.ou_theta = noise_params['ou_theta']
        self.mpn_str = noise_params['mpn_str']
        self.ou_init = noise_params['ou_init']

        RunScan.update_params(self)

    def update_params(self):
        """Error probabilities for the measurements"""
        self.pe_het = 1 - 0.25*(1 + erf(np.sqrt(self.mpn/2)))**2
        if self.mpn == 2.0:
            self.pe_ng = 4.47 * 10**-2  #This is the optimimzed error probability
        elif self.mpn == 5.0:
            self.pe_ng = 1.45 * 10**-3
        elif self.mpn == 10.0:
            self.pe_ng = 7.46 * 10**-6
        else:
            raise Exception('MPN not correct')

        """Numerically obtained parameters for finding the estimator variance
        The fit was to the form var=a[0]*N^-a[1]"""
        if self.mpn == 2.0:
            self.a_ph_bayes = [3.93 * 10**-1, 1.00 * 10**-0]
            self.a_mpn_bayes = [1.15 * 10**1, 9.82 * 10**-1]
            self.a_ph_nn = [5.03 * 10**-3, 2.33 * 10**-2]
            self.a_mpn_nn = [1.04 * 10**-0, 4.89 * 10**-1]
        elif self.mpn == 5.0:
            self.a_ph_bayes = [2.16 * 10**-1, 8.48 * 10**-1]
            self.a_mpn_bayes = [3.03 * 10**1, 9.94 * 10**-1]
            self.a_ph_nn = [8.12 * 10**-3, 3.52 * 10**-1]
            self.a_mpn_nn = [3.04 * 10**-0, 7.32 * 10**-1]
        elif self.mpn == 10.0:
            self.a_ph_bayes = [1.35 * 10**-1, 7.97 * 10**-1]
            self.a_mpn_bayes = [6.27 * 10**1, 9.45 * 10**-1]
            self.a_ph_nn = [6.33 * 10**-3, 3.76 * 10**-1]
            self.a_mpn_nn = [1.68 * 10**-0, 4.97 * 10**-1]
        else:
            raise Exception('MPN not correct')

        """Bayesian estimator variance for the specific num_avgs"""
        self.sp_est_bayes = self.a_ph_bayes[0]/(self.num_avgs**self.a_ph_bayes[1])
        self.sm_est_bayes = self.a_mpn_bayes[0]/(self.num_avgs**self.a_mpn_bayes[1])
        self.est_var_bayes = (self.sp_est_bayes, self.sm_est_bayes)

        """NN estimator variance for the specific num_avgs"""
        self.sp_est_nn = self.a_ph_nn[0]/(self.num_avgs**self.a_ph_nn[1])
        self.sm_est_nn = self.a_mpn_nn[0]/(self.num_avgs**self.a_mpn_nn[1])
        self.est_var_nn = (self.sp_est_nn, self.sm_est_nn)

        """Maximum strength for the OU walks. THese numbers were obtained
        empirically such that when the long-time variance was set to these
        values, none of the walks became negative. This effective bounded all
        the walks between 0 and 2*mpn"""
        if self.mpn == 2.0:
            self.ou_max = 0.25
        elif self.mpn == 5.0:
            self.ou_max = 1.5
        elif self.mpn == 10.0:
            self.ou_max = 6.0
        else:
            raise Exception('MPN not correct')

    def set_scan(self, scan_vec, scan_type):
        """Set the scanned vector and type"""
        self.scan_vec = scan_vec
        self.scan_type = scan_type

    def run_scan(self, which_exps):
        """Run the scan of scan_vec for all measurements and return
        the error probability and mean MSE. This works by instantiating the
        measuements, then scanning the supplied vector and type, and then
        collecting the results
        """

        """Initialize the necessary arrays for data"""
        self.mpe_raw = np.zeros([len(self.scan_vec), self.time_bins])
        self.mpe_np = np.zeros([len(self.scan_vec), self.time_bins])
        self.mpe_nc = np.zeros([len(self.scan_vec), self.time_bins])
        self.mpe_het = np.zeros([len(self.scan_vec), self.time_bins])
        self.mpe_nn = np.zeros([len(self.scan_vec), self.time_bins])
        self.mpe_bayes = np.zeros([len(self.scan_vec), self.time_bins])

        self.mmse_ph_bayes = np.zeros([len(self.scan_vec), ])
        self.mmse_mpn_bayes = np.zeros([len(self.scan_vec), ])
        self.mmse_ph_nn = np.zeros([len(self.scan_vec), ])
        self.mmse_mpn_nn = np.zeros([len(self.scan_vec), ])

        """Instatiate an instance of RunExps"""
        self.experiments = RunExperiments.RunExps(params=self.params,
                                                  sim_params=self.sim_params)

        """Instantiate the measurements"""
        self.experiments.instantiate_measurements()

        tt_start = time.time()
        """Actually RUN the scan"""
        for idx, val in enumerate(self.scan_vec):
            print('Iteration: ', idx, ' Out of: ', len(self.scan_vec))
            print('Scaning: ', self.scan_type, ' -- Scan value :', val)

            """Set the walk parameters depending on the type of scan"""
            if self.scan_type == 'phase_bw':
                noise_params_dict = {'phase_bw' : val,
                                     'mpn_str' : self.mpn_str,
                                     'ou_theta' : self.ou_theta,
                                     'ou_max' : self.ou_max,
                                     'ou_init' : self.ou_init}
            elif self.scan_type == 'mpn_str':
                noise_params_dict = {'phase_bw' : self.phase_bw,
                                     'mpn_str' : val,
                                     'ou_theta' : self.ou_theta,
                                     'ou_max' : self.ou_max,
                                     'ou_init' : self.ou_init}
            elif self.scan_type == 'ou_theta':
                noise_params_dict = {'phase_bw' : self.phase_bw,
                                     'mpn_str' : self.mpn_str,
                                     'ou_theta' : val,
                                     'ou_max' : self.ou_max,
                                     'ou_init' : self.ou_init}
            else:
                raise Exception('Scan type not  correct')

            """Set the experiments with new noise parameters"""
            self.experiments.set_noise_params(noise_params_dict=noise_params_dict)

            """Make new noise waveforms"""
            self.experiments.make_noise_waveforms()

            """Run the experiments with the new noise"""
            self.return_dict = self.experiments.run_exps(self.est_var_bayes,
                                                         self.est_var_nn,
                                                         which_exps)

            """Collect results"""
            self.ph_est_bayes = self.return_dict['ph_est_bayes']
            self.mpn_est_bayes = self.return_dict['mpn_est_bayes']
            self.ph_est_nn = self.return_dict['ph_est_nn']
            self.mpn_est_nn = self.return_dict['mpn_est_nn']

            """Get the true waveforms to compare and calculate the MSE"""
            self.ph_wf = self.experiments.ph_waveform
            self.mpn_wf = self.experiments.mpn_waveform

            """Calculate the MSE at each time bin by extending the walks"""
            self.mse_ph_bayes, _ = calc_mse(self.ph_wf, self.ph_est_bayes)
            self.mse_mpn_bayes, _ = calc_mse(self.mpn_wf, self.mpn_est_bayes)
            self.mse_ph_nn, _ = calc_mse(self.ph_wf, self.ph_est_nn)
            self.mse_mpn_nn, _ = calc_mse(self.mpn_wf, self.mpn_est_nn)

            """Get the total MSE for thw walks"""
            self.mmse_ph_bayes[idx] = np.mean(self.mse_ph_bayes[:])
            self.mmse_mpn_bayes[idx] = np.mean(self.mse_mpn_bayes[:])
            self.mmse_ph_nn[idx] = np.mean(self.mse_ph_nn[:])
            self.mmse_mpn_nn[idx] = np.mean(self.mse_mpn_nn[:])

            """Collet the error probabilities for each measurement"""
            self.mpe_raw[idx, :] = np.mean(self.return_dict['pe_raw'], axis=0)
            self.mpe_np[idx, :] = np.mean(self.return_dict['pe_np'], axis=0)
            self.mpe_nc[idx, :] = np.mean(self.return_dict['pe_nc'], axis=0)
            self.mpe_het[idx, :] = np.mean(self.return_dict['pe_het'], axis=0)
            self.mpe_nn[idx, :] = np.mean(self.return_dict['pe_nn'], axis=0)
            self.mpe_bayes[idx, :] = np.mean(self.return_dict['pe_bayes'], axis=0)

            scan_return_dict = {'mpe_raw' : self.mpe_raw,
                                'mpe_np' : self.mpe_np,
                                'mpe_nc' : self.mpe_nc,
                                'mpe_het' : self.mpe_het,
                                'mpe_nn' : self.mpe_nn,
                                'mpe_bayes' : self.mpe_bayes,
                                'mmse_ph_bayes' : self.mmse_ph_bayes,
                                'mmse_mpn_bayes' : self.mmse_mpn_bayes,
                                'mmse_ph_nn' : self.mmse_ph_nn,
                                'mmse_mpn_nn' : self.mmse_mpn_nn}

        return scan_return_dict
