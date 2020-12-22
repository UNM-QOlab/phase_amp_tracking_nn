"""
This module allows for definition and running of all the experiments

M. DiMario (2020)

"""
from AllFunctions import OU_process, calc_mse
from AllFunctions import state_disc_meas_single_tb, state_disc_meas
from AllFunctions import heterodyne, bayes_estimator_2D
import numpy as np
import tensorflow as tf
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
import contextlib

"""This is for the NN method, so we can watch the parallel operations"""
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback:
        def __init__(self, time, index, parallel):
            self.index = index
            self.parallel = parallel

        def __call__(self, index):
            tqdm_object.update()
            if self.parallel._original_iterator is not None:
                self.parallel.dispatch_next()

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


class RunRef():
    """Class for running the reference measurements with the given parameters and noise"""

    def __init__(self):
        self.name = 'ref'

    def set_params(self, params):
        self.num_avgs = params['navgs']
        self.mpn = params['mpn']
        self.amp = np.sqrt(self.mpn)
        self.detmat_dim = params['dmat_dim']
        self.num_samples = params['nsamp']
        self.time_bins = params['tbins']
        self.time_steps = params['tsteps']
        self.dt = params['dt']
        self.phase_wf = params['ph_wf']
        self.mpn_wf = params['mpn_wf']
        self.noise_params_dict = params['noise_params']
        self.exp_params = params['exp_params']

        self.phase_bw = self.noise_params_dict['phase_bw']
        self.mpn_str = self.noise_params_dict['mpn_str']
        self.ou_theta = self.noise_params_dict['ou_theta']
        self.ou_max = self.noise_params_dict['ou_max']
        self.ou_init = self.noise_params_dict['ou_init']
        self.mpn_noise_var = self.mpn_str*self.ou_max
        self.ou_sigma = self.mpn_noise_var*2*self.ou_theta

    def run_ref(self, indicies):
        """This runs the reference measurements given the supplied random
        walks. Reference measurements are:
            1. NG with perfect correction
            2. NG with perfect phase correction, no MPN correction
            3. NG with no correction
            4. Heterodyne with perfect correction"""

        """initialize arrays"""
        pe_raw = np.zeros([len(indicies), self.time_bins])
        pe_np = np.zeros([len(indicies), self.time_bins])
        pe_nc = np.zeros([len(indicies), self.time_bins])
        pe_het = np.zeros([len(indicies), self.time_bins])

        """Loop over indicies, simulating for the entire length of time. For
        these measurements, instead of running many separate expierments
        simulataneously, the different time bins are used to represent
        different experiments"""
        for i, index in enumerate(indicies):
            """Non-Gaussian receiver with perfect correction"""
            _, _, _, a, b = state_disc_meas(num_avgs=self.time_steps,
                                            input_mpn=self.mpn_wf[index, :],
                                            phase_offset=0,
                                            LO_mpn=self.mpn_wf[index, :],
                                            phase_corr=0,
                                            exp_params=self.exp_params)
            """Non-Gaussian receiver with perfect phase correction, no mpn"""
            _, _, _, c, d = state_disc_meas(num_avgs=self.time_steps,
                                            input_mpn=self.mpn_wf[index, :],
                                            phase_offset=0,
                                            LO_mpn=self.mpn,
                                            phase_corr=0,
                                            exp_params=self.exp_params)
            """Non-Gaussian receiver with no correction"""
            _, _, _, e, f = state_disc_meas(num_avgs=self.time_steps,
                                            input_mpn=self.mpn_wf[index, :],
                                            phase_offset=self.phase_wf[index, :],
                                            LO_mpn=self.mpn,
                                            phase_corr=0,
                                            exp_params=self.exp_params)
            """Heterodyne receiver with perfect correction"""
            _, _, g, h = heterodyne(num_avgs=self.time_steps,
                                    input_mpn=self.mpn_wf[index, :],
                                    phase_offset=0,
                                    phase_corr=0,
                                    exp_params=self.exp_params)

            """run through the results for each measurement to calculate
            the error probability as a function of time in bins of size navg"""
            for jj in range(self.time_bins):
                low = jj*self.num_avgs
                up = (jj+1)*self.num_avgs
                pe_raw[i, jj] = np.mean(a[low:up] != b[low:up])
                pe_np[i, jj] = np.mean(c[low:up] != d[low:up])
                pe_nc[i, jj] = np.mean(e[low:up] != f[low:up])
                pe_het[i, jj] = np.mean(g[low:up] != h[low:up])

            a, b, c, d, e, f, g, h = (0, 0, 0, 0, 0, 0, 0, 0)

        return pe_raw, pe_np, pe_nc, pe_het

class RunBayes():
    """Class for running the bayes based method with the given parameters and noise"""

    def __init__(self):
        self.name = 'bayes'

    def set_params(self, params):
        self.num_avgs = params['navgs']
        self.mpn = params['mpn']
        self.amp = np.sqrt(self.mpn)
        self.grid_len = params['grid_len']
        self.num_samples = params['nsamp']
        self.time_bins = params['tbins']
        self.time_steps = params['tsteps']
        self.dt = params['dt']
        self.est_var = params['est_var']
        self.phase_wf = params['ph_wf']
        self.mpn_wf = params['mpn_wf']
        self.noise_params_dict = params['noise_params']
        self.exp_params = params['exp_params']

        self.phase_bw = self.noise_params_dict['phase_bw']
        self.mpn_str = self.noise_params_dict['mpn_str']
        self.ou_theta = self.noise_params_dict['ou_theta']
        self.ou_max = self.noise_params_dict['ou_max']
        self.ou_init = self.noise_params_dict['ou_init']
        self.mpn_noise_var = self.mpn_str*self.ou_max
        self.ou_sigma = self.mpn_noise_var*2*self.ou_theta

        (sigma_ph_est, sigma_mpn_est) = self.est_var
        self.sigma_ph_est = sigma_ph_est
        self.sigma_mpn_est = sigma_mpn_est

        (DE, VIS, DC, num_states, L, PNR) = self.exp_params
        self.M = num_states
        self.PNR = PNR

    def calc_noise_params(self):
        """Calculate parameters for OU walk filtering"""
        self.a = 1 - self.ou_theta*self.dt
        self.b = self.ou_theta*self.dt
        self.c = self.ou_sigma*self.dt
        self.s, self.s2 = [0, 0]
        for n in range(self.num_avgs):
            self.s += self.a**n
            self.s2 += (self.a**2)**n

        """Calculate phase noise variance for given number of averages"""
        self.sigma_ph_noise = 2*np.pi*self.phase_bw*self.dt*self.num_avgs

    def run_bayes(self, indicies):
        """This runs the experiment with a Bayesian estimator for the supplied
        random walks and parameters"""

        """initialize arrays"""
        pe_bayes = np.zeros([len(indicies), self.time_bins])
        mpn_est = np.zeros([len(indicies), self.time_bins])
        ph_track = np.zeros([len(indicies), self.time_bins])

        prior_2D = np.ones([self.grid_len, self.grid_len])

        """make mesh for 2D Bayes estimator"""
        xx = np.linspace(0, 1.5*self.amp, self.grid_len)
        yy = np.linspace(-0.75*self.amp, 0.75*self.amp, self.grid_len)
        Gx, Gy = np.meshgrid(xx, yy)

        """Initialize LO MPN as either fixed or distributed"""
        for i, index in enumerate(indicies):
            if (self.ou_init == 'mpn'):
                LO_mpn = self.mpn
            elif (self.ou_init == 'dist'):
                LO_mpn = self.mpn_wf[index, 0]

            """Initialize phase correction and filtering variances"""
            phase_corr = 0
            sigma_ph_t = 0
            sigma_mpn_t = 0

            """Loop over the number of time bins applying the phase and MPN
            noise from the random walks and implemeting the Kalman filtering
            of the estimates"""
            for j in range(self.time_bins):
                """Select part of walk to apply"""
                ph_off = self.phase_wf[index, j*self.num_avgs : (j+1)*self.num_avgs]
                in_mpn = self.mpn_wf[index, j*self.num_avgs : (j+1)*self.num_avgs]

                """Run measurement and estimator"""
                meas_results = state_disc_meas(num_avgs=self.num_avgs,
                                                input_mpn=in_mpn,
                                                phase_offset=ph_off,
                                                LO_mpn=LO_mpn,
                                                phase_corr=phase_corr,
                                                exp_params=self.exp_params)
                pe_bayes[i, j], det_matrix, _, _, _ = meas_results

                est_results = bayes_estimator_2D(prior=prior_2D,
                                                  LO_mpn=LO_mpn,
                                                  dmat=det_matrix,
                                                  Gx=Gx,
                                                  Gy=Gy,
                                                  exp_params=self.exp_params)
                est_ph_raw, est_mpn_raw, post = est_results

                """Kalman filtering of the MPN estimate"""
                yhat_mpn = (self.a**self.num_avgs)*LO_mpn + self.b*self.s*self.mpn
                sigma_mpn_t = ((self.a**2)**self.num_avgs)*sigma_mpn_t + self.c*self.s2
                K_mpn = sigma_mpn_t/(sigma_mpn_t + self.sigma_mpn_est)

                mpn_est[i, j] = K_mpn*est_mpn_raw + (1 - K_mpn)*yhat_mpn
                sigma_mpn_t = (1 - K_mpn)*sigma_mpn_t

                """Kalman filtering of the phase estimate"""
                yhat_ph = 0
                sigma_ph_t = sigma_ph_t + self.sigma_ph_noise
                K_ph = sigma_ph_t/(sigma_ph_t + self.sigma_ph_est)

                ph_est = K_ph*est_ph_raw + (1 - K_ph)*yhat_ph
                sigma_ph_t = (1 - K_ph)*sigma_ph_t

                """Add estimate to correction, apply cycle-slip detection"""
                phase_corr += ph_est
                phase_corr += np.round((ph_off[-1] - phase_corr)*2/np.pi)*np.pi/2

                """Save corrrections and current MPN"""
                ph_track[i, j] = phase_corr
                LO_mpn = mpn_est[i, j]

        return pe_bayes, ph_track, mpn_est


class RunNN():
    """Class for running the NN based method with the given parameters and noise"""

    def __init__(self):
        self.name = 'nn'
        self.detmat_dim = 1

    def set_params(self, params):
        self.num_avgs = params['navgs']
        self.mpn = params['mpn']
        self.amp = np.sqrt(self.mpn)
        self.detmat_dim = params['dmat_dim']
        self.num_samples = params['nsamp']
        self.time_bins = params['tbins']
        self.time_steps = params['tsteps']
        self.dt = params['dt']
        self.num_cores = params['ncores']
        self.est_var = params['est_var']
        self.phase_wf_all = params['ph_wf']
        self.mpn_wf_all = params['mpn_wf']
        self.noise_params_dict = params['noise_params']
        self.exp_params = params['exp_params']

        self.phase_bw = self.noise_params_dict['phase_bw']
        self.mpn_str = self.noise_params_dict['mpn_str']
        self.ou_theta = self.noise_params_dict['ou_theta']
        self.ou_max = self.noise_params_dict['ou_max']
        self.ou_init = self.noise_params_dict['ou_init']
        self.mpn_noise_var = self.mpn_str*self.ou_max
        self.ou_sigma = self.mpn_noise_var*2*self.ou_theta

        (sigma_ph_est, sigma_mpn_est) = self.est_var
        self.sigma_ph_est = sigma_ph_est
        self.sigma_mpn_est = sigma_mpn_est

        (DE, VIS, DC, num_states, L, PNR) = self.exp_params
        self.M = num_states
        self.PNR = PNR
        self.detmat_dim = self.M * (self.PNR + 1)

    def run_nn(self):
        """initialize arrays"""
        self.pe = np.zeros([self.num_samples, self.time_bins])
        self.mpn_est = np.zeros([self.num_samples, self.time_bins])
        self.ph_track = np.zeros([self.num_samples, self.time_bins])
        self.LO_mpn = np.ones([self.num_samples, ])*self.mpn
        self.phase_corr = np.zeros([self.num_samples, ])

        """Load trained NN model"""
        tf.reset_default_graph()
        sess = tf.Session()
        saver = tf.train.import_meta_graph('trained_model.meta')
        saver.restore(sess, 'trained_model')
        model = tf.get_default_graph().get_tensor_by_name("nn_output:0")
        x = tf.get_default_graph().get_tensor_by_name('nn_input:0')

        """Calculate parameters for OU walk and phase noise filtering"""
        self.a = 1 - self.ou_theta*self.dt
        self.b = self.ou_theta*self.dt
        self.c = self.ou_sigma*self.dt
        self.s, self.s2 = [0, 0]
        for t in range(self.num_avgs):
            self.s += self.a**t
            self.s2 += (self.a**2)**t

        self.sigma_ph_noise = 2*np.pi*self.phase_bw*self.dt*self.num_avgs
        self.sigma_mpn_t = 0
        self.sigma_ph_t = 0

        """split indicies into chunks"""
        chunks = [np.arange(self.num_samples)[i::self.num_cores] for i in range(self.num_cores)]

        """Run strategy by iterating through each time bin"""
        for tb in tqdm(range(self.time_bins), position=0, leave=True):

            """Get the current bit of the noise walks and set the paramsters
            for the single time bin (stb)"""
            self.phase_wf = self.phase_wf_all[:, tb*self.num_avgs : (tb+1)*self.num_avgs]
            self.mpn_wf = self.mpn_wf_all[:, tb*self.num_avgs : (tb+1)*self.num_avgs]
            self.stb_params = (self.num_avgs,
                               self.detmat_dim,
                               self.phase_wf,
                               self.mpn_wf,
                               self.LO_mpn,
                               self.phase_corr,
                               self.exp_params)

            """Evaluate the measurement in parallel"""
            self.results = []
            self.results = Parallel(n_jobs=self.num_cores, verbose=0)(delayed(state_disc_meas_single_tb)(i, params=self.stb_params) for i in chunks)

            """Collect ehe results"""
            self.det_matrix = np.zeros([self.num_samples, self.detmat_dim])
            for cc, chunk in enumerate(chunks):
                self.pe[chunk, tb] = self.results[cc][0]
                self.det_matrix[chunk, :] = self.results[cc][1]

            """Evaluate the NN with the data from the time bin"""
            self.nn_in = np.zeros([self.num_samples, self.detmat_dim + 1])
            self.nn_in[:, 0:self.detmat_dim] = 2*self.det_matrix - 1
            self.nn_in[:, self.detmat_dim] = self.LO_mpn/10 - 1
            self.nn_out = sess.run(model, feed_dict={x: self.nn_in.reshape(self.num_samples, self.detmat_dim+1)}).reshape(self.num_samples, 2)
            self.est_ph_raw = self.nn_out[:, 0]/2
            self.est_mpn_raw = (self.nn_out[:, 1] + 1)*10

            """Kalman filtering of the MPN estimate"""
            self.yhat_mpn = (self.a**self.num_avgs)*self.LO_mpn + self.b*self.s*self.mpn
            self.sigma_mpn_t = ((self.a**2)**self.num_avgs)*self.sigma_mpn_t + self.c*self.s2
            self.K_mpn = self.sigma_mpn_t/(self.sigma_mpn_t + self.sigma_mpn_est)

            self.mpn_est[:, tb] = self.K_mpn*self.est_mpn_raw + (1 - self.K_mpn)*self.yhat_mpn
            self.sigma_mpn_t = (1 - self.K_mpn)*self.sigma_mpn_t

            """Kalman filtering of the phase estimate"""
            self.yhat_ph = 0
            self.sigma_ph_t = self.sigma_ph_t + self.sigma_ph_noise
            self.K_ph = self.sigma_ph_t/(self.sigma_ph_t + self.sigma_ph_est)

            self.ph_est = self.K_ph*self.est_ph_raw + (1 - self.K_ph)*self.yhat_ph
            self.sigma_ph_t = (1 - self.K_ph)*self.sigma_ph_t

            """Add phase estimate to correction, with cycle-slip correction"""
            self.phase_corr += self.ph_est
            self.phase_corr += np.round((self.phase_wf[:, -1] - self.phase_corr)*2/np.pi)*np.pi/2
            self.ph_track[:, tb] = self.phase_corr

            """Change LO to new input MPN estimate"""
            self.LO_mpn = self.mpn_est[:, tb]

        return self.pe, self.ph_track, self.mpn_est


class RunExps():
    """Class for running all of the experiments depending on the selection.
    Allows for setting the simulation and noise parameters for scanning"""

    def __init__(self, params, sim_params):
        """Experimental parameters"""
        self.mpn = params['mpn']
        self.num_avgs = params['num_avgs']
        self.num_states = params['num_states']
        self.DE = params['DE']
        self.VIS = params['VIS']
        self.DC = params['DC']
        self.PNR = params['PNR']
        self.L = params['L']
        self.exp_params = (self.DE, self.VIS, self.DC, self.num_states, self.L, self.PNR)
        self.dmat_dim = self.num_states*(self.PNR+1)
        self.in_dim = self.dmat_dim + 1
        self.dt = params['dt']

        """Simulation parameters"""
        self.num_samples = sim_params['num_samples']
        self.time_bins = sim_params['time_bins']
        self.grid_len = sim_params['grid_len']
        self.time_steps = int(self.time_bins*self.num_avgs)

        """Set parallel proecssing parameters. The total number of samples is split
        into num_chunks which are split between num_cores parallel workers"""
        self.num_cores = sim_params['num_cores']
        self.num_chunks = sim_params['num_chunks']
        self.chunks = [np.arange(self.num_samples)[i::self.num_chunks] for i in range(self.num_chunks)]

    def set_noise_params(self, noise_params_dict):
        self.noise_params = noise_params_dict
        self.phase_bw = noise_params_dict['phase_bw']
        self.mpn_str = noise_params_dict['mpn_str']
        self.ou_theta = noise_params_dict['ou_theta']
        self.ou_max = noise_params_dict['ou_max']
        self.ou_init = noise_params_dict['ou_init']

        self.mpn_noise_var = self.mpn_str*self.ou_max
        self.ou_sigma = self.mpn_noise_var*2*self.ou_theta

    def make_noise_waveforms(self):
        """Generate the MPN walks through an OU process. The parameters for the walk
        are determined by setting theta and the long-time variance"""

        self.mpn_waveform = OU_process(num_samples=self.num_samples,
                                       time_steps=self.time_steps,
                                       lt_var=self.mpn_noise_var,
                                       theta=self.ou_theta,
                                       dt=self.dt,
                                       mpn_start=self.mpn,
                                       init=self.ou_init)
        self.mpn_waveform[self.mpn_waveform < (0.05*self.mpn)] = (0.05*self.mpn)

        """Generate the random walks in phase for the given bandwidth"""
        self.ph_noise_var = 2*np.pi*self.phase_bw*self.dt
        self.ph_waveform = np.cumsum(np.random.normal(0, np.sqrt(self.ph_noise_var), [self.num_samples, self.time_steps]), axis=1)

    def instantiate_measurements(self):
        self.ref_inst = RunRef()
        self.bayes_inst = RunBayes()
        self.nn_inst = RunNN()

    def run_exps(self, est_var_bayes, est_var_nn, which_exps):
        """This actually runs each experiment in parallel with the noise
        generated by the supplied parameters. If which_exps[X] is False, then
        that measurement will NOT be run."""

        """Run the reference measurements"""
        print('Running reference measurements...')
        self.pe_raw = np.zeros([self.num_samples, self.time_bins])
        self.pe_np = np.zeros([self.num_samples, self.time_bins])
        self.pe_nc = np.zeros([self.num_samples, self.time_bins])
        self.pe_het = np.zeros([self.num_samples, self.time_bins])
        self.ref_params_dict = {'navgs' : self.num_avgs,
                              'mpn' : self.mpn,
                              'dmat_dim' : self.dmat_dim,
                              'nsamp' : self.num_samples,
                              'tbins' : self.time_bins,
                              'tsteps' : self.time_steps,
                              'dt' : self.dt,
                              'ph_wf' : self.ph_waveform,
                              'mpn_wf' : self.mpn_waveform,
                              'noise_params' : self.noise_params,
                              'exp_params' : self.exp_params}

        self.ref_inst.set_params(params=self.ref_params_dict)

        if which_exps['ref'] is True:
            with tqdm_joblib(tqdm(desc='', total=self.num_chunks, leave=True)) as progress_bar:
                results_ref = Parallel(n_jobs=self.num_cores, verbose=0)(delayed(self.ref_inst.run_ref)(i) for i in self.chunks)
            for k, chunk in enumerate(self.chunks):
                self.pe_raw[chunk, :] = results_ref[k][0]
                self.pe_np[chunk, :] = results_ref[k][1]
                self.pe_nc[chunk, :] = results_ref[k][2]
                self.pe_het[chunk, :] = results_ref[k][3]


        """RUN BAYES MEASUREMENT"""
        print('Running Bayes measurements...')
        self.pe_bayes = np.zeros([self.num_samples, self.time_bins])
        self.ph_est_bayes = np.zeros([self.num_samples, self.time_bins])
        self.mpn_est_bayes = np.zeros([self.num_samples, self.time_bins])
        self.bayes_params_dict = {'navgs' : self.num_avgs,
                             'mpn' : self.mpn,
                             'grid_len' : self.grid_len,
                             'nsamp' : self.num_samples,
                             'tbins' : self.time_bins,
                             'tsteps' : self.time_steps,
                             'dt' : self.dt,
                             'est_var' : est_var_bayes,
                             'ph_wf' : self.ph_waveform,
                             'mpn_wf' : self.mpn_waveform,
                             'noise_params' : self.noise_params,
                             'exp_params' : self.exp_params}

        if which_exps['bayes'] is True:
            self.bayes_inst.set_params(params=self.bayes_params_dict)
            self.bayes_inst.calc_noise_params()

            with tqdm_joblib(tqdm(desc='', total=self.num_chunks, leave=True)) as progress_bar:
                results_bayes = Parallel(n_jobs=self.num_cores, verbose=0)(delayed(self.bayes_inst.run_bayes)(i) for i in self.chunks)
            for k, chunk in enumerate(self.chunks):
                self.pe_bayes[chunk, :] = results_bayes[k][0]
                self.ph_est_bayes[chunk, :] = results_bayes[k][1]
                self.mpn_est_bayes[chunk, :] = results_bayes[k][2]


        """RUN NN MEASUREMENT"""
        print('Running NN measurements...')
        self.pe_nn = np.zeros([self.num_samples, self.time_bins])
        self.ph_est_nn = np.zeros([self.num_samples, self.time_bins])
        self.mpn_est_nn = np.zeros([self.num_samples, self.time_bins])
        self.nn_params_dict = {'navgs' : self.num_avgs,
                          'mpn' : self.mpn,
                          'dmat_dim' : self.dmat_dim,
                          'nsamp' : self.num_samples,
                          'tbins' : self.time_bins,
                          'tsteps' : self.time_steps,
                          'dt' : self.dt,
                          'ncores' : self.num_cores,
                          'est_var' : est_var_nn,
                          'ph_wf' : self.ph_waveform,
                          'mpn_wf' : self.mpn_waveform,
                          'noise_params' : self.noise_params,
                          'exp_params' : self.exp_params}

        if which_exps['nn'] is True:
            self.nn_inst.set_params(params=self.nn_params_dict)
            self.pe_nn, self.ph_est_nn, self.mpn_est_nn = self.nn_inst.run_nn()

        return_dict = {'pe_raw' : self.pe_raw,
                       'pe_np' : self.pe_np,
                       'pe_nc' : self.pe_nc,
                       'pe_het' : self.pe_het,
                       'pe_bayes' : self.pe_bayes,
                       'pe_nn' : self.pe_nn,
                       'ph_est_bayes' : self.ph_est_bayes,
                       'mpn_est_bayes' : self.mpn_est_bayes,
                       'ph_est_nn' : self.ph_est_nn,
                       'mpn_est_nn' : self.mpn_est_nn}

        return return_dict

