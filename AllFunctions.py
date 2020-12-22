"""
This module contains all the functions needed to run the experiments and scans

M. DiMario (2020)

"""
import numpy as np
import math
import scipy.special


def OU_process(num_samples, time_steps, lt_var, theta, dt, mpn_start, init):
    """Generate an OU process given the stochatic differential EQ
    if init==mpn, starting point for each wach is given by mpn
    if init==dist, starting point for each walk is chosen randomly"""
    sigma = np.sqrt(lt_var*2*theta)
    dW = np.random.normal(0, 1, size=[num_samples, time_steps])
    w = np.zeros([num_samples, time_steps])
    if (init == 'mpn'):
        w[:, 0] = mpn_start
    elif (init == 'dist'):
        w[:, 0] = np.random.normal(mpn_start, scale=np.sqrt(lt_var), size=[num_samples,])
    for t in range(time_steps - 1):
        w[:, t+1] = w[:, t] - theta*(w[:, t] - mpn_start)*dt + sigma*np.sqrt(dt)*dW[:, t]

    return w


def mean_ph_num(alpha, beta, exp_params):
    """Calculate the mean photon number from alpha and beta
    both are complex numbers."""
    (DE, VIS, DC, num_states, L, PNR) = exp_params
    aa = np.abs(alpha)
    ap = np.angle(alpha)
    ba = np.abs(beta)
    bp = np.angle(beta)

#    output = np.abs(alpha - beta)**2
    output = DE*(aa**2 + ba**2 - 2*VIS*aa*ba*np.cos(ap - bp)) + DC
    return output


def likelihood(det, alpha, beta, exp_params):
    """Calculate the likelihood function for detecting nn photons.
    Given by:  L(n|states)=exp(-mpn)*mpn**n / n!
    Calculates for both PNR and non_PNR simultaneously such that it can
    select which one to output. THis allows for calculating ALL likelihood
    functions for each MC sample simultaneously vs. 1 by 1"""
    (DE, VIS, DC, num_states, L, PNR) = exp_params

    mpn = mean_ph_num(alpha, beta, exp_params)
    temp = 0
    for det_ind in range(PNR):
        fac = math.factorial(det_ind)
        temp += (mpn**det_ind)*np.exp(-mpn)/fac
    output_pnr = np.abs(1 - temp)

    fac = scipy.special.factorial(det)
    output_non_pnr = (mpn**det)*np.exp(-mpn)/fac

    output = output_non_pnr*(det < PNR) + output_pnr*(det >= PNR)

    return output


def likelihood_single(det, alpha, beta, exp_params):
    """Calculate the likelihood function for detecting n photons.
    This is for doing it one by one, as opposed to above"""
    (DE, VIS, DC, num_states, L, PNR) = exp_params

    mpn = mean_ph_num(alpha, beta, exp_params)
    if det >= PNR:
        temp = 0
        for det_ind in range(PNR):
            fac = math.factorial(det_ind)
            temp += (mpn**det_ind)*np.exp(-mpn)/fac
        output = np.abs(1 - temp)
    else:
        fac = math.factorial(det)
        output = (mpn**det)*np.exp(-mpn)/fac
    return output


def heterodyne(num_avgs, input_mpn, phase_offset, phase_corr, exp_params):
    """This simulates a heterodyne measurement by MC. Data is drawn from
    Gaussian distributions with means Re[a]/sqrt(2), Im[a]/sqrt(2)"""
    (DE, VIS, DC, num_states, L, PNR) = exp_params

    """Generate the actual input states with the input mpn and phase offset"""
    a = np.sqrt(input_mpn)
    state_act_ind = np.random.randint(0, num_states, size=[num_avgs, ])
    sap = state_act_ind*2*np.pi/num_states + phase_offset
    state_act = a*np.exp(1j*sap)

    """Decompose input states into real and imag parts, each with 1/2 power"""
    x = np.real(state_act/np.sqrt(2))
    y = np.imag(state_act/np.sqrt(2))

    """Monte carlo sampling data from Gaussian distribution"""
    data_x = np.random.normal(x, scale=0.5)
    data_y = np.random.normal(y, scale=0.5)
    data_c = data_x + 1j*data_y
    data = np.mod(np.angle(data_c) + np.pi/num_states + phase_offset, 2*np.pi)

    """Monte carlo sampling data from Gaussian distribution with efficiency"""
    data_DE_x = np.random.normal(np.sqrt(DE)*x, scale=0.5)
    data_DE_y = np.random.normal(np.sqrt(DE)*y, scale=0.5)
    data_DE_c = data_DE_x + 1j*data_DE_y
    data_DE = np.mod(np.angle(data_DE_c) + np.pi/num_states + phase_offset, 2*np.pi)

    """Make guess for input state based on measurement outcome"""
    hyp = np.floor(data*num_states/(2*np.pi))
    hyp_DE = np.floor(data_DE*num_states/(2*np.pi))

    """Get error probability"""
    prob_error = np.sum(hyp != state_act_ind)/num_avgs
    prob_error_DE = np.sum(hyp_DE != state_act_ind)/num_avgs

    return prob_error, prob_error_DE, hyp, state_act_ind


def state_disc_meas(num_avgs, input_mpn, phase_offset, LO_mpn, phase_corr, exp_params):
    """This simulates the adaptive photon counting measurement when applying
    the noise and corrections to the states and LO."""
    (DE, VIS, DC, num_states, L, PNR) = exp_params

    a = np.sqrt(input_mpn/L)
    b = np.sqrt(LO_mpn/L)

    prior = np.ones([num_avgs, num_states])
    hyp = np.zeros([num_avgs, L+1], dtype=int)
    hyp_init = np.zeros([num_avgs, ])
    data_mat = np.zeros([num_avgs, L], dtype=int)

    """These are the actual input states with the phase offset and power"""
    state_act_ind = np.random.randint(0, num_states, size=[num_avgs, ])
    sap = state_act_ind*2*np.pi/num_states + phase_offset
    state_act = a*np.exp(1j*sap)

    """These are the states the receiver THINKS they are receiving, i.e. with
    the phase correction added. They are constructed this way to be able
    to implement the perfect correction measurement."""
    sv = np.linspace(0, num_states-1, num_states)
    svp = sv*2*np.pi/num_states + phase_corr
    zz = np.tile(np.exp(1j*svp), (num_avgs, 1))
    states_mat = np.zeros([num_avgs, num_states], dtype=complex)
    for s in range(num_states):
        states_mat[:, s] = b*zz[:, s]

    """Initial LO is the state with a nominal phase of 0"""
    hyp[:, 0] = hyp_init
    LOp = hyp[:, 0]*2*np.pi/num_states + phase_corr
    LO = b*np.exp(1j*LOp)

    for i in range(L):
        """Data sampled from poisson distribution"""
        mpn = mean_ph_num(state_act, LO, exp_params)
        data = np.random.poisson(mpn, size=(num_avgs,))
        data[data > PNR] = PNR
        data_mat[:, i] = data

        """Bayesian updating step, Post=Likelihood*Prior/N"""
        data_m = np.tile(data, (num_states, 1)).T
        LO_m = np.tile(LO, (num_states, 1)).T
        post = prior*likelihood(data_m, states_mat, LO_m, exp_params)
        post = post/np.sum(post, axis=1)[:, np.newaxis]

        """Update hypothesis, LO, and prior"""
        hyp[:, i+1] = np.argmax(post, axis=1)
        LOp = hyp[:, i+1]*2*np.pi/num_states + phase_corr
        LO = b*np.exp(1j*LOp)

        """For next adaptive step, the prior is the previous posterior"""
        prior = post

    final_hyp = hyp[:, L]
    final_hyp_mat = np.tile(final_hyp, (L, 1)).T
    act_mat = np.tile(state_act_ind, (L, 1)).T

    prob_error = np.sum(final_hyp != state_act_ind)/num_avgs

    """Calcualte the detection matrix"""
    det_mat = np.zeros([num_states, PNR+1])
    det_mat_act = np.zeros([num_states, PNR+1])
    z = np.mod(hyp[:, 0:L] - final_hyp_mat, num_states)
    # z_act = np.mod(hyp[:, 0:L] - act_mat, num_states)

    """Bin the detections for each relative phase for D_ij
    det_mat uses the actual hypothesis for the input state. i.e. outcome of measurement
    det_mat_act uses the actual input state (but we don't use this one)"""
    for j in range(L):
        i1 = z[:, j]
        # i1_act = z_act[:, j]
        i2 = data_mat[:, j]
        for k in range(num_states):
            for l in range(PNR+1):
                det_mat[k, l] += np.sum((i1 == k) & (i2 == l))
                # det_mat_act[k, l] += np.sum((i1_act == k) & (i2 == l))

    # det_mat_norm = det_mat/np.sum(det_mat + 1e-100, axis=1)[:, np.newaxis]
    # det_mat_act_norm = det_mat/np.sum(det_mat_act + 1e-100, axis=1)[:, np.newaxis]

    return prob_error, det_mat, det_mat_act, final_hyp, state_act_ind


def state_disc_meas_single_tb(indicies, params):
    """This is to slice up and run a bunch of state discimination measurements
    in parallel for the NN method. Each runs N_avg measurements. The returned
    value of dmat is the detection matrix"""
    (navg, dmdim, ph_off, in_mpn, LO_mpn, ph_corr, exp_params) = params

    pe = np.zeros([len(indicies), ])
    dmat_flat = np.zeros([len(indicies), dmdim])

    for ii, index in enumerate(indicies):
        pe[ii], dmat, _, _, _ = state_disc_meas(navg,
                                                in_mpn[index, :],
                                                ph_off[index, :],
                                                LO_mpn[index],
                                                ph_corr[index],
                                                exp_params)
        dmat_norm = dmat/np.sum(dmat + 1e-100, axis=1)[:, np.newaxis]
        dmat_flat[ii, :] = np.ndarray.flatten(dmat_norm)

    return pe, dmat_flat


def bayes_estimator_2D(prior, LO_mpn, dmat, Gx, Gy, exp_params):
    """Implement a Bayesian estimator in 2 dimensions"""
    (DE, VIS, DC, num_states, L, PNR) = exp_params

    lhood_total = np.ones(prior.shape) #Likelihood
    states_c = Gx + 1j*Gy

    """Construct entire likelihood function for each element in dmat"""
    for k in range(num_states):
        for l in range(PNR+1):
            bp = k*2*np.pi/num_states
            b = np.sqrt(LO_mpn/L)*np.exp(1j*bp)
            if (dmat[k, l] != 0):
                lhood = likelihood_single(l, states_c, b, exp_params)
                for m in range(int(dmat[k, l])):
                    lhood_total = lhood_total*lhood
                    lhood_total = lhood_total/np.max(lhood_total[:])

    """Calculate normalized posterior distribution"""
    post = prior*lhood_total
    post = post/np.sum(post[:])

    """Calculate amplitude and phase estimates"""
    b = np.sum(np.sum(states_c*post))
    phase_est = np.angle(b)
    mpn_est = L*(np.abs(b)**2)  #Multiply by number of adaptive steps

    return phase_est, mpn_est, post


def calc_mse(wf_act, wf_est):
    """Calculates the MSE between the estimated and true waveform after
    stretching the estimated waveform such that it is piecewise-constant."""
    num_est = wf_est.shape[0]
    tbins = wf_est.shape[1]
    tsteps = wf_act.shape[1]
    r = int(tsteps/tbins)

    """Stretch the estimated wf to match the size of the actual wf"""
    wf_est_s = np.zeros([num_est, tsteps]) + 100
    wf_est_s[:, ::r] = wf_est

    """Fill in the empty values from stretching"""
    for p in range(r-1):
        wf_est_s[:, (p+1)::r] = wf_est_s[:, ::r]

    mse = np.mean((wf_act - wf_est_s)**2, axis=1)

    return mse, wf_est_s
