# phase_amp_tracking_nn
This is the code for the project: "Channel noise tracking for sub-shot-noise-limited receivers with neural networks"

The files are:

1. TrainModel -> this trains the network using tensorflow. All network and training parameters can be specified here to get the best results including sample weights, network arcitecture, cost function, activation function, etc...

2. DefModel -> This is used with TrainModel and constructs the network based on the supplied parameters without having to change anything

3. AllFunctions -> This is a file which contains the necessary extra functions to run the simulations using the neural network estimator, the bayesian estimator, and the reference measurements such as perfect correction and heterodyne

4. RunMain -> This code is for running the simulations whih generate the plots in the paper. Mainly for the scans of the channel noise parameters.

5. RunScan -> This module is used by RunMain to implement a scan of the supplied parameters and supplied type (phase, amplitude...)

6. RunExperiments -> This module contains each separate noise tracking method (NN, Bayesian) and the reference measurements. The measurements are designed to be parallelized which works well for the Bayesian based approach. The NN is parallelized only in the simulated state discrimination part, the estimation using tensorflow and Kalman filtering is not done in parallel.

