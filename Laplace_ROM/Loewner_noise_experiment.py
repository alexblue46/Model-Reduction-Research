from Laplace_ROM_Functions import *
from model_problems import *

import numpy as np
from scipy.sparse import diags

# Set up FOMs

d1_vals = model_problem()
d2_vals = model_problem(dimension = 2, nxs = [30, 30], x_range = ([0, 1], [0, 1]))

# set D = 0 in both FOMs

d1_vals['D'] = 0
d2_vals['D'] = 0


##################### NOISE EXPERIMENT ########################
def u(t):
    return np.sin(2 * np.pi * t)


method = 'Alternate'
r = 20
tol = 1e-12
freqs = 1j * np.logspace(-3, 4, r)
convert_to_real = False
ts = np.linspace(0, 0.5, 50)

# 1 Dimensional

data_d1_0 = get_data(freqs, d1_vals, method)
data_d1_1 = get_data(freqs, d1_vals, method, noise = 0.01)
data_d1_5 = get_data(freqs, d1_vals, method, noise = 0.05)

d1_hat_vals_0 = Loewner_Framework(data_d1_0, tol, convert_to_real)
d1_hat_vals_1 = Loewner_Framework(data_d1_1, tol, convert_to_real)
d1_hat_vals_5 = Loewner_Framework(data_d1_5, tol, convert_to_real)

transfer_func_error(d1_vals, [d1_hat_vals_0, d1_hat_vals_1, d1_hat_vals_5], 
                    ["No Noise", "1% Noise", "5% Noise"],
                    data_d1_0,
                    title = "One Dimensional Error with Various Levels of Noise")
if convert_to_real:
    output_error(d1_vals, [d1_hat_vals_0, d1_hat_vals_1, d1_hat_vals_5], 
                 ["No Noise", "1% Noise", "5% Noise"], ts, u,
                 title = "One Dimensional Error with Various Levels of Noise")
    
plot_eigenvalues([d1_hat_vals_0, d1_hat_vals_1, d1_hat_vals_5], 
                 ["No Noise", "1% Noise", "5% Noise"],
                 title = "One Dimensional Eigenvalues with Various Levels of Noise")

# 2 Dimensional

data_d2_0 = get_data(freqs, d2_vals, method)
data_d2_1 = get_data(freqs, d2_vals, method, noise = 0.01)
data_d2_5 = get_data(freqs, d2_vals, method, noise = 0.05)

d2_hat_vals_0 = Loewner_Framework(data_d2_0, tol, convert_to_real)
d2_hat_vals_1 = Loewner_Framework(data_d2_1, tol, convert_to_real)
d2_hat_vals_5 = Loewner_Framework(data_d2_5, tol, convert_to_real)

transfer_func_error(d2_vals, [d2_hat_vals_0, d2_hat_vals_1, d2_hat_vals_5], 
                    ["No Noise", "1% Noise", "5% Noise"],
                    data = data_d1_0,
                    title = "Two Dimensional Error with Various Levels of Noise")
if convert_to_real:
    output_error(d2_vals, [d2_hat_vals_0, d2_hat_vals_1, d2_hat_vals_5], 
                 ["No Noise", "1% Noise", "5% Noise"], ts, u,
                 title = "Two Dimensional Error with Various Levels of Noise")

plot_eigenvalues([d2_hat_vals_0, d2_hat_vals_1, d2_hat_vals_5], 
                 ["No Noise", "1% Noise", "5% Noise"],
                 title = "Two Dimensional Eigenvalues with Various Levels of Noise")


