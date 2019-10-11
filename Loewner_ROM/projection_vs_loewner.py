from Laplace_ROM_Functions import *
from model_problems import *

import numpy as np

# Set up FOMs

d1_vals = model_problem()
d2_vals = model_problem(dimension = 2, nxs = [30, 30], x_range = ([0, 1], [0, 1]))

# set D = 0 in both FOMs

d1_vals['D'] = 0
d2_vals['D'] = 0


##################### Loewner and Projection Method Comparison ###############
method = 'Alternate'
r = 10
tol = 1e-12
freqs = 1j * np.logspace(-3, 4, r)
convert_to_real = True
ts = np.linspace(0, 0.5, 50)

def u(t):
    return np.sin(2 * np.pi * t)

# 1 Dimensional
   
data_d1 = get_data(freqs, d1_vals, method)
d1_hat_vals_P = Projection_Interpolate(d1_vals, freqs, tol, convert_to_real)
d1_hat_vals_L = Loewner_Framework(data_d1, tol, convert_to_real)
transfer_func_error(d1_vals, [d1_hat_vals_P, d1_hat_vals_L], 
                    ["Projection H_hat", "Loewner H_hat"], 
                    data = data_d1, 
                    title = "One Dimensional Error with No Noise")

if convert_to_real:
    output_error(d1_vals, [d1_hat_vals_P, d1_hat_vals_L], 
                 ["Projection H_hat", "Loewner H_hat"], 
                 ts, u, 
                 title = "One Dimensional Error with No Noise")

# 2 Dimensional

data_d2 = get_data(freqs, d2_vals, method)
d2_hat_vals_P = Projection_Interpolate(d2_vals, freqs, tol, convert_to_real)
d2_hat_vals_L = Loewner_Framework(data_d2, tol, convert_to_real)
transfer_func_error(d2_vals, [d2_hat_vals_P, d2_hat_vals_L], 
                    ["Projection H_hat", "Loewner H_hat"], 
                    data = data_d2, 
                    title = "Two Dimensional Error with No Noise")

if convert_to_real:
    output_error(d2_vals, [d2_hat_vals_P, d2_hat_vals_L], 
                 ["Projection H_hat", "Loewner H_hat"], 
                 ts, u, 
                 title = "Two Dimensional Error with No Noise")
