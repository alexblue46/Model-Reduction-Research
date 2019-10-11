import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def Projection_Interpolate(vals, s_vals, tol, convert_to_real, show_sing_vals = False, title = ""):
    """
    Inputs:
        vals - Dictionary of matrices A, B, C, D, and E
        s_vals - Numpy array of the frequencies at which to train the ROM 
                (assumed to be imaginary)
        tol - Tolerance with which to cut off the SVD of the V and W matrices
        convert_to_real - Boolean indicating whether the ROM will be real 
        show_sing_vals - Boolean indicating whether the normalized singular 
                         values of the V and W matrices will be plotted
        
    Returns a dictionary of hat matrices A, B, C, D, and E which make a ROM
    of transfer function. The ROM transfer function will equal the FOM transfer
    function at the frequencies in s_vals.
    """
    size = (vals['A'].shape[0], s_vals.shape[0])
    
    #Generate Interpolation Matrices V and W
    V = np.zeros(size, dtype = 'complex')
    W = np.zeros(size, dtype = 'complex')
    for s_idx in range(s_vals.shape[0]):
        V[:, s_idx] = np.reshape(
            np.linalg.solve((s_vals[s_idx] * vals['E'] - vals['A']), vals['B']), (size[0]))
        W[:, s_idx] = np.reshape(
            np.linalg.solve((s_vals[s_idx] * vals['E'] - np.transpose(vals['A'])), vals['C']), (size[0]))
    
    if convert_to_real:
        #Convert complex matrices to real matrices
        real_matrices = []
        for X in [V, W]:
            result = np.zeros((X.shape[0], X.shape[1] * 2))
            result[:, :X.shape[1]] = np.real(X)
            result[:, X.shape[1]:] = np.imag(X)
            real_matrices.append(result)
            
        V = real_matrices[0]
        W = real_matrices[1]
        
    u_v, s_v, v_v = linalg.svd(V)
    u_w, s_w, v_w = linalg.svd(W)

    # Remove linearly dependent columns from V and W
    r = 1
    while r <= s_v.shape[0] - 1 and (s_w[r] / s_w[0] > tol or s_v[r] / s_v[0] > tol):
        r += 1
    
    V = u_v[:, :r]
    W = u_w[:, :r]
    
    # Print out information about the FOM and ROM
    print("Projection Based Interpolation")    
    print ("FOM size is: " + str(size[0]) + "\t ROM size is: " + str(r))
    print("")
        
    
    #Calculate Hat Matrices
    A_hat = np.transpose(W) @ vals['A'] @ V
    B_hat = np.transpose(W) @ vals['B']
    C_hat = np.transpose(np.transpose(vals['C']) @ V)
    D_hat = vals['D']
    E_hat = np.transpose(W) @ vals['E'] @ V 
    
    if show_sing_vals:
        # Plot slingular values
        norm_sing1 = [s / s_v[0] for s in s_v]
        norm_sing2 = [s / s_w[0] for s in s_w]
        
        plt.semilogy(range(1, s_v.shape[0] + 1), norm_sing1, 
                     label = 'S1 normalized singular values', linestyle = 'None', marker = 'p')
        plt.semilogy(range(1, s_w.shape[0] + 1), norm_sing2, 
                     label = 'S2 normalized singular values', linestyle = 'None', marker = 'p')
        plt.axhline(tol, color = 'black', linestyle = '--', linewidth = 1, label = 'Tolerance')
        plt.legend(loc = 'best')
        
        if title != "":
            plt.title("Normalized Singular Values: " + title)
        
        plt.show()

    return {'A' : A_hat, 'B' : B_hat, 'C' : C_hat, 'D' : D_hat, 'E' : E_hat}


def Loewner_Framework(data, tol, convert_to_real, show_sing_vals = False, title = ""):
    """
    Inputs:
        data - Tuple of the frequencies and the tranfer function data. This 
               parameter is obtained with a call to get_data
        tol - Tolerance with which to cut off the svd of the Loewner pencils
        convert_to_real - Boolean indicating whether the ROM will be real 
        show_sing_vals - Boolean indicating whether the normalized singular 
                         values of the Loewner pencils will be plotted
        
    Returns a dictionary of hat matrices A, B, C, D, and E which make a ROM
    of transfer function, using the Loewner framework. The ROM transfer 
    function will interpolate at the inputted data.
    """
    freqs = data[0]
    transfer_data = data[1]
    
    num_left = freqs[0].shape[0]
    num_right = freqs[1].shape[0]
    
    print("Loewner Framework Interpolation")
    print("Number of frequencies: " + str(num_left + num_right))
    
    if convert_to_real:
        # Include the complex conjugate of the data
        freqs = (include_conj(freqs[0]), include_conj(freqs[1]))
        transfer_data = (include_conj(transfer_data[0]), include_conj(transfer_data[1]))
        
        num_left *= 2
        num_right *= 2
    
    L = np.zeros((num_left, num_right), dtype = 'complex')
    for i in range(num_left):
        for j in range(num_right):
            L[i][j] = ((transfer_data[0][i] - transfer_data[1][j]) / 
             (freqs[0][i] - freqs[1][j]))

            
    Ls = np.zeros((num_left, num_right), dtype = 'complex')
    for i in range(num_left):
        for j in range(num_right):
            Ls[i][j]= ((transfer_data[0][i] * freqs[0][i] - transfer_data[1][j] * freqs[1][j]) / 
              (freqs[0][i] - freqs[1][j]))
            
    if convert_to_real:
        assert(num_right == num_left)
        
        block = np.array([[1, -1j], [1, 1j]]) * (1. / np.sqrt(2)) 
        J = np.kron(np.eye(num_left // 2), block)
      
        L = np.real(J.conj().T @ L @ J)
        Ls = np.real(J.conj().T @ Ls @ J)
        
    # Construct the Loewner pencils       
    horz_pencil = np.concatenate((L, Ls), axis = 1)
    vert_pencil = np.concatenate((L, Ls), axis = 0)
    
    # Remove the linearly dependent columns of the Loewner pencils
    Y1, S1, X1 = linalg.svd(horz_pencil)
    Y2, S2, X2 = linalg.svd(vert_pencil)
    
    r = 1
    while r <= S2.shape[0] - 1 and (S1[r] / S1[0] > tol or S2[r] / S2[0] > tol):
        r += 1
        
        
    Y = Y1[:, :r]
    X = X2[:, :r]
    
    # Compute hat matrices
    A_hat = -Y.conj().T @ Ls @ X
    
    if convert_to_real:
        B_hat = np.real(Y.conj().T @ J.conj().T @ transfer_data[0].reshape((num_left, 1)))
        C_hat = np.real(np.transpose(transfer_data[1].reshape((1, num_right)) @ J @ X))
    else:
        B_hat = Y.conj().T @ transfer_data[0].reshape((num_left, 1))
        C_hat = np.transpose(transfer_data[1].reshape((1, num_right)) @ X)
    
    D_hat = 0
    E_hat = -Y.conj().T @ L @ X
    
    print("Size of ROM: " + str(r))
    print("")
    
    if show_sing_vals:
        # Plot slingular values
        norm_sing1 = [s / S1[0] for s in S1]
        norm_sing2 = [s / S2[0] for s in S2]
        
        plt.semilogy(range(1, S1.shape[0] + 1), norm_sing1, 
                     label = 'S1 normalized singular values', linestyle = 'None', marker = 'p')
        plt.semilogy(range(1, S2.shape[0] + 1), norm_sing2, 
                     label = 'S2 normalized singular values', linestyle = 'None', marker = 'p')
        plt.axhline(tol, color = 'black', linestyle = '--', linewidth = 1, label = 'Tolerance')
        plt.legend(loc = 'best')
        
        if title != "":
            plt.title("Normalize Singular Values: " + title)
        
        plt.show()
    
    return {'A' : A_hat, 'B' : B_hat, 'C' : C_hat, 'D' : D_hat, 'E' : E_hat}


def get_transfer_func(vals):
    """
    Inputs:
        vals - Dictionary of matrices A, B, C, D, and E
        
    Returns a function H(s), where H(s) is equal to the transfer function with
    the inputted matrices.
    """
    def H(s):
        return (np.transpose(vals['C']) @ linalg.solve(s * vals['E'] - vals['A'], vals['B']) + vals['D'])[0][0]
    return H


def transfer_func_error(vals, hat_val_list, label_list, data = None, title = ""):
    """
    Inputs:
        vals - Dictionary of matrices A, B, C, D, and E
        hat_val_list - List of dictionaries of hat matrices A, B, C, D, and E
        label_list - List of strings describing the ROMs in hat_val_list
        data - Tuple of the frequencies and the tranfer function data. This 
               parameter is obtained with a call to get_data.
        title - (optional) title for the graph
        
    Plots the error between the FOM transfer function and the ROM tranfer 
    functions. Also plots the error between the FOM transfer function and the
    given tranfer function data.
    """
    H = get_transfer_func(vals)
    H_vals = [H(s * 1j) for s in np.logspace(-3, 4, 200)]
    
    for (hat_vals, label) in zip(hat_val_list, label_list):
       H_hat = get_transfer_func(hat_vals)
       H_hat_vals = [H_hat(s * 1j) for s in np.logspace(-3, 4, 200)]
       plt.semilogx(np.logspace(-3, 4, 200), [abs(h1 - h2) for (h1, h2) in zip(H_hat_vals, H_vals)], 
                    label = label)
    
    if data != None:
        freqs_trained = np.concatenate((data[0][0], data[0][1]))
        plt.axvline(np.imag(freqs_trained[0]), color = 'black', linestyle = '--', 
                    linewidth = 0.25, label = 'Training Frequencies')
        for freq in freqs_trained[1:]:
            plt.axvline(np.imag(freq), color = 'black', linestyle = '--', linewidth = 0.25)
    
    plt.yscale('log')
    plt.xlabel('Frequency')
    plt.ylabel('Error')
    if title != "":
        plt.title("Transfer Function Error: " + title)
    plt.legend(loc = 'best')
    plt.show()
    

def transfer_func_comparison(vals, hat_val_list, label_list, data = None, title = ""):
    """
    Inputs:
        vals - Dictionary of matrices A, B, C, D, and E
        hat_val_list - List of dictionaries of hat matrices A, B, C, D, and E
        label_list - List of strings describing the ROMs in hat_val_list
        data - Tuple of the frequencies and the tranfer function data. This 
               parameter is obtained with a call to get_data.
        title - (optional) title for the graph
        
    Plots the FOM transfer function and the ROM tranfer functions. Also plots 
    the given tranfer function data.
    """
    H = get_transfer_func(vals)
    H_vals = [abs(H(s * 1j)) for s in np.logspace(-3, 4, 200)]
    plt.semilogx(np.logspace(-3, 4, 200), H_vals, label = 'FOM')
    
    for (hat_vals, label) in zip(hat_val_list, label_list):
       H_hat = get_transfer_func(hat_vals)
       H_hat_vals = [abs(H_hat(s * 1j)) for s in np.logspace(-3, 4, 200)]
       plt.semilogx(np.logspace(-3, 4, 200), H_hat_vals, label = label, linestyle = '--')
    
    if data != None:
        freqs_trained = np.concatenate((data[0][0], data[0][1]))
        observed_H = np.concatenate((data[1][0], data[1][1]))
        plt.semilogx(np.imag(freqs_trained), abs(observed_H), label = "Observed Data", 
                     linestyle = 'None', marker = 'p')
    
    plt.xlabel('Frequency')
    if title != "":
        plt.title("Transfer Function Comparison: " + title)
    plt.legend(loc = 'best')
    plt.show()

  
def output_error(vals, hat_val_list, label_list, ts, u_func, title = ""):
    """
    Inputs:
        vals - Dictionary of matrices A, B, C, D, and E
        hat_val_list - List of dictionaries of hat matrices A, B, C, D, and E
        label_list - List of strings describing the ROMs in hat_val_list
        ts - Numpy array of points at which to plot the output function
        u_func - Function for u(t)
        title - (optional) title for the graph
        
    Plots the error between the output of the FOM and the ROMs for times in ts.
    """
    us = np.asarray([u_func(t) for t in ts])
    
    fom_size = vals['A'].shape[0]
    
    def func(t, y):
        return (linalg.solve(vals['E'], vals['A'] @ y.reshape((fom_size, 1))) 
                + linalg.solve(vals['E'], vals['B'] * u_func(t))).reshape((fom_size))
    
    y0 = np.zeros((fom_size))
    fom_y = solve_ivp(func, (0, ts[-1]), y0, t_eval = ts).y
    fom_output = np.transpose(vals['C']) @ fom_y + vals['D'] * us
    
    for (hat_vals, label) in zip(hat_val_list, label_list):
        
        rom_size = hat_vals['A'].shape[0]
        
        def func_hat(t, y):
            return (linalg.solve(hat_vals['E'], hat_vals['A'] @ y.reshape((rom_size, 1))) 
                    + linalg.solve(hat_vals['E'], hat_vals['B'] * u_func(t))).reshape((rom_size))
    
        y0_hat = np.zeros((rom_size))
        rom_y = solve_ivp(func_hat, (0, ts[-1]), y0_hat, t_eval = ts).y
        rom_output = np.transpose(hat_vals['C']) @ rom_y + hat_vals['D'] * us
        plt.semilogy(ts, abs(rom_output - fom_output).reshape((ts.shape[0])), label = label)
    
    plt.xlabel('t')
    plt.ylabel('Error')
    if title != "":
        plt.title("Output Error: " + title)
    plt.legend(loc = 'best')
    plt.show()
    
def output_comparison(vals, hat_val_list, label_list, ts, u_func, title = ""):
    """
    Inputs:
        vals - Dictionary of matrices A, B, C, D, and E
        hat_val_list - List of dictionaries of hat matrices A, B, C, D, and E
        label_list - List of strings describing the ROMs in hat_val_list
        ts - Numpy array of points at which to plot the output function
        u_func - Function for u(t)
        title - (optional) title for the graph
        
    Plots the output of the FOM and the ROMs for times in ts.
    """
    us = np.asarray([u_func(t) for t in ts])
    
    fom_size = vals['A'].shape[0]
    
    def func(t, y):
        return (linalg.solve(vals['E'], vals['A'] @ y.reshape((fom_size, 1))) 
                + linalg.solve(vals['E'], vals['B'] * u_func(t))).reshape((fom_size))
    
    y0 = np.zeros((fom_size))
    fom_y = solve_ivp(func, (0, ts[-1]), y0, t_eval = ts).y
    fom_output = np.transpose(vals['C']) @ fom_y + vals['D'] * us
    
    plt.plot(ts, fom_output.reshape((ts.shape[0])), label = 'FOM')
    
    for (hat_vals, label) in zip(hat_val_list, label_list):
        
        rom_size = hat_vals['A'].shape[0]
        
        def func_hat(t, y):
            return (linalg.solve(hat_vals['E'], hat_vals['A'] @ y.reshape((rom_size, 1))) 
                    + linalg.solve(hat_vals['E'], hat_vals['B'] * u_func(t))).reshape((rom_size))
    
        y0_hat = np.zeros((rom_size))
        rom_y = solve_ivp(func_hat, (0, ts[-1]), y0_hat, t_eval = ts).y
        rom_output = np.transpose(hat_vals['C']) @ rom_y + hat_vals['D'] * us
        plt.plot(ts, rom_output.reshape((ts.shape[0])), label = label, linestyle = '--')
    
    plt.xlabel('t')
    plt.ylabel('Output')
    if title != "":
        plt.title("Output Comparison: " + title)
    plt.legend(loc = 'best')
    plt.show()
    
    
def plot_eigenvalues(val_list, legends, title = ""):
    """
    Inputs:
        val_list - list of dictionaries containing matrices A, B, C, D, and E
        legends - list of strings describing a dictionary of matrices in val_list
        title - (optional) title for the graph
    
    Plots the generalized eigenvalues of each dictionary of matrices. Generalized
    eigenvalues are x where det(A - Ex) = 0
    """
    for (vals, legend) in zip(val_list, legends):
        w, v = linalg.eig(vals['A'], b = vals['E'])
        plt.plot(np.real(w), np.imag(w), linestyle = 'None', marker = 'p', label = legend)
        
    plt.axvline(0, color = 'black', linestyle = '--', linewidth = 1) 
    plt.axhline(0, color = 'black', linestyle = '--', linewidth = 1)
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.title(title)
    plt.legend(loc = 'best')
    plt.show()

    
def include_conj(data_array):
    """
    Inputs:
        data_list - numpy array of complex numbers
    
    Returns an array of complex numbers where entries 2i and 2i + 1 equal 
    data_array[i] and the complex conjugate of data_array[i], respectively
    """
    result = np.zeros((data_array.shape[0] * 2), dtype = 'complex')
    result[np.arange(0, data_array.shape[0] * 2, 2)] = data_array
    result[np.arange(1, data_array.shape[0] * 2, 2)] = np.conj(data_array)
        
    return result


def get_data(freqs, vals, method, noise = 0, seed = None):
    """
    Inputs:
        freqs - Numpy array of frequencies for which to compute the transfer 
                function data
        vals - dictionary of matrices A, B, C, D, and E
        method - 'Alternate' if right and left frequencies alternate in freqs,
                 'Block' if right and left frequencies make up the first and last 
                  entries of freqs
        noise - percent of noise added to the transfer function data
        seed - natural number to make the results of get_data reproducable
    
    Produces the right and left driving frequencies as well as the right and
    left driving transfer function data. Returns a tuple that is used as the 
    data paramter in the Loewner_Framework function.
    """
    H = get_transfer_func(vals)
    
    transfer_data = np.zeros((freqs.shape[0]), dtype = 'complex')
    
    for freq_idx in range(freqs.shape[0]):
        transfer_data[freq_idx] = H(freqs[freq_idx])
        
    if seed != None:
        np.random.seed(seed)
            
    real_noise = np.random.randn(freqs.shape[0]) * noise
    imag_noise = np.random.randn(freqs.shape[0]) * noise
    
    transfer_data += (np.multiply(np.real(transfer_data), real_noise) 
            + 1j * np.multiply(np.imag(transfer_data), imag_noise))
    
    if method == 'Alternate':
        freq_split = (freqs[np.arange(0, freqs.shape[0], 2)], 
                            freqs[np.arange(1, freqs.shape[0], 2)])
        
        transfer_data_split = (transfer_data[np.arange(0, freqs.shape[0], 2)], 
                            transfer_data[np.arange(1, freqs.shape[0], 2)])
            
    if method == 'Block':
        freq_split = (freqs[:freqs.shape[0] // 2], freqs[freqs.shape[0] // 2:])
        transfer_data_split = (transfer_data[:freqs.shape[0] // 2], 
                               transfer_data[freqs.shape[0] // 2:])

    return freq_split, transfer_data_split