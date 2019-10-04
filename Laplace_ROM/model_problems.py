import numpy as np
from scipy.sparse import diags

def model_problem(alpha=.01, beta=1, nxs=[30], x_range=[0,1], dimension = 1, customize=[False]):
    """
    Inputs:
    alpha-- advection constant, >0
    beta-- diffusion constant, >0
    nxs-- a list of numbers that describes how many subintervals to divide up x_range into. nxs[0]= x1 split, nxs[1] = x2 split
    x_range-- For 1d, a list. For 2d, a list of lists (x_range[0]= x1 interval, xrange[1]= x2 interval).
    dimension-- an integer, either 1 or 2, that denotes whether matrices are being computed for the 1D or 2D model problem
    customize-- a list with a boolean and an array. If false, E= identity. If true, then you must provide an alternative E.
    
    Output: a dictionary with values being the A,B,C,D,E matrices for the general dynamical system
    
    """
    
    if dimension ==1:
        
        #Format x data
        NX = nxs[0]
        h = (x_range[1]-x_range[0])/NX
        
        #Compute Adiff
        
        #set up diagonals
        main_diag = [-2 for dummy in range(NX)]
        lower_diag = [1 for dummy in range(NX-1)]
        upper_diag = [1 for dummy in range(NX-1)]

        Adiff = diags([main_diag,lower_diag,upper_diag],[0,-1,1]).toarray()
        
        #boundary condition
        Adiff[NX-1,NX-2] = 2

        #Compute Aconv
        main_diag = [-1 for dummy in range(NX)]
        Aconv = diags([main_diag,lower_diag],[0,-1]).toarray()
        
        #Compute A,B,C,D
        A =  ((1/h**2) * alpha * Adiff) + (beta * Aconv * (1.0/h))
        b = np.zeros((NX,1))
        b[0][0] = alpha*(1/h**2) + beta*(1.0/h)
        c = [h for dummy in range(NX-1)]
        c.append(h/2.0)
        c = np.reshape(c, (NX , 1))
        d = h/2.0
        
        #Set E, if necessary
        if customize[0]:
            E = customize[1]
        else:
            #otherwise, set to the identity
            E = np.eye(NX)

        return {'A': A, 'B':b, 'C': c, 'D': d, 'E':E}
    
    else:
        #Format x data
        n1 = nxs[0]
        n2 = nxs[1]
        h1 = (x_range[0][1]-x_range[0][0])/n1
        h2 = (x_range[1][1]-x_range[1][0])/(n2+1)
        
        #Adiff
        
        #Set up diagonals
        main_diag = [-2 for dummy in range(n1)]
        lower_diag = [1 for dummy in range(n1-1)]
        upper_diag = [1 for dummy in range(n1-1)]

        Adiff = diags([main_diag,lower_diag,upper_diag],[0,-1,1]).toarray()
        #Boundary conditions
        Adiff[n1-1,n1-2] = 2

        a1 = Adiff * (1.0 / h1 ** 2)
        a2 = Adiff * (1.0 / h2 ** 2)
        
        #Aconv
        main_diag = [-1 for dummy in range(n1)]
        Aconv = diags([main_diag,lower_diag],[0,-1]).toarray()
        
        #Compute A,B,C,D
        I1 = np.eye(n1)
        I2 = np.eye(n2)
        A = np.kron(I2, (alpha * a1) + (beta* Aconv)) + np.kron(alpha * a2, I1)
        e = np.zeros((n1,1))
        e[0] = (alpha / (h1**2)) + (beta/ h1)
        B = np.kron(I2, e)
        c1 = h2 * np.ones((n2,1))
        c2 = h1 * np.ones((n1,1))
        c2[-1] = c2[-1] / 2
        C = np.kron(c1, c2)
        D = (h1*h2*.5)*np.ones((n2,))
        e = np.ones((n2,1))
        B = B @ e
        D = D @ e
        
        #Set E if necessary
        if customize[0]:
            E = customize[1]
        else:
            #E= Identity otherwise
            E = np.eye(n1*n2)

        return {'A': A, 'B':B, 'C': C, 'D': D, 'E':E}
        