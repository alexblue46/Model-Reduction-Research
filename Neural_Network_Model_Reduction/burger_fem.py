"""
Burger_fem.py

Solves steady-state Burger's Equation with homogeneous Dirichlet boundary
conditions by finding the Galerkin approximation using Newton's Method with 
Armijo Line-Search Globalization.

-viscosity * v_xx(x) + v(x) * v_x(x) = f(x); x in (0,1)
v(0) = v(1) = 0

Transforms steady-state Burger's Equation to the form: F(v) = A*v + G(v) - b = 0, 
where v is the Galerkin approximation of the solution.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
import math


def G(v):
    """
    Given an input vector v, computes the function G(v).
    """
    
    result = np.zeros((v.size))
    result[0] = v[0]*v[1] + v[1]**2
    for i in range(1, v.size - 1):
        result[i] = -1*v[i-1]**2 - v[i]*v[i-1] + v[i]*v[i+1] + v[i+1]**2
    result[-1] = -1*v[-2]**2 - v[-1]*v[-2] 
    return result / 6.0


def Gp(v):
    """
    Given an input vector v, computes the Jacobian of the function G at v.
    """
    
    e1 = np.add(2 * v[0:-1], -1 * v[1:])
    
    e2 = np.zeros((v.size))
    e2[0] = v[1] 
    e2[1:-1] = np.add(v[2:], -1 * v[0:-2])
    e2[-1] = -1 * v[-2]

    e3 = np.add(v[0:-1], 2*v[1:])
    
    return diags([e1, e2, e3], [-1, 0, 1], shape=(v.size, v.size)).toarray() / 6.0


def eval_soln_norm(A, G_func, b, v):
    """
    Given a matrix A, a function to evaluate G, and a vector b, computes
    the norm of F(v).
    """
    
    return np.linalg.norm(eval_F(A, G_func, b, v))


def eval_F(A, G_func, b, v):
    """
    Given a matrix A, a function to evaluate G, and a vector b, computes F(v).
    """
    
    return np.matmul(A, v) + G_func(v) - b


def eval_Fp(A, Gp_func, v):
    """
    Given a matrix A, a function to evaluate the jacobian of G, and a vector b, 
    computes the jacobian of F(v) at v.
    """
    
    return np.add(A, Gp_func(v))


def check_deriv(A, b, G_func, Gp_func):
    """
    Given a matrix A, a function to evaluate G, a function to evaluate the 
    jacobian of G, and a vector b, plots the sum of the errors at each node 
    between the approximated derivative of F and the actual derivative of F 
    for a random v.
    """
    
    h_vals = np.linspace(1e-10, 1e-8, 100)
    v = np.random.rand(np.size(b))
    errors = np.zeros((100))
    error_idx = 0
    for h in h_vals:
        for i in range(np.size(b)):
            errors[error_idx] += np.linalg.norm(np.add(Gp_func(v)[:,i], 
                  -1 * estimate_deriv(v, A, b, G_func, h, i)))
        error_idx += 1
    plt.plot(h_vals, errors)
    plt.xlabel("h")
    plt.ylabel("Sum of Errors")
    plt.title("Error of Estimated Derivative at Random v", y = 1.08)
    plt.show()
    return


def estimate_deriv(v, A, b, G_func, h, i):
    """
    Given a matrix A, a function to evaluate G, and a vector b, estimates 
    the derivative of F(v) in the direction of the ith node with a step size 
    of h.
    """
    e = np.zeros((np.size(v)))
    e[i] = h
    return (1.0 / h) * np.add(eval_F(A, G_func, b, np.add(v, e)), 
            -1 * eval_F(A, G_func, b, v))
    
  
def burger_fem(f, viscosity, maxiter, tol, show_deriv, show_converg, plot_sol):
    """
    Inputs:
        f - vector of forces at each node
        viscosity - scalar of viscosity
        maxiter - maximum number of Newton's method iterations before terminating
        tol - tolerance that F(v) must admit to before Newton's method terminates
        show_deriv - boolean indicating whether the plot verifying the derivative
                      of F(v) will be shown
        show_converg - boolean indicating whether the plot showing the convergence
                       of Newton's method will be shown
        plot_sol - boolean indicating whether to plot the solution
    
    Returns:
        v - approximated solution to Burger's equation
                   
    Solves the steady-state Burger's Equation by finding the Galerkin 
    approximation using Newton's Method with Armijo Line-Search Globalization. 
    Produces a plot of the Galerkin approximation of the solution with the 
    force vector f.
    """
    
    nx = np.size(f) - 1
    tol_req = False
    sigma = 1e-4
    deltax = 1.0 / nx
    e = np.ones((nx - 1))
    x = np.linspace(0, 1, num=(nx + 1))
    v = np.sin(math.pi*x)[1:-1]
    #v = np.zeros(nx-1)
    errors = []
    
    # Initialize matrix A and vector b
    A = (1.0 * viscosity / deltax) * diags([-e[1:], 2*e, -e[1:]], [-1, 0, 1], 
        shape=(nx - 1, nx - 1)).toarray()    
    b = deltax * f[1:-1]
    
    # Display plot verifying the derivative of F(v)
    if (show_deriv):
        check_deriv(A, b, G, Gp)
    
    # Execute Newton's method with Armijo line-search globalization
    for k in range(maxiter):
        errors.append(eval_soln_norm(A, G, b, v))
        
        # Break out of Newton's method if F(v) is arbitrarily close of 0
        if (eval_soln_norm(A, G, b, v) < tol):
            tol_req = True
            break
        
        Fp = eval_Fp(A, Gp, v)
        s = np.linalg.solve(Fp, -1 * np.transpose(eval_F(A, G, b, v)))
        t = 1.0
        v_new = np.add(v, t * s)
        while (eval_soln_norm(A, G, b, v_new) > (1 - 2*sigma*t) * eval_soln_norm(A, G, b, v)):
            t = t / 2
            v_new = np.add(v, t * s)
        v = v_new
    
    # Plot the norm of F(v) after each iteration of Newton's method    
    if (show_converg):
        plt.semilogy(np.array(errors))
        plt.title("Convergence of Newton's Method with Armijo Line-Search Globalization")
        plt.xlabel("Iteration Number")
        plt.ylabel("Error of Solution")
        plt.show()
     
    # Pad v with zeros because v(0) = v(1) = 0    
    v = np.pad(v, (1,1), 'constant', constant_values=(0, 0))
    
    if (plot_sol):
    
        # Plot the solution v with the force vector f
        titlestr = "Steady-State Burger's Equation Solved via "
        if (tol_req):
            titlestr += "Satisfying Tolerance Requirement with Viscosity = "
        else:
            titlestr += "Reaching Maximum Iteration with Viscosity = "
        titlestr += str(viscosity)
      
        plt.plot(np.linspace(0, 1, nx + 1), v, label = "v(x)")
        plt.plot(np.linspace(0, 1, nx + 1), f, label = "f(x)")
        plt.legend(loc = "upper left")
        plt.xlabel("x")
        plt.title(titlestr)
        plt.show()
    
    return v



