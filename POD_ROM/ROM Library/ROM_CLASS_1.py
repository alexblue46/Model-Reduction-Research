"""
ROM Suite
"""

import numpy as np
import math as m
from scipy import linalg
from scipy.sparse import diags
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def find_k(Snapshot_m, tol = .001):
    [S ,V ,D] = linalg.svd(Snapshot_m)
    
    counter = 0
    for value in V:
        if value>tol:
            counter+=1
        
    return counter

class ROM():

    def __init__(self,  kconstant, discretization =[], rom_type='POD', x_domain=[[1]], nxs = [20], time_int= 1):
        """
        defines a new R(educed)O(rder)M(odel) object.
        """
        self.disc = discretization
        self.romtype = rom_type
        self.x1 = x_domain[0]
        if len(x_domain)>1:
            self.x2 = x_domain[1]
        self.time = time_int
        self.k = kconstant
        self.n1 = nxs[0]
        if len(nxs)>1:
            self.n2 = nxs[1]

    def set_k(self,new_k):
        """
        changes the number of columns used in the low rank approximation
        """
        self.k = new_k
        return self
    
    def get_romtype(self):
        """
        returns a string describing the model reduction method
        """
        return str(self.romtype)

    def format_disc(self, left, right):
        """
        If discretization is passed through as list of matrices and coefficients, [[matrices],[coefficients]]
        then we efficiently multiply on the left or right by matrices. 'left' and 'right' are matrices whose dimensions match 
        the discretization matrix
        """
        matrices = self.disc[0]
        coeff = self.disc[1]
        
        summation = np.zeros((np.shape(left)[0],np.shape(right)[1]))

        for i in range(len(coeff)):
            summation += coeff[i]* (left @ matrices[:,:,i] @ right)
        
        return summation

    def pod_method(self, Y, y0, x_domain, plot_it = False, x2=False, nt=100, t_step=10):
        """
        Implementation of the POD method. Returns the snapshot matrix.
        """
        time_int = np.linspace(0,self.time,nt)
        if x2 == False:
            Ymean = np.mean(Y, 1)
            
            Y = Y - (Ymean @ np.ones((self.n1,nt)))
            U, S, V = linalg.svd(Y, full_matrices = True)
            
            Uk = U[:,0:self.k]
            Ukt = np.transpose(Uk)
            B =  Ukt @ self.disc @ Uk
            y0k = Ukt @ (y0-Ymean)

            ###Solve POD Dynamical System
            Y_hat= np.zeros((self.k,nt))

            Y_hat[:,0]=y0k

            for j in np.arange(1,nt,1):
                Y_hat[:,j] = linalg.expm(B*time_int[j]) @ np.array(y0k) + linalg.inv(B) @ (linalg.expm(B*time_int[j]) - np.eye(self.k)) @ np.transpose(Uk) @ self.disc @ Ymean
            ##################
            if plot_it == True:
                [x,y] = np.meshgrid(self.x1,time_int)
                fig = plt.figure()
                ax = plt.axes(projection='3d')
                ax.plot_surface(x, y, np.transpose(Uk @ Y_hat + Ymean @ np.ones((self.n1,nt))), rstride=1, cstride=1,cmap='viridis',edgecolor='none')
                plt.xlabel('x')
                plt.ylabel('time')
                plt.title(f'POD-Plot, k={self.k}')
                plt.show()

            return Y_hat
        else:
            DELTA_T =self.time/float(nt)
            ymean = np.mean(Y, 1)
            ymean = np.reshape(ymean,(self.n1*self.n2,1))
            ej=np.ones((1,nt+1))
            Y -= ymean @ ej
            U, S, V = linalg.svd(Y, full_matrices = True)
            Uk = U[: , 0:self.k]
            Ak = np.transpose(Uk) @ self.disc @ Uk
            Ik = np.eye(self.k)
            Yk = np.zeros((self.k, nt +1))

            y0k = np.transpose(Uk) @ (np.reshape(y0, (self.n1*self.n2,1)) - ymean)
            y0k = np.reshape(y0k,(self.k,))
            yAk = np.transpose(Uk) @ (self.disc @ ymean)
            yAk = np.reshape(yAk,(self.k,))
            Yk[: ,0] = y0k


            for it in range(nt):
                Yk[: , it+1] = linalg.solve((Ik-DELTA_T*Ak),(Yk[: , it] + DELTA_T* yAk ))

            Y += ymean @ np.ones((1,nt+1))

            if plot_it == True:
                for it in np.arange(0,nt+1,t_step):
                    fig = plt.figure() 
                    yPOD = Uk @ np.reshape(Yk[: , it],(self.k,1)) + ymean
                    yplot = np.zeros((self.n1+1,self.n2+2))
                    yplot[1:self.n1+1,1:self.n2+1] = np.reshape(yPOD,(self.n1 ,self.n2))
                    yplot[0 ,:] = yplot[self.n1,:]
                    
                    [X1,X2]= np.meshgrid(self.x1,self.x2)
                    ax = plt.axes(projection='3d')
                    ax.plot_surface(X1, X2, np.transpose(yplot), rstride=1, cstride=1,cmap='viridis',edgecolor='none')
                    plt.xlabel('x1')
                    plt.ylabel('x2')
                    plt.title(f't={it/100}')
                    plt.show()
            
            return Yk

    def solve_equation(self, initial_condition, parameter = [], f = [], method = 'exp', x2=False, nt= 100):    
        """
        solves system in one of a few ways.
        1) matrix exponential : 'exp'.
        2) backward euler : 'eul'
        3) linear system : 'lin'

        initial_condition must be a function
        """

        if method == 'exp':
            if x2==False:
                y0 = [initial_condition(x) for x in self.x1]
                time_int = np.linspace(0,self.time,nt)
                ###Solve Dynamical System
                Y= np.zeros((len(time_int),nt))

                Y[:,0]=y0

                for j in np.arange(2,nt,1):
                    Y[:,j] = linalg.expm(self.disc*time_int[j]) @ np.array(y0)

            else:
                return 'Error: only 1-spatial dimension is allowed for this solving technique'
            return Y
        if method == 'eul':
            if x2 == True:
                time_int = np.linspace(0,self.time,nt)
                delta_t = self.time/float(nt)
                nt = len(time_int)

                #initial condition
                y0 = np.zeros((self.n1 + 1, self.n2 + 2))

                for i in range(self.n1 + 1):
                    for j in range(self.n2 + 2):
                        y0[i,j] = initial_condition(self.x1[i],self.x2[j])

                y0 = y0[1:,1:self.n2+1]

                I = np.eye(np.shape(self.disc)[0])
                Y = np.zeros((self.n1*self.n2,nt + 1))
                Y[:,0] = np.reshape(y0,(self.n1*self.n2))

                #Solve
                for i_t in range(nt):
                    Y[:, i_t + 1] = linalg.solve((I-delta_t*self.disc), Y[:,i_t])
                return Y
        else:
            return 'Error: only 2-spatial dimension is allowed for this solving technique'

        if method == 'lin':
            if x2==True:
                Y = np.zeros((self.n1*self.n2,len(parameter)))

                for param in parameter:
                    Y[:,parameter.index(param)] = linalg.solve(self.disc[:,:,parameter.index(param)],f)
        else:
            return 'Error: only 2-spatial dimension is allowed for this solving technique'
        return Y
            
    def plot_snapshot_FD(self, snapshot, ylabel, xlabel, x2 = False,nt=100):
        """
        Plots the snapshot matrix in a surface plot. Only works for 2 spatial dimensions or less. 
        A different function is required to plot 3+ spatial dimensional problems
        """
        time_int = np.linspace(0,self.time,nt)
        if x2 == False:
            [x,y] = np.meshgrid(self.x1,time_int)
        else:
            [x,y] = np.meshgrid(self.x1,self.x2)

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(x, y, np.transpose(snapshot), rstride=1, cstride=1,cmap='viridis',edgecolor='none')
    
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f'FD-plot, k={self.k}')
        plt.show()

    def smart_set_k(self,Snapshot_m,tol=.001):
        self.set_k(find_k(Snapshot_m,tol))