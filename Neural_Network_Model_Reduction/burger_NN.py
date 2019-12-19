import tensorflow as tf
import numpy as np
import burger_fem as bf
import matplotlib.pyplot as plt

"""
burger_NN.py

Creates a neural network using TensorFlow to approximate the solution to 
steady-state Burger's equation with the following form:
    
-viscosity * u_xx(x) + u(x) * u_x(x) = f(x); x in (0,1)
v(0) = v(1) = 0

This program uses f(x) = 1 for x in (1/3, 2/3) and 0 elsewhere and viscosity
values ranging from 0.01 to 1.

Training data for the neural network is generated from the Galerkin approximation 
of burgers equation using Newton's Method with Armijo Line-Search Globalization.
"""

F = np.asarray([0]*100 + [1]*100 + [0]*100)
FILENAME = "piecewise_constants.npz"

def generate_all_training(num_data, visc_min, visc_max, tol, f, filename):
    """
    Inputs:
        num_data - number of training data to generate
        visc_min - minimum viscosity
        visc_max - maximum viscosity
        tol - tolerance used in Newton's method to compute training data
        f - force vector
        filename - name of file to save the training data to
        
    Computes the Galerkin approximation of burgers equation using Newton's 
    Method with Armijo Line-Search Globalization for each viscosity in the 
    range of viscosities. Saves the solutions to the inputted file name.
    """
    viscosities = np.logspace(np.log10(visc_min), np.log10(visc_max), num_data)
    u = []
    for visc in viscosities:
        u.append(bf.burger_fem(f, visc, 1000, tol, False, False, False))

    np.savez(filename, u=np.asarray(u), viscosities = viscosities,
             num_mesh = np.size(f), f = f)
    
def choose_training(num_visc, visc_min, visc_max, filename):
    """
    Inputs:
        num_data - number of training data to generate
        visc_min - minimum viscosity
        visc_max - maximum viscosity
        filename - name of file to load the training data from
    
    Loads the training data produced from the generate_all_training function
    and chooses a sect of the entire training data to use for training the neural
    network. 
    """
    database = np.load(filename)
    viscosities = database['viscosities']
    idx_min = 0
    idx_max = np.size(viscosities) - 1
    for idx in range(np.size(viscosities)):
        if idx != 0 and idx != np.size(viscosities) - 1:
            if viscosities[idx-1] <= visc_min and viscosities[idx] > visc_min:
                idx_min = idx - 1
            if viscosities[idx+1] >= visc_max and viscosities[idx] < visc_max:
                idx_max = idx + 1
    if num_visc >= idx_max - idx_min + 1:
        visc_idxs = np.arange(idx_min, idx_max + 1)
    else:
        visc_idxs = np.arange(idx_min, idx_max + 1, (idx_max - idx_min + num_visc) // 
                         num_visc)
    u = database['u']
    f = database['f']
    
    return u[visc_idxs], viscosities[visc_idxs], f
    

def initialize_NN(layers):
    """
    Inputs:
        layers - array of the number of nodes in each of the neural network's 
                 layers
                        
    Initializes the weights and biases of the neural network.
    """
    weights = []
    biases = []
    num_layers = len(layers) 
    for l in range(num_layers-1):
        W = xavier_init(size=[layers[l], layers[l+1]])
        b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float64), 
                        dtype=tf.float64)
        weights.append(W)
        biases.append(b)
    return weights, biases

def xavier_init(size):
    """
    Inputs:
        size - array of size two, indicating the number of nodes in the current
               layer and the following layer
               
    Helper function for initialize_NN. Creates the variable for the weights at
    the given layer.
    """
    in_dim = size[0]
    out_dim = size[1]
    xavier_stddev = np.sqrt(2.0/(in_dim + out_dim))
    return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, 
                                        dtype=tf.float64), dtype=tf.float64) 
    
def forward_prop(X, weights, biases):
    """
    Inputs:
        X - array of values of input nodes
        weights - array of the weights at each layer in the neural network
        biases - array of the biases at each layer in the neural network
        
    Computes the forward propagation of the neural network and returns the 
    values at the outputs nodes.
    """
    H = X
    for layer in range(len(weights) - 1):
        W = weights[layer]
        b = biases[layer]
        # Activation function is sin
        H = tf.sin(tf.add(tf.matmul(H, W), b))
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y

    
class NN:
    """
    Class to model the neural network
    """
    
    def __init__(self, viscosities, u, f, num_mesh_train, u_layers, f_layers, sol_layers, maxiter):
        """
        Inputs:
            viscosities - training data's viscosities
            u - training data's u(x)
            f - force vector
            num_mesh_train - number of mesh points to train the u and solution
                             neural network with (excludes the points at 0 and 1)
            u_layers - array of the number of nodes in the u neural network
                       (must start with 2 and end with 1)
            f_layers - array of the number of nodes in the f neural network
                       (must start with 4 and end with 1)
            layers - array of number of nodes in the solution neural network
                     (must start with 2 and end with 1)
            maxiter - maximum number of iterations that the neural networks
                      will be trained with
                         
        Initializes the NN object. Defines the loss function for all three 
        neural networks and the algorithms that TensorFlow will use to train the neural
        networks.
        """
        tf.compat.v1.reset_default_graph()
        
        if num_mesh_train >= np.size(f) - 2:
            mesh_idxs = np.arange(0, np.size(f))
        else:
            mesh_idxs = np.arange(0, np.size(f), (np.size(f) + num_mesh_train) // 
                         (num_mesh_train + 1))
            mesh_idxs = np.append(mesh_idxs, np.size(f) - 1)
            
        self.f_u_train = np.asarray([np.tile(f[mesh_idxs], np.size(viscosities))]).transpose()
        self.f_f_train = np.asarray([np.tile(f, np.size(viscosities))]).transpose()
        
        self.x_u_train = np.asarray([np.tile(
                np.linspace(0, 1, np.size(f))[mesh_idxs], np.size(viscosities))]).transpose()
        self.x_f_train = np.asarray([np.tile(
                np.linspace(0, 1, np.size(f)), np.size(viscosities))]).transpose()
    
        self.v_u_train = np.asarray([
                np.repeat(viscosities, min(num_mesh_train + 2, np.size(f)))]).transpose()
        self.v_f_train = np.asarray([np.repeat(viscosities, np.size(f))]).transpose()
        
        self.u = np.asarray([u[:, mesh_idxs].flatten()]).transpose()
        
        self.u_weights, self.u_biases = initialize_NN(u_layers)
        self.f_weights, self.f_biases = initialize_NN(f_layers)
        self.sol_weights, self.sol_biases = initialize_NN(sol_layers)
        
        self.u_tf = tf.placeholder(tf.float64, shape=[None, 1])
        self.x_tf = tf.placeholder(tf.float64, shape=[None, 1])
        self.f_tf = tf.placeholder(tf.float64, shape =[None, 1])
        self.v_tf = tf.placeholder(tf.float64, shape=[None, 1])
        
        self.idn_u_predict = self.run_u_NN(self.x_tf, self.v_tf)
        self.idn_f_predict = self.run_f_idn_NN(self.x_tf, self.v_tf)
        
        self.sol_u_predict = self.run_sol_NN(self.x_tf, self.v_tf)
        self.sol_f_predict = self.run_f_sol_NN(self.x_tf, self.v_tf)
        
        self.u_loss = tf.reduce_sum(tf.square(self.u_tf - self.idn_u_predict))
        self.f_loss = tf.reduce_sum(tf.square(self.f_tf - self.idn_f_predict))
        
        self.sol_loss = tf.reduce_sum(tf.square(self.u_tf - self.sol_u_predict)) + \
                    tf.reduce_sum(tf.square(self.f_tf - self.sol_f_predict))
                    
        self.idn_u_optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.u_loss,
                               var_list = self.u_weights + self.u_biases,
                               method = 'L-BFGS-B',
                               options = {'maxiter': maxiter,
                                          'maxfun': maxiter,
                                          'maxcor': 50,
                                          'maxls': 50,
                                          'ftol': 1.0*np.finfo(float).eps})
    
        self.idn_f_optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.f_loss,
                               var_list = self.f_weights + self.f_biases,
                               method = 'L-BFGS-B',
                               options = {'maxiter': maxiter,
                                          'maxfun': maxiter,
                                          'maxcor': 50,
                                          'maxls': 50,
                                          'ftol': 1.0*np.finfo(float).eps})
    
        self.idn_u_optimizer_Adam = tf.train.AdamOptimizer()
        self.idn_u_train_op_Adam = self.idn_u_optimizer_Adam.minimize(self.u_loss, 
                                   var_list = self.u_weights + self.u_biases)
        
        self.idn_f_optimizer_Adam = tf.train.AdamOptimizer()
        self.idn_f_train_op_Adam = self.idn_f_optimizer_Adam.minimize(self.f_loss, 
                                   var_list = self.f_weights + self.f_biases)
        
        self.sol_optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.sol_loss,
                               var_list = self.sol_weights + self.sol_biases,
                               method = 'L-BFGS-B',
                               options = {'maxiter': maxiter,
                                          'maxfun': maxiter,
                                          'maxcor': 50,
                                          'maxls': 50,
                                          'ftol': 1.0*np.finfo(float).eps})
        
        self.sol_optimizer_Adam = tf.train.AdamOptimizer()
        self.sol_train_op_Adam = self.sol_optimizer_Adam.minimize(self.sol_loss,
                                 var_list = self.sol_weights + self.sol_biases)
        
        self.saver = tf.train.Saver()
        
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
        
    def save(self, filename):
        """
        Inputs:
            filename - name of the file to save to
            
        Saves the NN class's variables to the file.
        """
        self.saver.save(self.sess, filename)
    
    def restore(self, filename):
        """
        Inputs:
            filename - name of the file to import from
            
        Read the NN class's variables from the file.
        """
        imported_graph = tf.train.import_meta_graph(filename + '.meta')
        imported_graph.restore(self.sess, filename)
        
    def run_u_NN(self, x, v):
        """
        Inputs:
            x - x coordinate
            v - viscosity value
            
        Runs the forward propagation of the u neural network and returns the 
        value of the output node.
        """
        return forward_prop(tf.concat([x,v],1), self.u_weights, self.u_biases)
    
    def run_sol_NN(self, x, v):
        """
        Inputs:
            x - x coordinate
            v - viscosity value
            
        Runs the forward propagation of the solution neural network and returns the 
        value of the output node.
        """
        return forward_prop(tf.concat([x,v],1), self.sol_weights, self.sol_biases)
    
    def run_f_idn_NN(self, x, v):
        """
        Inputs:
            x - x coordinate
            v - viscosity value
            
        Runs the forward propagation of the f neural network using the u neural
        network to generate u. Returns the value of the output node.
        """
        u = self.run_u_NN(x, v)
        
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]

        terms = tf.concat([u, u_x, u_xx, v],1)
        
        return forward_prop(terms, self.f_weights, self.f_biases)
    
    def run_f_sol_NN(self, x, v):
        """
        Inputs:
            x - x coordinate
            v - viscosity value
            
        Runs the forward propagation of the f neural network using the solution neural
        network to generate u. Returns the value of the output node.
        """
        u = self.run_sol_NN(x, v)
        
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]

        terms = tf.concat([u, u_x, u_xx, v],1)
        
        return forward_prop(terms, self.f_weights, self.f_biases)
    
    
    def train_u_NN(self, N_iter):
        """
        Inputs:
            N_iter - number of iterations that will be used in training the u 
                     neural network
            
        Trains the u neural network with the class's training data. 
        """
        tf_dict = {self.x_tf: self.x_u_train, self.v_tf: self.v_u_train, self.u_tf: self.u}
        print("Training u NN")
        loss_value = self.sess.run(self.u_loss, tf_dict)
        print("Loss before training: %f" % (loss_value))
        for iter in range(N_iter + 1):
            self.sess.run(self.idn_u_train_op_Adam, tf_dict)
        self.idn_u_optimizer.minimize(self.sess,
                                      feed_dict = tf_dict,
                                      fetches = [self.u_loss])
        loss_value = self.sess.run(self.u_loss, tf_dict)
        print("Loss after training: %f" % (loss_value))
    
    def train_f_NN(self, N_iter):
        """
        Inputs:
            N_iter - number of iterations that will be used in training the f 
                     neural network
            
        Trains the f neural network with the class's training data. 
        """
        tf_dict = {self.x_tf: self.x_f_train, self.v_tf: self.v_f_train, self.f_tf: self.f_f_train}
        print("Training f NN")
        loss_value = self.sess.run(self.f_loss, tf_dict)
        print("Loss before training: %f" % (loss_value))
        for iter in range(N_iter + 1):
            self.sess.run(self.idn_f_train_op_Adam, tf_dict)
        self.idn_f_optimizer.minimize(self.sess,
                                      feed_dict = tf_dict,
                                      fetches = [self.f_loss])
        loss_value = self.sess.run(self.f_loss, tf_dict)
        print("Loss after training: %f" % (loss_value))
        
    def train_sol_NN(self, N_iter, use_prev):
        """
        Inputs:
            N_iter - number of iterations that will be used in training the f 
                     neural network
            use_prev - boolean indicating whether the weights and biases generated
                       in the training of the u neural network should be used
                       as starting points in the training of the solution neural
                       network (structure of solution and u neural network must
                       be the same)
            
        Trains the solution neural network with the class's training data and 
        the f neural network. 
        """
        print("Training solution NN")
        tf_dict = {self.x_tf: self.x_u_train, self.u_tf: self.u, self.v_tf: self.v_u_train,
                   self.f_tf: self.f_u_train}
        
        if use_prev:
            self.sol_weights = self.u_weights
            self.sel_biases = self.u_biases
        
        loss_value = self.sess.run(self.sol_loss, tf_dict)
        print("Loss before training: %f" % (loss_value))
        
        for iter in range(N_iter + 1):
            self.sess.run(self.sol_train_op_Adam, tf_dict)  
        self.sol_optimizer.minimize(self.sess,
                                    feed_dict = tf_dict,
                                    fetches = [self.sol_loss])
        loss_value = self.sess.run(self.sol_loss, tf_dict)
        print("Loss after training: %f" % (loss_value))
            
    def predict_sol(self, x_pred, v_pred):
        """
        Inputs:
            x_pred - x coordinates
            v_pred - viscosity value
            
        Predicts u(x_pred) given the viscosity v_pred using the solution neural
        network.
        """
        num_mesh = np.size(x_pred)
        x = np.transpose(np.asarray([x_pred]))
        v = np.transpose(np.asarray([[v_pred]*num_mesh]))
        return self.sess.run(self.sol_u_predict, {self.x_tf : x, self.v_tf: v})
    
    def predict_u(self, x_pred, v_pred):
        """
        Inputs:
            x_pred - x coordinates
            v_pred - viscosity value
            
        Predicts u(x_pred) given the viscosity v_pred using the u neural
        network.
        """
        num_mesh = np.size(x_pred)
        x = np.transpose(np.asarray([x_pred]))
        v = np.transpose(np.asarray([[v_pred]*num_mesh]))
        return self.sess.run(self.idn_u_predict, {self.x_tf : x, self.v_tf: v})
    
    def predict_sol_f(self, x_pred, v_pred):
        """
        Inputs:
            x_pred - x coordinates
            v_pred - viscosity value
            
        Predicts u * u_x - v_pred * u_xx at the given x coordinate using the f 
        neural network with u's generated from the solution neural network.
        """
        num_mesh = np.size(x_pred)
        x = np.transpose(np.asarray([x_pred]))
        v = np.transpose(np.asarray([[v_pred]*num_mesh]))
        return self.sess.run(self.sol_f_predict, {self.x_tf : x, self.v_tf: v})
    
    def predict_idn_f(self, x_pred, v_pred):
        """
        Inputs:
            x_pred - x coordinates
            v_pred - viscosity value
            
        Predicts u * u_x - v_pred * u_xx at the given x coordinate using the f 
        neural network with u's generated from the u neural network.
        """
        num_mesh = np.size(x_pred)
        x = np.transpose(np.asarray([x_pred]))
        v = np.transpose(np.asarray([[v_pred]*num_mesh]))
        return self.sess.run(self.idn_f_predict, {self.x_tf : x, self.v_tf: v})

        
def train_and_save_model(viscosities, u, f, num_mesh_train, u_layers, f_layers, 
                         sol_layers, maxiter, filename):
    """
    Inputs:
        viscosities - training data's viscosities
        u - training data's u(x)
        f - force vector
        num_mesh_train - number of mesh points to train the u and solution
                         neural network with (excludes the points at 0 and 1)
        u_layers - array of the number of nodes in the u neural network
                   (must start with 2 and end with 1)
        f_layers - array of the number of nodes in the f neural network
                   (must start with 3 and end with 1)
        layers - array of number of nodes in the solution neural network
                 (must start with 2 and end with 1)
        maxiter - maximum number of iterations that the neural networks
                  will be trained with
        filename - name of the file to save the NN
                         
    Initializes a NN object with the given inputs and trains all three of its 
    neural networks. Saves the NN object to the given file.
    """
    model = NN(viscosities, u, f, num_mesh_train, u_layers, f_layers, sol_layers, maxiter)
    model.train_u_NN(0)
    model.train_f_NN(0)
    model.train_sol_NN(0, use_prev)
    model.save(filename)
    
def restore_model(viscosities, u, f, num_mesh_train, u_layers, f_layers, sol_layers, maxiter, filename):
    """
    Inputs:
        viscosities - training data's viscosities
        u - training data's u(x)
        f - force vector
        num_mesh_train - number of mesh points to train the u and solution
                         neural network with (excludes the points at 0 and 1)
        u_layers - array of the number of nodes in the u neural network
                   (must start with 2 and end with 1)
        f_layers - array of the number of nodes in the f neural network
                   (must start with 3 and end with 1)
        layers - array of number of nodes in the solution neural network
                 (must start with 2 and end with 1)
        maxiter - maximum number of iterations that the neural networks
                  will be trained with
        filename - name of the file to save the NN
                         
    Initializes a NN object with the given inputs and loads each of the neural
    networks weights and biases from the given file.
    """
    model = NN(viscosities, u, f, num_mesh_train, u_layers, f_layers, sol_layers, maxiter)
    model.restore(filename)
    return model
        
def compare_NNs(model, viscosity, filename):
    """
    Inputs:
        model - NN object
        viscosity - value for viscosity
        filename - file to retrieve the solutions for u(x)
        
    Compares the solution and u neural network by plotting u(x) generated by
    the solution neural network and the u neural network against each other as 
    well as the solution to u(x). Also plots the predicted force vectors 
    generated by f neural network with u values comping from the solution
    neural network and the u neural network. 
    """
    database = np.load(filename)
    viscosities = database['viscosities']
    idx = (np.abs(viscosities - viscosity)).argmin()
    u_solved = database['u'][idx]
    nearest_visc = viscosities[idx]
    sol_label = "Solution with viscosity = " + str(nearest_visc)
    
    x = np.linspace(0, 1, 300)
    u = model.predict_u(x, viscosity)
    sol = model.predict_sol(x, viscosity)
    plt.plot(x, u, label = "U NN")
    plt.plot(x, sol, label = "Sol NN")
    plt.plot(x, u_solved, label = sol_label)
    plt.legend(loc = "upper left")
    plt.show()
    
    f_idn = model.predict_sol_f(x, viscosity)
    f_sol = model.predict_idn_f(x, viscosity)
    plt.plot(x, f_idn, label = "f NN with U NN")
    plt.plot(x, f_sol, label = "f NN with Sol NN")
    plt.plot(x, database['f'], label = "Actual f")
    plt.legend(loc = "upper left")
    plt.show()
    
        
# generate_all_training(500, 0.01, 1, 1e-5, F, FILENAME)
 
u_layers = [2, 100, 1]
sol_layers = [2, 100, 1]
f_layers = [4, 100, 1]
maxiter = 1000
num_mesh_train = 5
u, viscosities, f = choose_training(50, 0.01, 1, FILENAME)

#train_and_save_model(viscosities, u, f, num_mesh_train, u_layers, f_layers, 
#                     sol_layers, maxiter, "./saved_models/experiment1")

model = restore_model(viscosities, u, f, num_mesh_train, u_layers, f_layers, 
                    sol_layers, maxiter, "./saved_models/experiment1")
compare_NNs(model, 0.05, FILENAME)

    
    
    

