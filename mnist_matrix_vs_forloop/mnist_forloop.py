'''
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning algorithm for a feedforward neural network. 
Gradients are calculated using backpropagation. Note that I have focused on making the code simple, easily
readable, and easily modifiable. It is not optimized, and omits many desirable features.
'''

#### Libraries
# Standard Library
import random
import time
random.seed(1) # for reproducibility

# Third-Party Libraries
import numpy as np

class FeedForwardNetwork:
    def __init__(self, sizes):
        '''
        The list ```sizes``` contains the number of neurons in the respective layers of the network.
        For example, if the list was [2,3,1] then it would be a three-layer network,
        with the first layer containing 2 neurons, the second layer 3 neurons, and the third layer
        1 neuron. The biases and weights for the network are initialized randomly, using a 
        Gaussian distribution with mean 0, and variance 1. Note that the first layer is assumed
        to be an input layer, and by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later layers.
        '''
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                           for x, y in zip(sizes[:-1], sizes[1:])]
        
        
    def feedforward(self, a):
        '''Return the output of the network if "a" is input.'''
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        """
        Train the neural network using mini-batch stochastic gradient descent. 
        The 'training_data' is a list of tuples '(x,y)' representing the training inputs
        and the desired outputs. The other non-optional parameters are self explanatory. If 'test_data' os provided
        then the network will be evaluated against the test data after each epoch, and partial progress printed out.
        The evaluating on test_data + printing is useful for tracking progress, but slows things down substantially"""
        if test_data: 
            n_test = len(test_data)
        n = len(training_data)
        print('Start for looped version of updating mini batches')
        tic = time.perf_counter()
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate) # Use this for looping method
#                self.matrix_update_mini_batch(mini_batch, learning_rate) # Use this for matrix calculation method
            if test_data:
                correct_results = self.evaluate(test_data)
                percentage = correct_results/n_test
                print("Epoch {0}: {1} / {2} ---> {3}".format(
                    j, self.evaluate(test_data), n_test, percentage))
            else:
                print("Epoch {0} complete".format(j))
        toc = time.perf_counter()
        print(f"Finished {epochs} iterations in {toc - tic:0.4f} seconds")
        
    def update_mini_batch(self, mini_batch, learning_rate):
        '''
        Update the network's weights and biases by applying gradient descent using backprop
        to a single mini batch.
        The "mini_batch" is a list of tuples "(x,y)"
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            self.weights = [w-(learning_rate/len(mini_batch)) * nw
                               for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b-(learning_rate/len(mini_batch)) * nb
                            for b, nb in zip(self.biases, nabla_b)]
            
    def matrix_update_mini_batch(self, mini_batch, learning_rate):
        '''
        Update the network's weights and biases by applying gradient descent using backprop
        to a single mini batch.
        The "mini_batch" is a list of tuples "(x,y)"
        '''
        xs, ys = zip(*mini_batch)
        xs = np.array(xs)
        ys = np.array(ys)
        delta_b, delta_w = self.matrix_backprop(xs, ys)
        
        self.weights = [w-(learning_rate/len(mini_batch)) * nw
                               for w, nw in zip(self.weights, delta_w)]
        self.biases = [b-(learning_rate/len(mini_batch)) * nb
                              for b, nb in zip(self.biases, delta_b)]
                              
    def backprop(self, x, y):
        '''
        This does not get explained in chapter 1.
        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x. ``nabla_b`` and 
        ``nabla_w`` are layer-by-layer lists of numpy arrays similar to
        ``self.biases`` and ``self.weights``.
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]
        zs = [] # list to store all of the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # Backwards pass
        delta = sigmoid_prime(zs[-1]) * \
                    self.cost_derivative(activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book. here,
        # l = 1 means the last layer of neurons, l = 2 is the 
        # second-last layer, and so on. It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers): 
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    def matrix_backprop(self, xs, ys):
        '''
        This does not get explained in chapter 1.
        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x. ``nabla_b`` and 
        ``nabla_w`` are layer-by-layer lists of numpy arrays similar to
        ``self.biases`` and ``self.weights``.
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = xs.view().reshape(10, -1).T
        activations = [activation]
        zs = [] # list to store all of the z vectors, layer by layer
    
        z1 = np.dot(self.weights[0], activation) # (30, 784) * (784,10)
        z1 += self.biases[0] # (30,10) + (30,1)
        zs.append(z1)
        activation = sigmoid(z1)
        activations.append(activation)
        z2 = np.dot(self.weights[1], activation) # (10,30) * (30,10)
        z2 += self.biases[1] # (10,10) + (10,1)
        zs.append(z2)
        activation = sigmoid(z2) # (10,10) output
        activations.append(activation)
        # Backwards pass
        # memorize code and write this from memory once you feel like you understand it
        reshaped_ys = ys.view().reshape(10,10).T
        delta = sigmoid_prime(zs[-1]) * \
                    self.matrix_cost_derivative(activations[-1], reshaped_ys)
        nabla_b[-1] = delta.sum(axis=1,keepdims=True)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book. here,
        # l = 1 means the last layer of neurons, l = 2 is the 
        # second-last layer, and so on. It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers): 
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta.sum(axis=1,keepdims=True)
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    def evaluate(self, test_data):
        '''
        Return the number of test inputs for which the neural network outputs
        the correct result. Note that the network's output is the index
        of the neuron in the final layer with the highest activation
        '''
        test_results = [(np.argmax(self.feedforward(x)), y)
                           for (x,y) in test_data]
        return sum(int(x == y) for (x,y) in test_results)
        
    def cost_derivative(self, output_activations, y):
        '''
        Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations
        '''
        return (output_activations-y)
    
    def matrix_cost_derivative(self, output_activations, ys):
        '''
        Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations
        '''            
        return (output_activations-ys)

# Miscellaneous functions
def sigmoid(z):
    '''
    The sigmoid function
    '''
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    '''
    Derivative of the sigmoid function
    '''
    return sigmoid(z)*(1-sigmoid(z))
    
        