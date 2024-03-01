import mnist_loader
import mnist_matrix
import mnist_forloop

class DriverMnistMatrixVsForLoop:
    training_data, validation_data, test_data, training_inputs, training_results = mnist_loader.load_data_wrapper()
    # Contains a 30 neuron hidden layer using matrixed code
    matrix_net = mnist_matrix.FeedForwardNetwork([784, 30, 10])
    # Contains a 30 neuron hidden layer using code with for loops
    forloop_net = mnist_forloop.FeedForwardNetwork([784, 30, 10])
    
    
    matrix_net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
    forloop_net.SGD(training_data, 30, 10, 3.0, test_data=test_data)