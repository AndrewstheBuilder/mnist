import mnist_loader

class DriverMnistMatrixVsForLoop:
    training_data, validation_data, test_data, training_inputs, training_results = mnist_loader.load_data_wrapper()