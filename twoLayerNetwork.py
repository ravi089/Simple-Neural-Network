# Simple 2 layer neural network.
from numpy import exp, array, random, dot, tanh

class NeuronLayer():
    # Weight matrix for a layer.
    def __init__(self, num_neurons, num_inputs_per_neuron):
        self.weight_matrix = 2 * random.random((num_inputs_per_neuron, num_neurons)) - 1

class NeuralNetwork():
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    # tanh as activation fucntion.
    def tanh(self, x):
        return tanh(x)

    # derivative of tanh function.
    def tanh_derivative(self, x):
        return 1.0 - tanh(x) ** 2

    # The neural network forward propagation.
    def forward_propagation(self, inputs):
        output_layer1 = self.tanh(dot(inputs, self.layer1.weight_matrix))
        output_layer2 = self.tanh(dot(output_layer1, self.layer2.weight_matrix))
        return output_layer1, output_layer2
    
    def train(self, train_inputs, train_outputs, num_train_iterations):
        for iteration in range(num_train_iterations):
            output_layer1, output_layer2 = self.forward_propagation(train_inputs)
            # Calculate error for layer 2.
            error_layer2 = train_outputs - output_layer2
            delta_layer2 = error_layer2 * self.tanh_derivative(output_layer2)

            # Calculate error for layer 1.
            error_layer1 = delta_layer2.dot(self.layer2.weight_matrix.T)
            delta_layer1 = error_layer1 * self.tanh_derivative(output_layer1)

            # Calculate weights adjustment.
            adjust_layer1 = train_inputs.T.dot(delta_layer1)
            adjust_layer2 = output_layer1.T.dot(delta_layer2)

            # Adjust the weights.
            self.layer1.weight_matrix += adjust_layer1
            self.layer2.weight_matrix += adjust_layer2

    def print_weights(self):
        print ('->Layer 1 weights')
        print (self.layer1.weight_matrix)
        print ('->Layer 2 weights')
        print (self.layer2.weight_matrix)

if __name__ == '__main__':
    random.seed(1)
    
    # Layer 1 (4 neurons each 3 inputs each)
    layer1 = NeuronLayer(4,3)
    
    # Layer 2 (single neuron with 4 inputs)
    layer2 = NeuronLayer(1,4)

    neural_network = NeuralNetwork(layer1, layer2)

    print ('Random weights at the start of training.')
    neural_network.print_weights()

    # Training inputs/outputs.
    train_inputs = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    train_outputs = array([[0, 1, 1, 1, 1, 0, 0]]).T

    neural_network.train(train_inputs, train_outputs, 60000)

    print ('New weights after training.')
    neural_network.print_weights()

    print ('Testing Neural network with new example.')
    hidden_layer, output = neural_network.forward_propagation(array([1, 1, 0]))
    print (output)
