import os
import numpy as np
import matplotlib.pyplot as plt
from math import exp
from random import seed, random

# Function to load data from a text file
def load_data(filename):
    script_dir = os.path.dirname(__file__)  # Get the script's directory
    file_path = os.path.join(script_dir, filename)

    with open(file_path, 'r') as file:
        dataset = [line.strip().split() for line in file]
        dataset = [[float(x) for x in row] for row in dataset]

    return dataset

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for _ in range(n_inputs + 1)]} for _ in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for _ in range(n_hidden + 1)]} for _ in range(n_outputs)]
    network.append(output_layer)
    return network

# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation

# Transfer neuron activation using the sigmoid function
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

# Calculate the derivative of a neuron output with respect to the sigmoid function
def transfer_derivative(output):
    return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(neuron['output'] - expected[j])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] -= l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for _ in range(n_outputs)]
            expected[int(row[-1])] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)

        # Print epoch information
        if epoch % 10 == 0:  # Print every 5 epochs, adjust as needed
            print(f'>epoch={epoch}, lrate={l_rate:.3f}, error={sum_error:.3f}')

    # Print final weights and output for each neuron
    for layer in network:
        print(layer)

# Function to visualize decision boundary for a dataset
def plot_decision_boundary(network, dataset, title):
    min_x, max_x = min([row[0] for row in dataset]), max([row[0] for row in dataset])
    min_y, max_y = min([row[1] for row in dataset]), max([row[1] for row in dataset])
    xx, yy = np.meshgrid(np.arange(min_x, max_x, 0.1),
                         np.arange(min_y, max_y, 0.1))

    flat_grid = np.c_[xx.ravel(), yy.ravel()]
    predictions = np.array([round(forward_propagate(network, row)[0]) for row in flat_grid])

    predictions = predictions.reshape(xx.shape)

    plt.contourf(xx, yy, predictions, cmap=plt.cm.Spectral, alpha=0.8)
    scatter = plt.scatter([row[0] for row in dataset], [row[1] for row in dataset], c=[int(row[2]) for row in dataset], cmap=plt.cm.Spectral)
    plt.title(title)

    # Add legend
    plt.legend(*scatter.legend_elements(), title="Classes", loc="upper right")

    plt.show()

# Function to calculate accuracy on a dataset
def calculate_accuracy(network, dataset):
    correct_predictions = 0
    for row in dataset:
        outputs = forward_propagate(network, row)
        predicted_class = np.argmax(outputs)
        if predicted_class == int(row[-1]):
            correct_predictions += 1
    accuracy = correct_predictions / len(dataset)
    return accuracy

# Seed for reproducibility
seed(1)

# Load training datasets
dataset_a = load_data('dataset1_training.txt')
dataset_b = load_data('dataset2_training.txt')

# Set parameters
n_inputs = 2
n_outputs = 2
learning_rate = 0.8
num_epochs = 100

# Initialize and train network on dataset_a
print("Training and testing on dataset_a:")
network_a = initialize_network(n_inputs, 4, n_outputs)
train_network(network_a, dataset_a, learning_rate, num_epochs, n_outputs)
test_accuracy_a = calculate_accuracy(network_a, load_data('dataset1_testing.txt'))
print(f"Accuracy on testing dataset a: {test_accuracy_a:.2%}")

# Visualize decision boundary for dataset_a
plot_decision_boundary(network_a, dataset_a, "Decision Boundary for Training Dataset A")

# Initialize and train network on dataset_b
print("\nTraining and testing on dataset_b:")
network_b = initialize_network(n_inputs, 4, n_outputs)
train_network(network_b, dataset_b, learning_rate, num_epochs, n_outputs)
test_accuracy_b = calculate_accuracy(network_b, load_data('dataset2_testing.txt'))
print(f"Accuracy on testing dataset b: {test_accuracy_b:.2%}")

# Visualize decision boundary for dataset_b
plot_decision_boundary(network_b, dataset_b, "Decision Boundary for Training Dataset B")
