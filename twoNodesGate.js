const training_set = [[1, 0], [0, 1], [0, 0], [1, 1]];
// const labels = [1, 1, 0, 0];
const labels = [0, 0, 1, 1];

// Sigmoid function and its derivative
function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function sigmoidDerivative(x) {
    return x * (1 - x); // Derivative of sigmoid
}

class Node {
    constructor(total_inputs) {
        this.weights = Array(total_inputs).fill().map(() => Math.random());
        this.bias = Math.random();
    }

    forward(inputs) {
        this.inputs = inputs; // Store inputs for backpropagation
        this.sum = this.bias;
        for (let i = 0; i < this.weights.length; i++) {
            this.sum += inputs[i] * this.weights[i];
        }
        this.output = sigmoid(this.sum); // Using sigmoid activation
        return this.output;
    }

    // Update the weights and bias based on the error
    update_weights(error) {
        const delta = error * sigmoidDerivative(this.output); // Derivative of sigmoid
        for (let i = 0; i < this.weights.length; i++) {
            this.weights[i] += delta * this.inputs[i]; // Adjust weights
        }
        this.bias += delta; // Adjust bias
    }

    show() {
        console.log("Weights:", this.weights, "Bias:", this.bias);
    }
}

const output_node = new Node(2);
const hidden_node_1 = new Node(2);
const hidden_node_2 = new Node(2);

// Training loop
for (let epoch = 0; epoch < 100000; epoch++) {
    for (let j = 0; j < training_set.length; j++) {
        const inputs = training_set[j];
        const label = labels[j];

        // Forward pass
        const h1 = hidden_node_1.forward(inputs);
        const h2 = hidden_node_2.forward(inputs);
        const output = output_node.forward([h1, h2]);

        const output_error = label - output;

        // Backpropagate error to hidden nodes
        const hidden_error_1 = output_error * output_node.weights[0];
        const hidden_error_2 = output_error * output_node.weights[1];

        // Update output node weights and bias
        output_node.update_weights(output_error);

        // Update hidden nodes' weights and biases
        hidden_node_1.update_weights(hidden_error_1);
        hidden_node_2.update_weights(hidden_error_2);
    }
}

// Show final weights and biases
console.log('Trained output node:');
output_node.show();
console.log('Trained hidden node 1:');
hidden_node_1.show();
console.log('Trained hidden node 2:');
hidden_node_2.show();

// Testing the trained model
console.log('Testing the trained network:');
for (let i = 0; i < training_set.length; i++) {
    const h1 = hidden_node_1.forward(training_set[i]);
    const h2 = hidden_node_2.forward(training_set[i]);
    const output = output_node.forward([h1, h2]);
    console.log(`Input: ${training_set[i]}, Output: ${Math.round(output)}`);
}
