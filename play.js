const training_set = [[1,0],[0,1],[0,0],[1,1]];
const labels = [1, 1, -1, -1];

// Tanh function as activation
function tanh(x) {
    return Math.tanh(x); // JavaScript has built-in tanh function
}

// Derivative of tanh function
function tanh_derivative(x) {
    return 1 - Math.pow(tanh(x), 2); // Derivative: 1 - tanh^2(x)
}

class Node {
    constructor(total_inputs) {
        // Initialize weights and bias with random values between -1 and 1
        this.weights = Array(total_inputs).fill().map(() => Math.random() * 2 - 1);
        this.bias = Math.random() * 2 - 1;
    }

    // Forward pass through the node
    forward(inputs) {
        this.inputs = inputs;
        this.sum = this.bias;
        for (let i = 0; i < this.weights.length; i++) {
            this.sum += inputs[i] * this.weights[i];
        }
        this.output = tanh(this.sum);
        return this.output; // Apply tanh activation
    }

    // Update the weights and bias based on the error and learning rate
    update_weights(error, learning_rate) {
        const delta = error * tanh_derivative(this.sum); // Derivative of tanh
        for (let i = 0; i < this.weights.length; i++) {
            this.weights[i] += learning_rate * delta * this.inputs[i]; // Adjust weights
        }
        this.bias += learning_rate * delta; // Adjust bias
    }

    // Debugging helper to view weights and biases
    show() {
        console.log("Weights:", this.weights, "Bias:", this.bias);
    }
}

// Initialize network with two hidden nodes and one output node
const output_node = new Node(2);
const hidden_node_1 = new Node(2);
const hidden_node_2 = new Node(2);

// Training loop
const learning_rate = 0.01; // Learning rate for smooth weight updates

for (let epoch = 0; epoch < 1000000; epoch++) {
    for (let j = 0; j < training_set.length; j++) {
        const inputs = training_set[j];
        const label = labels[j];

        // Forward pass
        const h1 = hidden_node_1.forward(inputs);
        const h2 = hidden_node_2.forward(inputs);
        const output = output_node.forward([h1, h2]);

        // Calculate the output error
        const output_error = label - output;

        // Update output node weights and bias
        output_node.update_weights(output_error, learning_rate);

        // Backpropagate the error to hidden nodes
        const hidden_error_1 = output_error * output_node.weights[0];
        const hidden_error_2 = output_error * output_node.weights[1];

        // Update hidden node weights and biases
        hidden_node_1.update_weights(hidden_error_1, learning_rate);
        hidden_node_2.update_weights(hidden_error_2, learning_rate);
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
    console.log(`Input: ${training_set[i]}, Output: ${output}`);
}
