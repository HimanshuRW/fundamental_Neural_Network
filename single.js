const training_set = [[1, 0], [0, 1], [0, 0], [1, 1]];  // Inputs

// OR GATE: Expected outputs
const labels = [1, 1, 0, 1];
const learning_rate = 0.001;

class Node {
    constructor(total_inputs) {
        this.weights = Array(total_inputs).fill().map(() => Math.random());
        this.bias = Math.random();
    }

    // Linear forward pass without sigmoid
    forward(inputs) {
        this.inputs = inputs;
        this.sum = this.bias;
        for (let i = 0; i < this.weights.length; i++) {
            this.sum += inputs[i] * this.weights[i];
        }
        return this.sum;  // No activation function, just the weighted sum
    }

    // Update weights using the error
    update_weights(error) {
        for (let i = 0; i < this.weights.length; i++) {
            let direction = this.inputs[i] > 0 ? 1 : -1;
            // Weight update is proportional to the error and input
            this.weights[i] += learning_rate * error * direction;
        }
        // Bias update
        this.bias += 1 * learning_rate * error;
    }
}

const output_node = new Node(2);

// Train the model
for (let i = 0; i < 100000; i++) {
    for (let j = 0; j < training_set.length; j++) {
        const inputs = training_set[j];
        const label = labels[j];
        const prediction = output_node.forward(inputs);
        const error = label - prediction;
        output_node.update_weights(error);
    }
}

// Test the model
for (let i = 0; i < training_set.length; i++) {
    const inputs = training_set[i];
    const prediction = output_node.forward(inputs);
    console.log(inputs, prediction);
}
