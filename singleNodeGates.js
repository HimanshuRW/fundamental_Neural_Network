const training_set = [[1,0],[0,1],[0,0],[1,1]];

// OR GATE 
// const labels = [1,1,0,1];
// keep the weights positive and bias = 0 or positive

// AND Gate
// const labels = [0,0,0,1];
// bias = negetive
// keep the weights positive
// bias < |w1 + w2| for 11
// bias > w1, w2 for 01 or 10

// Nand Gate
// const labels = [1,1,1,0]; 
// bias = positive for 0 0
// keep for w negetive
// bias > w1, w2 for 01 or 10
// bias < |w1+ w2| for 11


// Nor Gate
// const labels = [0,0,1,0]; 
// make the and keep the weights negetive and bias = - wight1 - weight2

// XOR Gate -> 1 on different , 0 on same
// const labels = [1,1,0,0]; 
// 00 -> 0 (bias = 0 or negetive)
// 11 -> 0 (w1 + w2 < bias)
// 01 , 10 -> 1 (w1,w2 > bias) 
// NOT POSSIBLE WITH SINGLE NODE

// XNOR Gate -> 1 on same , 0 on different
const labels = [0,0,1,1];
// 00 -> 1 (bias = positive)
// let w1,w2 negetive
// 11 -> 1 (bias > |w1| + |w2|)
// 01 , 10 -> 0 (|w1|, |w2| > bias)
// NOT POSSIBLE WITH SINGLE NODE


class Node {
    constructor(total_inputs) {
        this.weights = Array(total_inputs).fill().map(() => Math.random());
        this.bias = Math.random();
    }

    forward(inputs) {
        let sum = this.bias;
        for (let i = 0; i < this.weights.length; i++) {
            sum += inputs[i] * this.weights[i];
        }
        return sum > 0 ? 1 : 0;
    }

    show() {
        console.log(this.weights, this.bias);
    }
}

let outputNode = new Node(2);

// train
for (let i = 0; i < 10000000; i++) {
    for (let j = 0; j < training_set.length; j++) {
        const prediction = outputNode.forward(training_set[j]);
        const error = labels[j] - prediction;
        for (let k = 0; k < outputNode.weights.length; k++) {
            outputNode.weights[k] += error * training_set[j][k];
        }
        outputNode.bias += error;
    }
}
outputNode.show();
for (const training_data of training_set) {
    const prediction = outputNode.forward(training_data);
    console.log(training_data, prediction);
}