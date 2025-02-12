#include "header/perceptronLayer.hpp"

PerceptronLayer::PerceptronLayer(const std::vector<Perceptron>& neurons) 
    : neurons(neurons) {}

std::vector<int> PerceptronLayer::feedForward(const std::vector<int>& input) const 
{
    // Predict the output for each perceptron 
    std::vector<int> outputs;
    // Propagate the input through each layer sequentially.
    for (const Perceptron& neuron : neurons) {
        outputs.push_back(neuron.predict(input));
    }
    return outputs;
}

void PerceptronLayer::__str__(int verbose) const
{
    // For each neuron in the layer print the data
    for (const Perceptron& neuron : neurons) {
        neuron.__str__(verbose);
    }
}