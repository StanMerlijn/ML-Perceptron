#include "header/perceptronLayer.hpp"
#include <cstdlib>   // For rand()
#include <ctime>     // For time()

PerceptronLayer::PerceptronLayer(int numNeurons, int inputDimension, double learningRate) 
{
    // Initialize random seed
    std::srand(static_cast<unsigned>(std::time(0)));

    for (int i = 0; i < numNeurons; i++) {
        // Initialize weights randomly between -0.5 and 0.5
        std::vector<double> weights(inputDimension);
        for (int j = 0; j < inputDimension; j++) {
            weights[j] = (double)std::rand() / RAND_MAX - 0.5;
        }
        // initialize bias randomly
        double bias = (double)std::rand() / RAND_MAX - 0.5;
        neurons.emplace_back(weights, bias, learningRate);
    }
}

std::vector<int> PerceptronLayer::feed_forward(const std::vector<int>& input) const 
{
    // Predict the output for each perceptron 
    std::vector<int> outputs;
    for (const Perceptron& neuron : neurons) {
        outputs.push_back(neuron.predict(input));
    }
    return outputs;
}

void PerceptronLayer::train(const std::vector<std::vector<int>>& inputs, 
    const std::vector<std::vector<int>>& targets, 
    int epochs)
{
        // Training each perceptron in the layer
        // Inputs are the same for all perceptrons 2 inputs
        for (int i = 0; i < neurons.size(); i++) {   
            neurons[i].train(inputs, targets[i], epochs);
        }
}

void PerceptronLayer::__str__(int verbose) const
{
    // For each neuron in the layer print the data
    for (const Perceptron& neuron : neurons) {
        neuron.__str__(verbose);
    }
}