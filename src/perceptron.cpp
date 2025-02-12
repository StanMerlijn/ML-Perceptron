#include "header/perceptron.hpp"

Perceptron::Perceptron(std::vector<double> weights, double bias, double learningRate)
    : weights(weights), bias(bias), learningRate(learningRate) {}

int Perceptron::predict(const std::vector<int>& inputs) const 
{
    // Dot prodcut for the perceptron  
    double dot_product = bias;
    for (int i = 0; i < weights.size(); i++) {
        dot_product += weights[i] * inputs[i];
    }
    // Threshold function 
    return dot_product >= 0 ? 1 : 0;
}

void Perceptron::train(const std::vector<std::vector<int>>& inputs, const std::vector<int>& targets, int epochs) 
{
    // ensure both arrays are the same size
    if (inputs.size() != targets.size()) return;

    // Train the perceptron
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < inputs.size(); i++) {
            double pred = predict(inputs[i]);
            double error = targets[i] - pred;
            // Update each weight based on the input value
            for (int j = 0; j < weights.size(); j++) {
                weights[j] += learningRate * error * inputs[i][j];
            }
            // Update bias 
            bias += learningRate * error;
        }
    }
}

void Perceptron::__str__(int verbose) const
{
    // Printing the weights 
    std::cout << "weights for perceptron:\n";
    for (auto i : weights)
        std::cout << i << " ";

    // Other info 
    if (verbose >= 1) {
        std::cout << "\nbias = " << bias << "\n";
        std::cout << "Learning rate = " << learningRate << std::endl;
    }
}