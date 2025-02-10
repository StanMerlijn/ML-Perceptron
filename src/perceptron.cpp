#include "header/perceptron.hpp"

Perceptron::Perceptron(std::vector<double> weights, double bias, double learningRate)
    : weights(weights), bias(bias), learningRate(learningRate) {}

int Perceptron::predict(const std::vector<int>& inputs) const 
{
    // Dot prodcut for an array of size 2  
    double dot_product = bias;
    for (int i = 0; i < weights.size(); i++) {
        dot_product += weights[i] * inputs[i];
    }
    // double dot_product = weights[0] * inputs[0] + weights[1] * inputs[1] + bias;
    return dot_product >= 0 ? 1 : 0;
}

void Perceptron::train(const std::vector<std::vector<int>>& inputs, const std::vector<int>& targets, int epochs) 
{
    // Train the perceptron
    // ensure both arrays are the same size
    if (inputs.size() != targets.size()) return;

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