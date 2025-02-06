#include "header/perceptron.hpp"

Perceptron::Perceptron(std::vector<double> weights, double bias, double learningRate)
    : weights(weights), bias(bias), learningRate(learningRate) {}

double Perceptron::predict(const std::vector<double>& x) const 
{
    // Dot prodcut for an array of size 2  
    double dot_product = bias;
    for (int i = 0; i < weights.size(); i++) {
        dot_product += weights[i] * x[i];
    }
    // double dot_product = weights[0] * x[0] + weights[1] * x[1] + bias;
    return dot_product >= 0 ? 1 : 0;
}

void Perceptron::train(const std::vector<std::vector<double>>& x, const std::vector<double>& targets, int epochs) 
{
    // Train the perceptron
    // ensure both arrays are the same size
    if (x.size() != targets.size()) return;

    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < x.size(); i++) {
            double pred = predict(x[i]);
            double error = targets[i] - pred;
            // Update each weight based on the input value
            for (int j = 0; j < weights.size(); j++) {
                weights[j] += learningRate * error * x[i][j];
            }
            // Update bias 
            bias += learningRate * error;
        }
    }
}

void Perceptron::__str__(int verbose) 
{
    // Printing the weights 
    std::cout << "weights\n";
    for (auto i : weights)
        std::cout << i << " ";
    std::cout << "\n";

    // Other info 
    if (verbose >= 1) {
        std::cout << "\nbias = " << bias << "\n";
        std::cout << "Learning rate = " << learningRate << std::endl;
    }
}

SimplePerceptron::SimplePerceptron(const std::vector<double>& weights, double bias) 
    : weights(weights), bias(bias) {}


int SimplePerceptron::predict(const std::vector<double>& inputs) const {
    double total = 0.0;
    // Calculate weighted sum.
    for (size_t i = 0; i < inputs.size() && i < weights.size(); ++i) {
        total += inputs[i] * weights[i];
    }
    total += bias;
    // Activation: return 1 if total is non-negative, otherwise return 0.
    return (total >= 0) ? 1 : 0;
}
