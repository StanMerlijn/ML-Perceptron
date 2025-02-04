#include "header/perceptron.hpp"

Perceptron::Perceptron(std::vector<double> weights, double bias, double learningRate)
    : weights(weights), bias(bias), learningRate(learningRate) {}

Perceptron::~Perceptron() {}

double Perceptron::output(std::vector<double> inputs)
{
    double sum = 0.0f;
    double threshold = -1 * bias;
    for (int i = 0; i < inputs.size(); i++)
    {
        sum += inputs[i] * weights[i];
    }
    
    return sum >= threshold ? 1 : 0;
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