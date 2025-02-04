#include <iostream>
#include "header/perceptron.hpp"
#include <vector>

int main()
{
    // Create a Perceptron object with initial weights, bias, and learning rate using uniform initialization syntax
    Perceptron p({0.3, 0.3}, 0.5, 0.1);

    // Example usage of the Perceptron object
    // std::vector<double> inputs{1.0, 2.0};
    // double result = p.output(inputs);

    // std::cout << "Perceptron output: " << result << std::endl;

    // Print the Perceptron's internal state
    p.__str__();
    
    return 0;
}