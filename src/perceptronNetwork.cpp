/**
 * @file perceptronNetwork.cpp
 * @author Stan Merlijn
 * @brief Implementation of the PerceptronNetwork class
 * @version 0.1
 * @date 2025-02-12
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#include "header/perceptronNetwork.hpp"
#include "header/perceptronLayer.hpp" 
#include <iostream>

PerceptronNetwork::PerceptronNetwork(std::vector<PerceptronLayer> layers) 
    : layers(layers) {}

std::vector<int> PerceptronNetwork::feedForward(const std::vector<int>& input) const
{
    std::vector<int> activation = input;
    // Propagate the input through each layer sequentially. Also called feedforward.
    for (const PerceptronLayer& layer : layers) {
        activation = layer.feedForward(activation);
    }
    return activation;
}

void PerceptronNetwork::__str__(int verbose) const
{  
    // Print the network structure
    std::cout << "Perceptron Network Structure:" << std::endl;
    std::cout << "Number of layers: " << layers.size() << std::endl;
    // For each layer in the network print the data
    for (int i = 0; i < layers.size(); ++i)
    {
        std::cout << "Layer " << i + 1 << ": ";
        layers[i].__str__(verbose);
    }
}