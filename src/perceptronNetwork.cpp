#include "header/perceptronNetwork.hpp"
#include "header/perceptronLayer.hpp" 
#include <iostream>

PerceptronNetwork::PerceptronNetwork(std::vector<PerceptronLayer> layers) 
    : layers(layers) {}

std::vector<double> PerceptronNetwork::feed_forward(const std::vector<double>& input) const
{
    std::vector<double> activation = input;
    // Propagate the input through each layer sequentially.
    for (const PerceptronLayer& layer : layers) {
        activation = layer.feed_forward(activation);
    }
    return activation;
}

void PerceptronNetwork::__str__(int verbose) const
{
    std::cout << "Perceptron Network Structure:" << std::endl;
    std::cout << "Number of layers: " << layers.size() << std::endl;
    for (int i = 0; i < layers.size(); ++i)
    {
        std::cout << "Layer " << i + 1 << ": ";
        layers[i].__str__(verbose);
    }
}