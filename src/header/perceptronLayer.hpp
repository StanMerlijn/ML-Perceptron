#pragma once
#include "perceptron.hpp"
#include <iostream>
#include <vector>

class PerceptronLayer
{
public:
    PerceptronLayer(int numNeurons, int inputDimension, double learningRate);

    std::vector<double> predict(const std::vector<double>& input) const;    
    void train(const std::vector<std::vector<double>>& x, const std::vector<double>& targets, int epochs);
    void __str__(int verbose) const;

private:
    std::vector<Perceptron> neurons;
};