#pragma once
#include "perceptron.hpp"
#include <iostream>
#include <vector>

class PerceptronLayer
{
public:
    PerceptronLayer(int numNeurons, int inputDimension, double learningRate);

    std::vector<int> predict(const std::vector<int>& input) const;    
    void train(const std::vector<std::vector<int>>& inputs, 
        const std::vector<std::vector<int>>& targets, 
        int epochs);
    
    void __str__(int verbose) const;

private:
    std::vector<Perceptron> neurons;
};