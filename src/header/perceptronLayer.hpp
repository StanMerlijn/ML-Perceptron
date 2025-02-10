#pragma once
#include "perceptron.hpp"
#include <iostream>
#include <vector>

class PerceptronLayer
{
public:
    PerceptronLayer(const std::vector<Perceptron>& neurons);
    std::vector<int> predict(const std::vector<int>& input) const;    
    void __str__(int verbose) const;

private:
    std::vector<Perceptron> neurons;
};