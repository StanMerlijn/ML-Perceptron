#pragma once
#include <iostream>
#include <vector>

class Perceptron 
{
public:
    Perceptron(std::vector<double> weights, double bias, double learningRate);

    int predict(const std::vector<int>& x) const;
    void train(const std::vector<std::vector<int>>& inputs, const std::vector<int>& targets, int epochs);
    void __str__(int verbose) const;

private:
    std::vector<double> weights;
    double bias;
    double learningRate;
};
