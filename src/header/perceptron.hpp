#pragma once
#include <iostream>
#include <vector>

class Perceptron 
{
public:
    Perceptron(std::vector<double> weights, double bias, double learningRate);
    ~Perceptron();
    double output(std::vector<double> inputs);
    void __str__(int verbose);
    
private:
    std::vector<double> weights;
    double bias;
    double learningRate;
};

