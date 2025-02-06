#pragma once
#include <iostream>
#include <vector>

class Perceptron 
{
public:
    Perceptron(std::vector<double> weights, double bias, double learningRate);
    ~Perceptron();

    double predict(std::vector<double>& x);
    void train(std::vector<std::vector<double>>& x, std::vector<double>& y, int epochs);
    void __str__(int verbose);
    
private:
    std::vector<double> weights;
    double bias;
    double learningRate;
};

