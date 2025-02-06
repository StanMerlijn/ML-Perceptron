#pragma once
#include <iostream>
#include <vector>

class Perceptron 
{
public:
    Perceptron(std::vector<double> weights, double bias, double learningRate);
    ~Perceptron();

    double predict(const std::vector<double>& x) const;
    void train(const std::vector<std::vector<double>>& x, const std::vector<double>& y, int epochs);
    void __str__(int verbose);
    
private:
    std::vector<double> weights;
    double bias;
    double learningRate;
};



