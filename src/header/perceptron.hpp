#pragma once
#include <iostream>
#include <vector>

class Perceptron 
{
public:
    Perceptron(std::vector<double> weights, double bias, double learningRate);
    ~Perceptron();

    double predict(const std::vector<double>& x) const;
    void train(const std::vector<std::vector<double>>& x, const std::vector<double>& targets, int epochs);
    void __str__(int verbose);
    
private:
    std::vector<double> weights;
    double bias;
    double learningRate;
};


class SimplePerceptron 
{
public:
    SimplePerceptron(const std::vector<double>& weights, double bias);
    int predict(const std::vector<double>& inputs) const;

private:
    std::vector<double> weights;
    double bias;

};
