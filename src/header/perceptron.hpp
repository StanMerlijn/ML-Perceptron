/**
 * @file perceptron.hpp
 * @author Stan Merlijn
 * @brief In this file the Perceptron class is defined. 
 * @version 0.1
 * @date 2025-02-12
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once
#include <iostream>
#include <vector>

/**
 * @class Perceptron
 * @brief A simple perceptron model for binary classification.
 */
class Perceptron 
{    
public:
    /**
     * @brief Constructs a Perceptron with given weights, bias, and learning rate.
     * @param weights Initial weights.
     * @param bias Initial bias.
     * @param learningRate Learning rate for training.
     */
    Perceptron(std::vector<double> weights, double bias, double learningRate);

    /**
     * @brief Predicts the output for a given input vector.
     * @param inputs Input vector.
     * @return 1 if activated, otherwise 0.
     */
    int predict(const std::vector<int>& inputs) const;

    /**
     * @brief Trains the perceptron using the given dataset. Using th learning rule to update the weights.
     * @param inputs Input samples.
     * @param targets Target outputs.
     * @param epochs Number of training iterations.
     */
    void train(const std::vector<std::vector<int>>& inputs, const std::vector<int>& targets, int epochs);

    /**
     * @brief Prints perceptron details.
     * @param verbose Verbosity level.
     */
    void __str__(int verbose) const;

private:
        std::vector<double> weights; /**< Weights for the perceptron. */
        double bias; /**< Bias term. */
        double learningRate; /**< Learning rate for weight updates. */ 
};
