/**
 * @file perceptronLayer.hpp
 * @author Stan Merlijn
 * @brief In this file the PerceptronLayer class is defined.
 * @version 0.1
 * @date 2025-02-12
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#pragma once
#include "perceptron.hpp"
#include <iostream>
#include <vector>

/**
 * @class PerceptronLayer
 * @brief Represents a layer of perceptrons in a neural network.
 */
class PerceptronLayer 
{
public:
    /**
     * @brief Constructs a perceptron layer.
     * @param neurons List of perceptrons.
     */
    PerceptronLayer(const std::vector<Perceptron>& neurons);

    /**
     * @brief Feeds input forward through the layer.
     * @param input Input vector.
     * @return Output vector after applying all perceptrons.
     */
    std::vector<int> feedForward(const std::vector<int>& input) const;

    /**
     * @brief Prints layer details.
     * @param verbose Verbosity level.
     */
    void __str__(int verbose) const;

private:
    std::vector<Perceptron> neurons; /**< List of perceptrons in the layer. */
};
