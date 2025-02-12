/**
 * @file perceptronNetwork.hpp
 * @author Stan Merlijn
 * @brief In this file the PerceptronNetwork class is defined.
 * @version 0.1
 * @date 2025-02-12
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#pragma once
#include "perceptronLayer.hpp"
#include <iostream>
#include <vector>

/**
 * @class PerceptronNetwork
 * @brief Represents a multi-layer perceptron network.
 */
class PerceptronNetwork 
{   
public:
    /**
     * @brief Constructs a perceptron network.
     * @param layers List of perceptron layers.
     */
    PerceptronNetwork(std::vector<PerceptronLayer> layers);

    /**
     * @brief Feeds input forward through the network.
     * @param input Input vector.
     * @return Output vector after processing through all layers.
     */
    std::vector<int> feedForward(const std::vector<int>& input) const;

    /**
     * @brief Prints network details.
     * @param verbose Verbosity level.
     */
    void __str__(int verbose) const;

private:
    std::vector<PerceptronLayer> layers; /**< Layers of the network. */

};