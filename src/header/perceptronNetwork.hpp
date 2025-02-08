#pragma once

#include "perceptronLayer.hpp"
#include <iostream>
#include <vector>

class PerceptronNetwork
{
public:
    PerceptronNetwork(std::vector<PerceptronLayer> layers);

    std::vector<double> feed_forward(const std::vector<double>& input) const;
    void __str__(int verbose) const;

private:
    std::vector<PerceptronLayer> layers;
};

