#pragma once

#include "perceptronLayer.hpp"
#include <iostream>
#include <vector>

class PerceptronNetwork
{
public:
    PerceptronNetwork(std::vector<PerceptronLayer> layers);

    std::vector<int> feedForward(const std::vector<int>& input) const;
    void __str__(int verbose) const;

private:
    std::vector<PerceptronLayer> layers;
};

