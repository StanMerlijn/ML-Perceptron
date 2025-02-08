#define CATCH_CONFIG_MAIN
#define EPOCHS 100

#include "catch.hpp"
#include "../src/header/perceptron.hpp"
#include "../src/header/perceptronLayer.hpp"
#include "../src/header/perceptronNetwork.hpp"
#include <iostream>

/**
 * @file test.cpp
 * @brief Unit tests for the Perceptron, PerceptronLayer and PerceptronNetwork classes.
 *
 * This file contains a series of test cases to verify the functionality of the Perceptron and PerceptronLayer classes.
 * The tests include training and prediction for various logic gates.
 *
 * Test Cases:
 * - Perceptron for INVERT Gate: Tests the perceptron's ability to learn the INVERT gate.
 * - Perceptron for AND Gate: Tests the perceptron's ability to learn the AND gate.
 * - Perceptron for OR Gate: Tests the perceptron's ability to learn the OR gate.
 * - Perceptron for NOR Gate (3 inputs): Tests the perceptron's ability to learn the NOR gate with 3 inputs.
 * - Perceptron for 3-input Majority Gate: Tests the perceptron's ability to learn the 3-input Majority gate.
 * - PerceptronLayer for AND and OR Gates: Tests the PerceptronLayer's ability to learn the AND and OR gates.
 * - PerceptronNetwork for the XOR gate with 2 inputs.
 * - PerceptronNetwork for a half adder.
 * 
 * @note The tests use the Catch2 framework for unit testing.
 */


TEST_CASE("Perceptron for INVERT Gate", "[perceptron]") 
{
    // The perceptron is instantiated with two weights. We use the first weight for the real input
    // and ignore the second input by always passing 0.
    Perceptron invert_gate({0.1, 0.1}, 1, 0.1);

    // Training data: for input 0 we expect output 1, and for input 1 we expect output 0.
    // The second element in the input vector is always 0.
    std::vector<std::vector<double>> x = {{0, 0}, {1, 0}};
    std::vector<double> y = {1, 0};

    // Train the perceptron
    invert_gate.train(x, y, EPOCHS);
    
    std::vector<double> in1 = {1, 0};
    std::vector<double> in0 = {0, 0};

    CHECK(invert_gate.predict(in1) == 0);
    CHECK(invert_gate.predict(in0) == 1);
}

TEST_CASE("Perceptron for AND Gate", "[perceptron]") 
{
    Perceptron and_gate({0.1, 0.1}, 1, 0.1);

    // Initialize the inputs and output expected for an AND gate
    std::vector<std::vector<double>> x = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<double> y = {0,0,0,1};

    // Train the perceptron
    and_gate.train(x, y, EPOCHS);

    // Test the possible inputs for an AND gate
    std::vector<double> in00 = {0, 0};
    std::vector<double> in01 = {0, 1};
    std::vector<double> in10 = {1, 0};
    std::vector<double> in11 = {1, 1};

    CHECK(and_gate.predict(in00) == 0);
    CHECK(and_gate.predict(in01) == 0);
    CHECK(and_gate.predict(in10) == 0);
    CHECK(and_gate.predict(in11) == 1);
}

TEST_CASE("Perceptron for OR Gate", "[perceptron]") 
{
    Perceptron or_gate({0.1, 0.1}, 1, 0.1);

    // Initialize the inputs and output expected for an OR gate
    std::vector<std::vector<double>> x = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<double> y = {0,1,1,1};

    // Train the perceptron
    or_gate.train(x, y, EPOCHS);

    // Test the possible inputs for an OR gate
    std::vector<double> in00 = {0, 0};
    std::vector<double> in01 = {0, 1};
    std::vector<double> in10 = {1, 0};
    std::vector<double> in11 = {1, 1};

    CHECK(or_gate.predict(in00) == 0);
    CHECK(or_gate.predict(in01) == 1);
    CHECK(or_gate.predict(in10) == 1);
    CHECK(or_gate.predict(in11) == 1);
}

TEST_CASE("Perceptron for NOR Gate (3 inputs)", "[perceptron]") {
    // Instantiate the perceptron with three weights.
    Perceptron nor_gate({-0.1, -0.1, -0.1}, 1, 0.1);

    // Training data for a NOR gate with 3 inputs:
    // Only (0,0,0) should yield 1; all others yield 0.
    std::vector<std::vector<double>> x = {
        {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {1, 0, 0},
        {1, 1, 0}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}
    };
    std::vector<double> y = {1, 0, 0, 0, 0, 0, 0, 0};

    // Train the perceptron
    nor_gate.train(x, y, EPOCHS);

    std::vector<double> in000 = {0, 0, 0};
    std::vector<double> in100 = {1, 0, 0};
    std::vector<double> in010 = {0, 1, 0};
    std::vector<double> in001 = {0, 0, 1};
    std::vector<double> in111 = {1, 1, 1};

    // Test various cases
    CHECK(nor_gate.predict(in000) == 1);
    CHECK(nor_gate.predict(in100) == 0);
    CHECK(nor_gate.predict(in010) == 0);
    CHECK(nor_gate.predict(in001) == 0);
    CHECK(nor_gate.predict(in111) == 0);
}

TEST_CASE("Perceptron for 3-input Majority Gate", "[perceptron]") {
    // Instantiate the perceptron with three inputs. Here we choose small positive initial weights
    // and a negative bias. Adjust these parameters if necessary to speed up convergence.
    Perceptron majority_gate({0.1, 0.1, 0.1}, -0.2, 0.1);

    // Training data for a majority gate:
    // Output 1 if at least two inputs are 1, else output 0.
    std::vector<std::vector<double>> x = {
        {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {1, 0, 0}, 
        {0, 1, 1}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}  
    };
    std::vector<double> y = {0, 0, 0, 0, 1, 1, 1, 1};

    // Train the perceptron
    majority_gate.train(x, y, EPOCHS);

    // Define input vectors as done in other tests
    std::vector<double> in000 = {0, 0, 0};
    std::vector<double> in001 = {0, 0, 1};
    std::vector<double> in010 = {0, 1, 0};
    std::vector<double> in100 = {1, 0, 0};
    std::vector<double> in011 = {0, 1, 1};
    std::vector<double> in101 = {1, 0, 1};
    std::vector<double> in110 = {1, 1, 0};
    std::vector<double> in111 = {1, 1, 1};

    CHECK(majority_gate.predict(in000) == 0);
    CHECK(majority_gate.predict(in001) == 0);
    CHECK(majority_gate.predict(in010) == 0);
    CHECK(majority_gate.predict(in100) == 0);
    CHECK(majority_gate.predict(in011) == 1);
    CHECK(majority_gate.predict(in101) == 1);
    CHECK(majority_gate.predict(in110) == 1);
    CHECK(majority_gate.predict(in111) == 1);
}

TEST_CASE("PerceptronLayer for AND and OR Gates", "[perceptronLayer]") {
    // Training data common to both gates:
    std::vector<std::vector<double>> inputs = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };
    
    std::vector<std::vector<double>> targets = {
        // AND gate targets: only {1,1} should yield 1.
        {0, 0, 0, 1},
        // OR gate targets: all inputs except {0,0} yield 1.
        {0, 1, 1, 1}
    };

    // Create a layer with two neurons (2 inputs) for the AND gate and a learning rate of 0.1.
    // Train the layer with the AND gate targets and OR gate targets.
    PerceptronLayer and_layer(2, 2, 0.1);
    and_layer.train(inputs, targets, EPOCHS);
    
    // Define input vectors.
    std::vector<double> in00 = {0, 0};
    std::vector<double> in01 = {0, 1};
    std::vector<double> in10 = {1, 0};
    std::vector<double> in11 = {1, 1};

    // Define expected outputs for the AND gate and OR gate.
    std::vector<double> out00 = {0, 0};
    std::vector<double> out01 = {0, 1};
    std::vector<double> out11 = {1, 1};
    
    // Verify layer's predictions for the AND/ OR gates.
    CHECK(and_layer.feed_forward(in00) == out00);
    CHECK(and_layer.feed_forward(in01) == out01);
    CHECK(and_layer.feed_forward(in10) == out01);
    CHECK(and_layer.feed_forward(in11) == out11);
}

TEST_CASE("PerceptronNetwork for the XOR gate with 2 inputs", "[perceptronNetwork]") {
    // Create a network with two layers: one for the AND gate and one for the OR gate.
    PerceptronLayer inputLayer(2, 2, 0.1);
    PerceptronLayer outputLayer(1, 2, 0.1);

    // Training data for the XOR gate:
    // Output 1 if inputs are different, else output 0.
    std::vector<std::vector<double>> inputs = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };

    // OR and NAND gates for the input layer
    std::vector<std::vector<double>> targetsInput = {
        {0, 1, 1, 1},
        {1, 1, 1, 0}
    };

    // And gate for the output layer
    std::vector<std::vector<double>> targetsOutput = {
        {0, 0, 0, 1}
    };
    // Train the layers
    inputLayer.train(inputs, targetsInput, EPOCHS);
    outputLayer.train(inputs, targetsOutput, EPOCHS);

    PerceptronNetwork xor_network({inputLayer, outputLayer});

    // Define input vectors.
    std::vector<double> in00 = {0, 0};
    std::vector<double> in01 = {0, 1};
    std::vector<double> in10 = {1, 0};
    std::vector<double> in11 = {1, 1};

    // Define expected outputs for the XOR gate.
    std::vector<double> out00 = {0};
    std::vector<double> out01 = {1};
    std::vector<double> out10 = {1};
    std::vector<double> out11 = {0};

    // Verify network's predictions for the XOR gate.
    CHECK(xor_network.feed_forward(in00) == out00);
    CHECK(xor_network.feed_forward(in01) == out01);
    CHECK(xor_network.feed_forward(in10) == out10);
    CHECK(xor_network.feed_forward(in11) == out11);
}