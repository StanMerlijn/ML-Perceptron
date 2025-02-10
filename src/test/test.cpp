#define CATCH_CONFIG_MAIN
#define EPOCHS 100

#include "catch.hpp"
#include "../header/perceptron.hpp"
#include "../header/perceptronLayer.hpp"
#include "../header/perceptronNetwork.hpp"
#include "../header/halfAdder.hpp"

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

// Define the input vectors for the logic gates
std::vector<std::vector<int>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

// Define inputs for 2-input gates
std::vector<int> in00 = {0, 0};
std::vector<int> in01 = {0, 1};
std::vector<int> in10 = {1, 0};
std::vector<int> in11 = {1, 1};

// Define inputs for 3-input gates
std::vector<int> in000 = {0, 0, 0};
std::vector<int> in001 = {0, 0, 1};
std::vector<int> in010 = {0, 1, 0};
std::vector<int> in100 = {1, 0, 0};
std::vector<int> in011 = {0, 1, 1};
std::vector<int> in101 = {1, 0, 1};
std::vector<int> in110 = {1, 1, 0};
std::vector<int> in111 = {1, 1, 1};

TEST_CASE("Perceptron for INVERT Gate", "[perceptron]") 
{
    Perceptron invert_gate({0.1, 0.1}, 1, 0.1);

    // Training data: for input 0 we expect output 1, and for input 1 we expect output 0.
    // The second element in the input vector is always 0.
    std::vector<std::vector<int>> inputsInverter = {{0, 0}, {1, 0}};
    std::vector<int> targets = {1, 0};
    invert_gate.train(inputsInverter, targets, EPOCHS);

    REQUIRE(invert_gate.predict(in10) == 0);
    REQUIRE(invert_gate.predict(in01) == 1);
}

TEST_CASE("Perceptron for AND Gate", "[perceptron]") 
{
    Perceptron p_and({0.1, 0.1}, 1, 0.1);
    std::vector<int> targets = {0,0,0,1};
    p_and.train(inputs, targets, EPOCHS);

    REQUIRE(p_and.predict(in00) == 0);
    REQUIRE(p_and.predict(in01) == 0);
    REQUIRE(p_and.predict(in10) == 0);
    REQUIRE(p_and.predict(in11) == 1);
}

TEST_CASE("Perceptron for OR Gate", "[perceptron]") 
{
    Perceptron p_or({0.1, 0.1}, 1, 0.1);
    std::vector<int> targets = {0,1,1,1};
    p_or.train(inputs, targets, EPOCHS);

    REQUIRE(p_or.predict(in00) == 0);
    REQUIRE(p_or.predict(in01) == 1);
    REQUIRE(p_or.predict(in10) == 1);
    REQUIRE(p_or.predict(in11) == 1);
}

TEST_CASE("Perceptron for NOR Gate (3 inputs)", "[perceptron]") {
    // Instantiate the perceptron with three weights.
    Perceptron norGate({-0.1, -0.1, -0.1}, 1, 0.1);

    // Training data for a NOR gate with 3 inputs:
    // Only (0,0,0) should yield 1; all others yield 0.
    std::vector<std::vector<int>> inputsNOR = {
        {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {1, 0, 0},
        {1, 1, 0}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}
    };
    std::vector<int> targets = {1, 0, 0, 0, 0, 0, 0, 0};
    norGate.train(inputsNOR, targets, EPOCHS);

    REQUIRE(norGate.predict(in000) == 1);
    REQUIRE(norGate.predict(in001) == 0);
    REQUIRE(norGate.predict(in010) == 0);
    REQUIRE(norGate.predict(in100) == 0);
    REQUIRE(norGate.predict(in011) == 0);
    REQUIRE(norGate.predict(in101) == 0);
    REQUIRE(norGate.predict(in110) == 0);
    REQUIRE(norGate.predict(in111) == 0);
}

TEST_CASE("Perceptron for 3-input Majority Gate", "[perceptron]") {
    // Instantiate the perceptron with three inputs. Here we choose small positive initial weights
    // and a negative bias. Adjust these parameters if necessary to speed up convergence.
    Perceptron majorityGate({0.1, 0.1, 0.1}, -0.2, 0.1);

    // Training data for a majority gate:
    // Output 1 if at least two inputs are 1, else output 0.
    std::vector<std::vector<int>> inputsMajority = {
        {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {1, 0, 0}, 
        {0, 1, 1}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}  
    };
    std::vector<int> y = {0, 0, 0, 0, 1, 1, 1, 1};
    majorityGate.train(inputsMajority, y, EPOCHS);

    REQUIRE(majorityGate.predict(in000) == 0);
    REQUIRE(majorityGate.predict(in001) == 0);
    REQUIRE(majorityGate.predict(in010) == 0);
    REQUIRE(majorityGate.predict(in100) == 0);
    REQUIRE(majorityGate.predict(in011) == 1);
    REQUIRE(majorityGate.predict(in101) == 1);
    REQUIRE(majorityGate.predict(in110) == 1);
    REQUIRE(majorityGate.predict(in111) == 1);
}

TEST_CASE("PerceptronLayer for AND and OR Gates", "[perceptronLayer]") {
    // Training data common to both gates:
    Perceptron p_or({0.1, 0.1}, 1, 0.1);
    Perceptron p_and({0.1, 0.1}, 1, 0.1);

    // Training the OR and AND gates.
    p_or.train(inputs, {0, 1, 1, 1}, EPOCHS);
    p_and.train(inputs, {0, 0, 0, 1}, EPOCHS);

    // Create a layer with two neurons (2 inputs) for the AND gate and a learning rate of 0.1.
    // Train the layer with the AND gate targets and OR gate targets.
    PerceptronLayer andLayer({p_and, p_or});

    // Define expected outputs for the AND gate and OR gate.
    std::vector<int> out00 = {0, 0};
    std::vector<int> out01 = {0, 1};
    std::vector<int> out11 = {1, 1};
    
    REQUIRE(andLayer.feedForward(in00) == out00);
    REQUIRE(andLayer.feedForward(in01) == out01);
    REQUIRE(andLayer.feedForward(in10) == out01);
    REQUIRE(andLayer.feedForward(in11) == out11);
}

TEST_CASE("PerceptronNetwork for the XOR gate with 2 inputs", "[perceptronNetwork]") {
    // Create a network with two layers: one for the AND gate and one for the OR gate.
    // OR and NAND gates for the input layer
    Perceptron p_or({0.1, 0.1}, 1, 0.1);
    Perceptron p_nand({0.1, 0.1}, 1, 0.1);
    Perceptron p_and({0.1, 0.1}, 1, 0.1);

    // Training The gates
    p_or.train(inputs,  {0, 1, 1, 1} , EPOCHS);
    p_nand.train(inputs, {1, 1, 1, 0}, EPOCHS);
    p_and.train(inputs,  {0, 0, 0, 1} , EPOCHS);

    PerceptronLayer inputLayer({p_or, p_nand});
    PerceptronLayer outputLayer({p_and});

    PerceptronNetwork xor_network({inputLayer, outputLayer});

    // Define expected outputs for the XOR gate.
    std::vector<int> out00 = {0};
    std::vector<int> out01 = {1};
    std::vector<int> out10 = {1};
    std::vector<int> out11 = {0};

    // Verify network's predictions for the XOR gate.
    REQUIRE(xor_network.feedForward(in00) == out00);
    REQUIRE(xor_network.feedForward(in01) == out01);
    REQUIRE(xor_network.feedForward(in10) == out10);
    REQUIRE(xor_network.feedForward(in11) == out11);
}

TEST_CASE("PerceptronNetwork for half adder", "[perceptronNetwork]")
{
    // Hidden layer: compute OR and AND
    Perceptron n_or({0.1, 0.1}, 0.1, 0.1);
    Perceptron n_and({0.1, 0.1}, 0.1, 0.1);
    
    n_or.train(inputs, {0, 1, 1, 1}, EPOCHS);
    n_and.train(inputs, {0, 0, 0, 1}, EPOCHS);

    PerceptronLayer hiddenLayer({n_or, n_and});

    // Output layer: compute XOR (for sum) and carry
    // XOR = OR - AND; Perceptron with weights {1, -1} and bias -0.5
    // x1 = 0, x2 = 0: 1 * 0 + -1 * 0 - 0.5 = -0.5
    // x1 = 1, x2 = 0: 1 * 1 + -1 * 0 - 0.5 = 0.5
    // x1 = 1, x2 = 1: 1 * 1 + -1 * 1 - 0.5 = -0.5
    
    Perceptron n_xor({0.1, 0.1}, 0.1, 0.1);
    Perceptron n_carry({0.1, 0.1}, 0.1, 0.1);

    n_xor.train({{0, 0}, {1, 0}, {1, 1}}, {0, 1, 0}, EPOCHS);
    n_carry.train(inputs, {0, 0, 0, 1}, EPOCHS);

    PerceptronLayer outputLayer({n_xor, n_carry});

    PerceptronNetwork halfAdder({hiddenLayer, outputLayer});

    // Test cases for half adder: {Sum, Carry}
    REQUIRE(halfAdder.feedForward({0, 0}) == std::vector<int>{0, 0});
    REQUIRE(halfAdder.feedForward({0, 1}) == std::vector<int>{1, 0});
    REQUIRE(halfAdder.feedForward({1, 0}) == std::vector<int>{1, 0});
    REQUIRE(halfAdder.feedForward({1, 1}) == std::vector<int>{0, 1});
}