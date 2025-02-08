#define CATCH_CONFIG_MAIN
#define EPOCHS 100

#include "catch.hpp"
#include "../src/header/perceptron.hpp"
#include "../src/header/perceptronLayer.hpp"
#include "../src/header/perceptronNetwork.hpp"
#include "../src/header/halfAdder.hpp"
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
 * - Half adder for the XOR gate: Tests the half adder's ability to learn the XOR gate.
 * - PerceptronNetwork for the XOR gate with 2 inputs.
 * - PerceptronNetwork for a half adder. #TODO: iam losing my mind
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

    CHECK(invert_gate.predict(in10) == 0);
    CHECK(invert_gate.predict(in01) == 1);
}

TEST_CASE("Perceptron for AND Gate", "[perceptron]") 
{
    Perceptron and_gate({0.1, 0.1}, 1, 0.1);
    std::vector<int> targets = {0,0,0,1};
    and_gate.train(inputs, targets, EPOCHS);

    CHECK(and_gate.predict(in00) == 0);
    CHECK(and_gate.predict(in01) == 0);
    CHECK(and_gate.predict(in10) == 0);
    CHECK(and_gate.predict(in11) == 1);
}

TEST_CASE("Perceptron for OR Gate", "[perceptron]") 
{
    Perceptron or_gate({0.1, 0.1}, 1, 0.1);
    std::vector<int> targets = {0,1,1,1};
    or_gate.train(inputs, targets, EPOCHS);

    CHECK(or_gate.predict(in00) == 0);
    CHECK(or_gate.predict(in01) == 1);
    CHECK(or_gate.predict(in10) == 1);
    CHECK(or_gate.predict(in11) == 1);
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

    CHECK(norGate.predict(in000) == 1);
    CHECK(norGate.predict(in001) == 0);
    CHECK(norGate.predict(in010) == 0);
    CHECK(norGate.predict(in100) == 0);
    CHECK(norGate.predict(in011) == 0);
    CHECK(norGate.predict(in101) == 0);
    CHECK(norGate.predict(in110) == 0);
    CHECK(norGate.predict(in111) == 0);
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

    CHECK(majorityGate.predict(in000) == 0);
    CHECK(majorityGate.predict(in001) == 0);
    CHECK(majorityGate.predict(in010) == 0);
    CHECK(majorityGate.predict(in100) == 0);
    CHECK(majorityGate.predict(in011) == 1);
    CHECK(majorityGate.predict(in101) == 1);
    CHECK(majorityGate.predict(in110) == 1);
    CHECK(majorityGate.predict(in111) == 1);
}

TEST_CASE("PerceptronLayer for AND and OR Gates", "[perceptronLayer]") {
    // Training data common to both gates:
    std::vector<std::vector<int>> targets = {
        // AND gate targets: only {1,1} should yield 1.
        {0, 0, 0, 1},
        // OR gate targets: all inputs except {0,0} yield 1.
        {0, 1, 1, 1}
    };

    // Create a layer with two neurons (2 inputs) for the AND gate and a learning rate of 0.1.
    // Train the layer with the AND gate targets and OR gate targets.
    PerceptronLayer andLayer(2, 2, 0.1);
    andLayer.train(inputs, targets, EPOCHS);

    // Define expected outputs for the AND gate and OR gate.
    std::vector<int> out00 = {0, 0};
    std::vector<int> out01 = {0, 1};
    std::vector<int> out11 = {1, 1};
    
    CHECK(andLayer.predict(in00) == out00);
    CHECK(andLayer.predict(in01) == out01);
    CHECK(andLayer.predict(in10) == out01);
    CHECK(andLayer.predict(in11) == out11);
}


TEST_CASE("PerceptronNetwork for the XOR gate with 2 inputs", "[perceptronNetwork]") {
    // Create a network with two layers: one for the AND gate and one for the OR gate.
    PerceptronLayer inputLayer(2, 2, 0.1);
    PerceptronLayer outputLayer(1, 2, 0.1);

    // Training data for the XOR gate:
    // Output 1 if inputs are different, else output 0.
    std::vector<std::vector<int>> inputs = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };

    // OR and NAND gates for the input layer
    std::vector<std::vector<int>> targetsInput = {
        {0, 1, 1, 1},
        {1, 1, 1, 0}
    };

    // And gate for the output layer
    std::vector<std::vector<int>> targetsOutput = {
        {0, 0, 0, 1}
    };
    // Train the layers
    inputLayer.train(inputs, targetsInput, EPOCHS);
    outputLayer.train(inputs, targetsOutput, EPOCHS);

    PerceptronNetwork xor_network({inputLayer, outputLayer});

    // Define expected outputs for the XOR gate.
    std::vector<int> out00 = {0};
    std::vector<int> out01 = {1};
    std::vector<int> out10 = {1};
    std::vector<int> out11 = {0};

    // Verify network's predictions for the XOR gate.
    CHECK(xor_network.feed_forward(in00) == out00);
    CHECK(xor_network.feed_forward(in01) == out01);
    CHECK(xor_network.feed_forward(in10) == out10);
    CHECK(xor_network.feed_forward(in11) == out11);
}

TEST_CASE("Half adder", "[halfAdder]") {
    // Create a half adder
    halfAdder halfAdder;

    // Define expected outputs for the XOR gate.
    halfAdderOutput out00 = {0, 0};
    halfAdderOutput out01 = {1, 0};
    halfAdderOutput out10 = {1, 0};
    halfAdderOutput out11 = {0, 1};

    CHECK(halfAdder.predict(in00).sum == out00.sum);
    CHECK(halfAdder.predict(in00).carry == out00.carry);
    CHECK(halfAdder.predict(in01).sum == out01.sum);
    CHECK(halfAdder.predict(in01).carry == out01.carry);
    CHECK(halfAdder.predict(in10).sum == out10.sum);
    CHECK(halfAdder.predict(in10).carry == out10.carry);
    CHECK(halfAdder.predict(in11).sum == out11.sum);
    CHECK(halfAdder.predict(in11).carry == out11.carry);
}

// TEST_CASE("PerceptronNetwork for a half adder", "[perceptronNetwork]") {
//     // Create a network with three layers.
//     PerceptronLayer inputLayer(2, 2, 0.1);
//     PerceptronLayer hiddenLayer(2, 2, 0.1);
//     // Update the output layer to have 2 neurons.
//     PerceptronLayer outputLayer(2, 2, 0.1);

//     // Training data for the half adder:
//     // Inputs: {0, 0}, {0, 1}, {1, 0}, {1, 1}
//     std::vector<std::vector<int>> inputs = {
//         {0, 0}, {0, 1}, {1, 0}, {1, 1}
//     };

//     // For the input layer, we train two neurons:
//     // - First neuron: AND gate: outputs {0, 0, 0, 1}
//     // - Second neuron: OR gate: outputs {0, 1, 1, 1}
//     std::vector<std::vector<int>> targetsInput = {
//         {0, 0, 0, 1},
//         {0, 1, 1, 1}
//     };

//     // For the hidden layer, we train a neuron to compute XOR: outputs {0, 1, 1, 0}
//     std::vector<std::vector<int>> targetsHidden = {
//         {0, 1, 1, 0}
//     };

//     // For the output layer, we now train two neurons:
//     // - First neuron (Sum): XOR: outputs {0, 1, 1, 0}
//     // - Second neuron (Carry): AND: outputs {0, 0, 0, 1}
//     std::vector<std::vector<int>> targetsOutput = {
//         {0, 1, 1, 0},  // Sum (XOR)
//         {0, 0, 0, 1}   // Carry (AND)
//     };

//     // Train the layers with the corresponding targets.
//     inputLayer.train(inputs, targetsInput, EPOCHS);
//     hiddenLayer.train(inputs, targetsHidden, EPOCHS);
//     outputLayer.train(inputs, targetsOutput, EPOCHS);

//     PerceptronNetwork half_adder({inputLayer, hiddenLayer, outputLayer});

//     // Define input vectors.
//     std::vector<int> in00 = {0, 0};
//     std::vector<int> in01 = {0, 1};
//     std::vector<int> in10 = {1, 0};
//     std::vector<int> in11 = {1, 1};

//     // Define expected outputs for the half adder.
//     std::vector<int> out00 = {0, 0};
//     std::vector<int> out10 = {1, 0};
//     std::vector<int> out01 = {0, 1};

//     // Verify network's predictions for the half adder.
//     CHECK(half_adder.feed_forward(in00) == out00);
//     CHECK(half_adder.feed_forward(in01) == out10);
//     CHECK(half_adder.feed_forward(in10) == out10);
//     CHECK(half_adder.feed_forward(in11) == out01);
// }
