#define CATCH_CONFIG_MAIN
#define EPOCHS 100

// #include "catch_amalgamated.hpp"
#include "catch.hpp"
#include "../src/header/perceptron.hpp"
#include <iostream>

//2. 
//a. Initialiseer een Perceptron voor elk van de INVERT-, AND- en OR-poorten en test of ze op de juiste manier werken.
//b. Initialiseer een Perceptron voor een NOR-poort met drie ingangen en test of deze op de juiste manier werkt.
//c. Initialiseer ook een Perceptron voor een uitgebreider beslissysteem (minimaal 3 inputs, zie bijvoorbeeld Figuur 2.8 uit de reader) en test of deze naar verwachting werkt.

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

    // Initialize the inputs and outpout expected for an AND gate
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

    // Initialize the inputs and outpout expected for an OR gate
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