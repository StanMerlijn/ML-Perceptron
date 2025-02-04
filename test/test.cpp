#define CATCH_CONFIG_MAIN
#include "catch_amalgamated.hpp"
#include "../src/header/perceptron.hpp"
#include <iostream>

//2. 
//a. Initialiseer een Perceptron voor elk van de INVERT-, AND- en OR-poorten en test of ze op de juiste manier werken.
//b. Initialiseer een Perceptron voor een NOR-poort met drie ingangen en test of deze op de juiste manier werkt.
//c. Initialiseer ook een Perceptron voor een uitgebreider beslissysteem (minimaal 3 inputs, zie bijvoorbeeld Figuur 2.8 uit de reader) en test of deze naar verwachting werkt.

unsigned int Factorial( unsigned int number ) {
    return number <= 1 ? number : Factorial(number-1)*number;
}

TEST_CASE( "Factorials are computed", "[factorial]" ) {
    REQUIRE( Factorial(1) == 1 );
    REQUIRE( Factorial(2) == 2 );
    REQUIRE( Factorial(3) == 6 );
    REQUIRE( Factorial(10) == 3628800 );
}
