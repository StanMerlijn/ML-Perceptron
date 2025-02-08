#include "perceptron.hpp"
#include <iostream>
#include <vector>

#define EPOCHS 100

struct halfAdderOutput
{
    int sum;
    int carry;
};

class halfAdder
{
private:
    // Perceptrons for the half adder
    Perceptron andGate;
    Perceptron orGate;
    Perceptron nanGate;
    
public:
    halfAdder() 
        : andGate({0.1, 0.1}, 1, 0.1),
        orGate({0.1, 0.1}, 1, 0.1),
        nanGate({0.1, 0.1}, 1, 0.1)
    {
        // Training data for the half adder
        std::vector<std::vector<int>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        std::vector<int> targetsAnd  = {0, 0, 0, 1};
        std::vector<int> targetsOr   = {0, 1, 1, 1};
        std::vector<int> targetsNand = {1, 1, 1, 0};
        
        // Train the perceptrons
        andGate.train(inputs, targetsAnd, EPOCHS);
        orGate.train(inputs, targetsOr, EPOCHS);
        nanGate.train(inputs, targetsNand, EPOCHS);
}


    halfAdderOutput predict(const std::vector<int>& x) const
    {
        halfAdderOutput output;
        output.sum = andGate.predict({orGate.predict(x), nanGate.predict(x)});
        output.carry = andGate.predict(x);
        return output;
    }

    void __str__(int verbose) const 
    {
        std::cout << "Half Adder Structure:" << std::endl;
        std::cout << "AND Gate: ";
        andGate.__str__(verbose);
        std::cout << "OR Gate: ";
        orGate.__str__(verbose);
        std::cout << "NAND Gate: ";
        nanGate.__str__(verbose);
    }

};
