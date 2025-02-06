cmake_minimum_required(VERSION 3.10)

# Set the project name
project(MLPerceptron)

# Set the C++ standard to C++11
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(MLPerceptronTest test.cpp ../src/perceptron.cpp ../src/perceptronLayer.cpp)
