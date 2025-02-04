cmake_minimum_required(VERSION 3.10)

# Set the project name
project(MLPerceptron)
set(CMAKE_CXX_STANDARD 17)

# Collect all source files in the src directory
file(GLOB SOURCES "src/*.cpp")

# Create the executable
add_executable(MLPerceptron ${SOURCES})

