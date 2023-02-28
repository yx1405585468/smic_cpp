cmake_minimum_required(VERSION 3.24)
project(smic_cpp)

set(CMAKE_CXX_STANDARD 14)

add_executable(smic_cpp
        main.cpp ewma.cpp solver.cpp)
