cmake_minimum_required(VERSION 3.25)

project(pymlg VERSION 0.1)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
enable_testing()

add_compile_options(-fdiagnostics-color=always)
add_subdirectory(tests/cpp)
