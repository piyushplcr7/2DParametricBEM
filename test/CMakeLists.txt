set(CMAKE_CXX_FLAGS "-std=c++11")

include_directories(${PROJECT_SOURCE_DIR}/include)

include_directories(../../CppHilbert/Library/source/)

include_directories(./googletest/include)

include_directories(/usr/include/eigen3)

#add_subdirectory(../../CppHilbert/Library ${CMAKE_BINARY_DIR}/CppHilbert)

#link_directories(${CMAKE_BINARY_DIR}/CppHilbert)

# For using the g-test framework
add_subdirectory(googletest)

add_executable(parametricbem2d_tests tests.cpp)
add_executable(convergence convergence_test.cpp)

target_link_libraries(parametricbem2d_tests parametrizations gtest adj_double_layer hypersingular single_layer double_layer quadrature CppHilbert)
target_link_libraries(convergence parametrizations gtest adj_double_layer hypersingular single_layer double_layer quadrature CppHilbert)
