set(CMAKE_CXX_FLAGS "-std=c++11")

include_directories(${PROJECT_SOURCE_DIR}/include)

include_directories(../../CppHilbert/Library/source/)
include_directories(/usr/include/eigen3)

include_directories(./googletest/include)

# For using the g-test framework
add_subdirectory(googletest)

add_executable(parametricbem2d_tests tests.cpp)
add_executable(convergence convergence_test.cpp)
add_executable(convergence_panels convergence_test_panels.cpp)
#add_executable(wtf wtf.cpp)
add_executable(loog logweight.cpp)

target_link_libraries(parametricbem2d_tests parametrizations gtest adj_double_layer hypersingular single_layer double_layer quadrature CppHilbert)
target_link_libraries(convergence parametrizations gtest adj_double_layer hypersingular single_layer double_layer quadrature CppHilbert)
#target_link_libraries(wtf parametrizations gtest adj_double_layer hypersingular single_layer double_layer quadrature CppHilbert)
target_link_libraries(convergence_panels parametrizations gtest adj_double_layer hypersingular single_layer double_layer quadrature CppHilbert)
target_link_libraries(loog parametrizations gtest adj_double_layer hypersingular single_layer double_layer quadrature CppHilbert)
