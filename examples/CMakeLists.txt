include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/usr/include/eigen3)

set(CMAKE_CXX_FLAGS "-std=c++11")

add_executable(log log.cpp)
add_executable(single_layer_test single_layer_circle_test.cpp)
add_executable(dirichlet_example dirichlet_example.cpp)
add_executable(annulardvp annular_dvp.cpp)

target_link_libraries(single_layer_test single_layer parametrizations quadrature)
target_link_libraries(log quadrature)
target_link_libraries(dirichlet_example parametrizations gtest adj_double_layer hypersingular single_layer double_layer quadrature CppHilbert)
target_link_libraries(annulardvp parametrizations gtest adj_double_layer hypersingular single_layer double_layer quadrature CppHilbert)
