include_directories(${PROJECT_SOURCE_DIR}/include)

include_directories(../../../CppHilbert/Library/source/)

FILE(GLOB QuadratureSources ./*.cpp)

add_library(quadrature STATIC ${QuadratureSources})



